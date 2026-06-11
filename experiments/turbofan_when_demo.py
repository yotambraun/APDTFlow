#!/usr/bin/env python
"""C-MAPSS FD001 turbofan: predict WHEN sensor s11 crosses the maintenance threshold.

Second canonical NASA prognostics benchmark (Section 10.6 of the v0.4.0
methodology). Sensor s11 (HPC outlet static pressure, rolling-5 smoothed) is
the degradation indicator. The maintenance threshold is defined on TRAINING
engines only: the median s11 value 15 cycles before failure. APDTFlow
(continuous decoder + conformal calibration) is fitted on the training
engines' s11 series and audited on sliding windows of UNSEEN engines:

  * train engines 1-60, audit engines 61-100, horizon 40 cycles
  * full-event-set timing MAE vs persistence and linear extrapolation
    (a censored "no crossing" answer is scored at the horizon)
  * event catch rate, false-alarm rate on true no-crossing windows
  * calibrated 90%-window coverage (alpha=0.1, asymmetric signed-error
    crossing-time calibration)
  * the honest nuance: linear extrapolation on the matched subset where it
    also yields an estimate

Outputs:
  * experiments/results/turbofan_when.json
  * assets/images/apdtflow_turbofan_when.png (hero panel, held-out engine)

Usage:
  python experiments/turbofan_when_demo.py                # full run
  python experiments/turbofan_when_demo.py --epochs 1 --train-engines 6 \
      --max-engines 3 --stride 15                         # smoke run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from apdtflow import APDTFlowForecaster, set_seed  # noqa: E402
from apdtflow.event_time import first_crossing_time  # noqa: E402

warnings.filterwarnings("ignore")
for _name in ("apdtflow.forecaster", "apdtflow.conformal"):
    logging.getLogger(_name).setLevel(logging.WARNING)

DATA_FILE = REPO / "dataset_examples" / "cmapss" / "train_FD001.txt"
RESULTS_DIR = REPO / "experiments" / "results"
IMAGES_DIR = REPO / "assets" / "images"

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR = "s11"
HISTORY = 30
HORIZON = 40
ALPHA = 0.1
SEED = 4

GREEN, PURPLE, RED, BLUE = "#2e8b57", "#6a5acd", "#d62839", "#4477aa"


def load_series() -> dict:
    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: missing C-MAPSS data file {DATA_FILE.relative_to(REPO)}\n"
            "Download it first:  python dataset_examples/get_cmapss.py"
        )
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    return {
        int(u): g[SENSOR].rolling(5, min_periods=1).mean().to_numpy(dtype=float)
        for u, g in df.groupby("unit")
    }


def fit_forecaster(series: dict, train_units: list, epochs: int) -> APDTFlowForecaster:
    """Fit on the concatenation of the training engines' s11 series."""
    df = pd.DataFrame({SENSOR: np.concatenate([series[u] for u in train_units])})
    forecaster = APDTFlowForecaster(
        forecast_horizon=HORIZON,
        history_length=HISTORY,
        num_epochs=epochs,
        batch_size=64,
        hidden_dim=32,
        num_scales=3,
        learning_rate=1e-3,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    forecaster.fit(df, target_col=SENSOR)
    return forecaster


def linear_eta(window: np.ndarray, threshold: float) -> float | None:
    t = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(t, window, 1)
    if slope <= 0:  # threshold crossing is upward on FD001
        return None
    anchor_value = intercept + slope * (len(window) - 1)
    tau = (threshold - anchor_value) / slope
    return float(tau) if 0 < tau <= HORIZON else None


def audit(forecaster, series: dict, test_units: list, threshold: float,
          stride: int) -> list:
    grid = np.arange(0.0, HORIZON + 1)
    records, windows = [], []
    for u in test_units:
        z = series[u]
        for i in range(0, len(z) - HISTORY - HORIZON + 1, stride):
            window = z[i:i + HISTORY]
            anchor = window[-1]
            future = z[i + HISTORY:i + HISTORY + HORIZON]
            t_act = first_crossing_time(
                grid, np.concatenate([[anchor], future]), threshold, "above")
            records.append({
                "unit": u, "origin": i, "anchor": float(anchor), "t_act": t_act,
                "t_lin": linear_eta(window, threshold),
                "t_per": 0.0 if anchor >= threshold else None,
            })
            windows.append(window)

    x_norm = (np.asarray(windows) - forecaster.scaler_mean_) / forecaster.scaler_std_
    t_pred = forecaster._batch_crossing_times(
        torch.tensor(x_norm[:, None, :], dtype=torch.float32),
        None, threshold, "above")
    for record, tp in zip(records, t_pred):
        record["t_pred"] = None if np.isnan(tp) else float(tp)
    return records


def summarize(records: list, lo: float, hi: float) -> dict:
    events = [r for r in records if r["t_act"] is not None and r["t_act"] > 0]
    no_cross = [r for r in records if r["t_act"] is None]

    def full_mae(key):
        if not events:
            return None
        return float(np.mean([abs((r[key] if r[key] is not None else HORIZON)
                                  - r["t_act"]) for r in events]))

    caught = [r for r in events if r["t_pred"] is not None]
    coverage = None
    if caught and np.isfinite(lo) and np.isfinite(hi):
        coverage = float(np.mean([
            np.clip(r["t_pred"] - hi, 0, HORIZON)
            <= r["t_act"]
            <= np.clip(r["t_pred"] - lo, 0, HORIZON)
            for r in caught
        ]))
    false_alarms = [r for r in no_cross if r["t_pred"] is not None]
    matched = [r for r in events if r["t_pred"] is not None and r["t_lin"] is not None]
    return {
        "n_windows": len(records),
        "n_events": len(events),
        "n_no_crossing": len(no_cross),
        "mae_full_apdtflow": full_mae("t_pred"),
        "mae_full_linear": full_mae("t_lin"),
        "mae_full_persistence": full_mae("t_per"),
        "mae_caught": (float(np.mean([abs(r["t_pred"] - r["t_act"]) for r in caught]))
                       if caught else None),
        "catch_rate": (len(caught) / len(events)) if events else None,
        "n_false_alarms": len(false_alarms),
        "false_alarm_rate": (len(false_alarms) / len(no_cross)) if no_cross else None,
        "coverage": coverage,
        "matched_subset_n": len(matched),
        "matched_apdtflow_mae": (float(np.mean([abs(r["t_pred"] - r["t_act"])
                                                for r in matched])) if matched else None),
        "matched_linear_mae": (float(np.mean([abs(r["t_lin"] - r["t_act"])
                                              for r in matched])) if matched else None),
    }


def dense_trajectory(forecaster, window: np.ndarray, n_points: int = 240):
    taus = np.linspace(HORIZON / n_points, HORIZON, n_points)
    win_norm = (window - forecaster.scaler_mean_) / forecaster.scaler_std_
    x = torch.tensor(win_norm[None, None, :], dtype=torch.float32,
                     device=forecaster.device)
    t_span = torch.linspace(0, 1, steps=HISTORY, device=forecaster.device)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=forecaster.device)
    forecaster.model.eval()
    with torch.no_grad():
        vals, _ = forecaster.model.forward_at(x, t_span, taus_t)
    traj = vals.squeeze(-1).cpu().numpy()[0]
    return taus, traj * forecaster.scaler_std_ + forecaster.scaler_mean_


def plot_hero(forecaster, series, record, threshold, lo, hi, path: Path):
    z = series[record["unit"]]
    origin = record["origin"]
    hist_x = np.arange(-(HISTORY - 1), 1)
    hist_y = z[origin:origin + HISTORY]
    future = z[origin + HISTORY:origin + HISTORY + HORIZON]
    taus, traj = dense_trajectory(forecaster, z[origin:origin + HISTORY])

    eta = record["t_pred"]
    earliest = float(np.clip(eta - hi, 0, HORIZON)) if np.isfinite(hi) else eta
    latest = float(np.clip(eta - lo, 0, HORIZON)) if np.isfinite(lo) else HORIZON
    t_act = record["t_act"]

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.plot(hist_x, hist_y, "-o", color=BLUE, ms=4, lw=1.6,
            label=f"Sensor s11 — engine #{record['unit']} (real, never in training)")
    ax.plot(np.arange(1, len(future) + 1), future, "o", color="black", ms=5,
            label="Actual future")
    ax.plot(taus, traj, color=GREEN, lw=2.4, label="APDTFlow continuous forecast")
    ax.axhline(threshold, color=RED, ls="--", lw=1.8, label="Maintenance threshold")
    ax.axvspan(earliest, latest, color=RED, alpha=0.12,
               label="90% crossing-time window")
    ax.axvline(eta, color=RED, lw=2.4)
    ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.plot([t_act], [threshold], "*", color="black", ms=24,
            label=f"Actual crossing: cycle +{t_act:.0f}")
    ax.annotate(f"predict_when = +{eta:.0f} cycles", xy=(eta, threshold),
                xytext=(eta + 1.0, threshold + 0.25 * np.std(future)),
                color=RED, fontsize=12)
    ax.set_xlabel("flight cycles from forecast origin")
    ax.set_ylabel("s11 — HPC static pressure")
    ax.set_title("NASA C-MAPSS turbofan degradation (held-out engine)\n"
                 "'When does this engine cross the maintenance threshold?'",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=25,
                        help="training epochs (default 25)")
    parser.add_argument("--stride", type=int, default=3,
                        help="audit window stride (default 3)")
    parser.add_argument("--max-engines", type=int, default=40,
                        help="number of audit engines from 61.. (default 40)")
    parser.add_argument("--train-engines", type=int, default=60,
                        help="number of training engines from 1.. (default 60)")
    args = parser.parse_args()

    set_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    series = load_series()
    train_units = list(range(1, args.train_engines + 1))
    test_units = list(range(61, 61 + args.max_engines))

    # Maintenance threshold from TRAIN engines only: median s11 fifteen
    # cycles before failure (exactly the prototype definition).
    threshold = float(np.median([series[u][-15] for u in train_units]))
    print(f"C-MAPSS FD001 | threshold s11 > {threshold:.3f} "
          f"(train-engines definition), horizon {HORIZON}, alpha {ALPHA}")
    print(f"train engines 1-{args.train_engines}, audit engines "
          f"{test_units[0]}-{test_units[-1]}, stride {args.stride}, "
          f"epochs {args.epochs}")

    forecaster = fit_forecaster(series, train_units, args.epochs)
    lo, hi, n_cal = forecaster._crossing_calibration(threshold, "above", ALPHA)
    print(f"crossing-time calibration: n={n_cal}, signed-error window "
          f"[{lo:+.2f}, {hi:+.2f}] cycles")

    records = audit(forecaster, series, test_units, threshold, args.stride)
    metrics = summarize(records, lo, hi)
    metrics["n_calibration_crossings"] = n_cal

    fmt = lambda v, s=1.0: "   --" if v is None else f"{v * s:6.2f}"
    print("\n=== FD001 audit on unseen engines "
          "(full event set; censored answers scored at the horizon) ===")
    print(f"windows audited      : {metrics['n_windows']}")
    print(f"real events          : {metrics['n_events']}")
    print(f"timing MAE  APDTFlow : {fmt(metrics['mae_full_apdtflow'])} cycles")
    print(f"timing MAE  linear   : {fmt(metrics['mae_full_linear'])} cycles")
    print(f"timing MAE  persist. : {fmt(metrics['mae_full_persistence'])} cycles")
    print(f"caught-event MAE     : {fmt(metrics['mae_caught'])} cycles "
          f"(catch rate {fmt(metrics['catch_rate'], 100)}%)")
    print(f"false alarms         : {metrics['n_false_alarms']}/"
          f"{metrics['n_no_crossing']} no-crossing windows "
          f"({fmt(metrics['false_alarm_rate'], 100)}%)")
    print(f"90%-window coverage  : {fmt(metrics['coverage'], 100)}%")
    print(f"matched subset (n={metrics['matched_subset_n']}): linear "
          f"{fmt(metrics['matched_linear_mae'])} vs APDTFlow "
          f"{fmt(metrics['matched_apdtflow_mae'])} cycles -- linear stays a sharp "
          f"point-estimator on easy windows; APDTFlow's edge is reliability")

    payload = {
        "demo": "turbofan_when",
        "dataset": "NASA C-MAPSS FD001 (100 run-to-failure engines)",
        "config": {
            "sensor": SENSOR, "threshold": threshold, "direction": "above",
            "history": HISTORY, "horizon": HORIZON, "alpha": ALPHA,
            "epochs": args.epochs, "stride": args.stride,
            "train_engines": args.train_engines, "audit_engines": args.max_engines,
            "seed": SEED,
        },
        "metrics": metrics,
    }
    out_json = RESULTS_DIR / "turbofan_when.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_json.relative_to(REPO)}")

    candidates = [r for r in records
                  if r["t_act"] is not None and 5 < r["t_act"] <= HORIZON - 8
                  and r["t_pred"] is not None]
    if candidates:
        record = min(candidates, key=lambda r: abs(r["t_act"] - 19))
        plot_hero(forecaster, series, record, threshold, lo, hi,
                  IMAGES_DIR / "apdtflow_turbofan_when.png")
        print(f"wrote assets/images/apdtflow_turbofan_when.png "
              f"(engine #{record['unit']}, actual crossing +{record['t_act']:.0f})")
    else:
        print("WARNING: no caught event window available; hero plot skipped")


if __name__ == "__main__":
    main()
