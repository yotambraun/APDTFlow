#!/usr/bin/env python
"""C-MAPSS FD002 robustness: condition-aware multivariate predict_when across 6 regimes.

The hardest C-MAPSS variant (Sections 10.8-10.9 of the v0.4.0 methodology):
260 engines under SIX shifting operating regimes — sensor values jump with the
regime, the structure of real industrial fleets. Pipeline:

  1. ``apdtflow.preprocessing.regime_normalize`` on op1..op3 — per-regime
     sensor z-normalization with statistics computed on TRAIN engines (1-150)
     only and reused for the unseen engines.
  2. Top-5 trending sensors selected on TRAIN engines by trend consistency
     (|correlation with cycle| of the smoothed normalized signal).
  3. The multivariate learned-health-indicator model
     (``fit(feature_cols=...)``), threshold = median indicator value fifteen
     cycles before failure on TRAIN engines, horizon 40.
  4. Audit on 110 UNSEEN engines (151-260): full-event-set timing MAE vs
     linear/persistence, catch rate, false alarms, calibrated 90%-window
     coverage — plus the fleet snapshot and the trust panel.

Outputs:
  * experiments/results/fd002_robustness.json
  * assets/images/apdtflow_fd002_robustness.png  (audit summary)
  * assets/images/apdtflow_fleet_dashboard.png   (predict_when_fleet schedule)
  * assets/images/apdtflow_trust_panel.png       (calibration + lead-time honesty)

Usage:
  python experiments/fd002_robustness_demo.py             # full run
  python experiments/fd002_robustness_demo.py --epochs 1 \
      --train-engines 8 --max-engines 6 --stride 25       # smoke run
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
from apdtflow.preprocessing import regime_normalize  # noqa: E402

warnings.filterwarnings("ignore")
for _name in ("apdtflow.forecaster", "apdtflow.conformal"):
    logging.getLogger(_name).setLevel(logging.WARNING)

DATA_FILE = REPO / "dataset_examples" / "cmapss" / "train_FD002.txt"
RESULTS_DIR = REPO / "experiments" / "results"
IMAGES_DIR = REPO / "assets" / "images"

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
TARGET = "health_indicator"
HISTORY = 30
HORIZON = 40
ALPHA = 0.1
SEED = 6
SMOOTH = 9

GREEN, PURPLE, RED, BLUE, PINK = "#2e8b57", "#6a5acd", "#d62839", "#4477aa", "#f4a7b4"


# ----------------------------------------------------------------- data
def load_and_normalize(train_units: list):
    """Load FD002 and regime-normalize with TRAIN-engine statistics only."""
    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: missing C-MAPSS data file {DATA_FILE.relative_to(REPO)}\n"
            "Download it first:  python dataset_examples/get_cmapss.py"
        )
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    # The six FD002 regimes are the rounded operating settings; snap the
    # op-columns to their regime centers (+0.0 collapses IEEE -0.0).
    for col in OP_COLS:
        df[col] = df[col].round(0) + 0.0
    _, stats = regime_normalize(df[df.unit.isin(train_units)], OP_COLS, SENSOR_COLS)
    df, _ = regime_normalize(df, OP_COLS, SENSOR_COLS, stats=stats)
    n_regimes = len(stats)
    return df, n_regimes


def select_sensors(df: pd.DataFrame, train_units: list, top_k: int = 5):
    """Trend-consistency sensor selection on TRAIN engines only."""
    probe_units = train_units[:min(60, len(train_units))]
    scores = {}
    for s in SENSOR_COLS:
        cs = []
        for u in probe_units:
            v = (df.loc[df.unit == u, s]
                 .rolling(SMOOTH, min_periods=1).mean().to_numpy())
            cs.append(0.0 if np.std(v) < 1e-6
                      else float(np.corrcoef(np.arange(len(v)), v)[0, 1]))
        scores[s] = float(np.mean(cs))
    selected = sorted(scores, key=lambda s: -abs(scores[s]))[:top_k]
    sign = 1.0 if scores[selected[0]] > 0 else -1.0
    return selected, sign, scores


def build_frames(df: pd.DataFrame, sensors: list, sign: float) -> dict:
    """Per-engine engineered frames: oriented health indicator + 4 features."""
    frames = {}
    for u, g in df.groupby("unit"):
        smoothed = {s: g[s].rolling(SMOOTH, min_periods=1).mean().to_numpy(dtype=float)
                    for s in sensors}
        data = {TARGET: sign * smoothed[sensors[0]]}
        data.update({s: smoothed[s] for s in sensors[1:]})
        frames[int(u)] = pd.DataFrame(data)
    return frames


# ----------------------------------------------------------------- model
def fit_forecaster(frames: dict, train_units: list, features: list,
                   epochs: int) -> APDTFlowForecaster:
    df = pd.concat([frames[u] for u in train_units], ignore_index=True)
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
    forecaster.fit(df, target_col=TARGET, feature_cols=features)
    return forecaster


def dense_trajectory(forecaster, window: np.ndarray, n_points: int = 240):
    """Denormalized health-indicator trajectory for one raw (C, T) window."""
    taus = np.linspace(HORIZON / n_points, HORIZON, n_points)
    win_norm = ((window - forecaster.feature_means_[:, None])
                / forecaster.feature_stds_[:, None])
    x = torch.tensor(win_norm[None], dtype=torch.float32, device=forecaster.device)
    t_span = torch.linspace(0, 1, steps=HISTORY, device=forecaster.device)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=forecaster.device)
    forecaster.model.eval()
    with torch.no_grad():
        vals, _ = forecaster.model.forward_at(x, t_span, taus_t)
    traj = vals.squeeze(-1).cpu().numpy()[0]
    return taus, traj * forecaster.scaler_std_ + forecaster.scaler_mean_


# ----------------------------------------------------------------- audit
def linear_eta(window: np.ndarray, threshold: float) -> float | None:
    t = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(t, window, 1)
    if slope <= 0:
        return None
    anchor_value = intercept + slope * (len(window) - 1)
    tau = (threshold - anchor_value) / slope
    return float(tau) if 0 < tau <= HORIZON else None


def audit(forecaster, frames: dict, test_units: list, channels: list,
          threshold: float, stride: int) -> list:
    grid = np.arange(0.0, HORIZON + 1)
    records, windows = [], []
    for u in test_units:
        matrix = frames[u][channels].to_numpy(dtype=float).T  # (C, T)
        target = matrix[0]
        for i in range(0, matrix.shape[1] - HISTORY - HORIZON + 1, stride):
            window = matrix[:, i:i + HISTORY]
            anchor = window[0, -1]
            future = target[i + HISTORY:i + HISTORY + HORIZON]
            t_act = first_crossing_time(
                grid, np.concatenate([[anchor], future]), threshold, "above")
            records.append({
                "unit": u, "origin": i, "anchor": float(anchor), "t_act": t_act,
                "t_lin": linear_eta(window[0], threshold),
                "t_per": 0.0 if anchor >= threshold else None,
            })
            windows.append(window)

    x = np.stack(windows)
    x_norm = ((x - forecaster.feature_means_[None, :, None])
              / forecaster.feature_stds_[None, :, None])
    t_pred = forecaster._batch_crossing_times(
        torch.tensor(x_norm, dtype=torch.float32), None, threshold, "above")
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


# ----------------------------------------------------------------- plots
def style_axis(ax):
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_robustness(forecaster, frames, channels, record, threshold, lo, hi,
                    metrics, n_audit_engines, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: one unseen engine, hero-style.
    ax = axes[0]
    matrix = frames[record["unit"]][channels].to_numpy(dtype=float).T
    target = matrix[0]
    origin = record["origin"]
    hist_x = np.arange(-(HISTORY - 1), 1)
    hist_y = target[origin:origin + HISTORY]
    future = target[origin + HISTORY:origin + HISTORY + HORIZON]
    taus, traj = dense_trajectory(forecaster, matrix[:, origin:origin + HISTORY])
    eta = record["t_pred"]
    earliest = float(np.clip(eta - hi, 0, HORIZON)) if np.isfinite(hi) else eta
    latest = float(np.clip(eta - lo, 0, HORIZON)) if np.isfinite(lo) else HORIZON
    ax.plot(hist_x, hist_y, "-o", color=BLUE, ms=3.5, lw=1.4,
            label=f"Health indicator — engine #{record['unit']} "
                  "(unseen, regime-normalized)")
    ax.plot(np.arange(1, len(future) + 1), future, "o", color="black", ms=4,
            label="Actual future")
    ax.plot(taus, traj, color=GREEN, lw=2.2, label="APDTFlow multivariate forecast")
    ax.axhline(threshold, color=RED, ls="--", lw=1.6, label="Maintenance threshold")
    ax.axvspan(earliest, latest, color=RED, alpha=0.12,
               label="90% crossing-time window")
    ax.axvline(eta, color=RED, lw=2.2)
    ax.axvline(0, color="gray", ls=":", lw=1.0)
    ax.plot([record["t_act"]], [threshold], "*", color="black", ms=20,
            label=f"Actual crossing: +{record['t_act']:.0f}")
    ax.annotate(f"predict_when = +{eta:.0f} cycles", xy=(eta, threshold),
                xytext=(eta + 1.5, threshold + 0.18 * np.ptp(ax.get_ylim())),
                color=RED, fontsize=10)
    ax.set_xlabel("flight cycles")
    ax.set_ylabel("degradation health indicator")
    ax.set_title("C-MAPSS FD002 — engine never seen in training\n"
                 "6 operating regimes, condition-aware multivariate pipeline",
                 fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    style_axis(ax)

    # Right: audit bars.
    ax = axes[1]
    methods = [("APDTFlow\nmultivariate", metrics["mae_full_apdtflow"], GREEN),
               ("Linear\nextrapolation", metrics["mae_full_linear"], PURPLE),
               ("Persistence", metrics["mae_full_persistence"], RED)]
    for j, (label, value, color) in enumerate(methods):
        value = value or 0.0
        ax.bar([j], [value], 0.62, color=color)
        ax.annotate(f"{value:.1f}", (j, value), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel("EOL timing error (cycles)")
    cov = (metrics["coverage"] or 0) * 100
    fa = (metrics["false_alarm_rate"] or 0) * 100
    catch = (metrics["catch_rate"] or 0) * 100
    ax.set_title(f"{n_audit_engines} unseen engines, {metrics['n_events']} real events\n"
                 f"coverage {cov:.0f}% | false alarms {fa:.1f}% | catch {catch:.0f}%",
                 fontsize=11)
    style_axis(ax)

    ratio = ((metrics["mae_full_linear"] or 0)
             / max(metrics["mae_full_apdtflow"] or 1e-9, 1e-9))
    fig.suptitle(f"FD002: the multi-regime robustness win — {ratio:.1f}x better than "
                 "linear under varying operating conditions", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fleet_dashboard(schedule: pd.DataFrame, actual_by_unit: dict,
                         n_engines: int, act_by_ok: float, coverage: float,
                         path: Path, max_rows: int = 30):
    shown = schedule.head(max_rows)
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.34 * len(shown) + 2.5)))
    ys = np.arange(len(shown))[::-1]
    for y, (_, row) in zip(ys, shown.iterrows()):
        ax.barh(y, row["latest"] - row["act_by"], left=row["act_by"],
                height=0.62, color=PINK, zorder=2)
        ax.plot([row["act_by"], row["act_by"]], [y - 0.31, y + 0.31],
                color=RED, lw=2.6, zorder=3)
        t_act = actual_by_unit.get(row["asset_id"])
        if t_act is not None:
            ax.plot([t_act], [y], "*", color="black", ms=13, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"engine {row['asset_id']}" for _, row in shown.iterrows()],
                       fontsize=8)
    ax.set_xlabel("cycles from today")
    ax.set_title(
        "predict_when_fleet(): the maintenance schedule from one call\n"
        f"{n_engines} real engines (NASA FD002, unseen in training) — sorted by "
        "ACT-BY date\n"
        f"in this snapshot: act-by date was early enough for {act_by_ok * 100:.0f}% "
        f"of engines | window coverage {coverage * 100:.0f}%",
        fontsize=12,
    )
    handles = [
        plt.Line2D([0], [0], color=RED, lw=2.6,
                   label="ACT-BY date (earliest plausible crossing — schedule on this)"),
        plt.Line2D([0], [0], marker="*", color="black", ls="none", ms=12,
                   label="what actually happened"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.005,
             "Operational rule (calibrated): the point estimate runs late on "
             "smoothed indicators — always schedule by the window's earliest edge.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trust_panel(records, lo, hi, metrics, n_engines, path: Path):
    caught = [r for r in records
              if r["t_act"] is not None and r["t_act"] > 0 and r["t_pred"] is not None]
    t_act = np.array([r["t_act"] for r in caught])
    t_pred = np.array([r["t_pred"] for r in caught])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    xs = np.linspace(0, HORIZON, 50)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.fill_between(xs, xs + lo, xs + hi, color=GREEN, alpha=0.15,
                        label="90% calibrated band")
    ax.plot(xs, xs, "k--", lw=1.2, label="perfect timing")
    ax.plot(t_act, t_pred, "o", color=GREEN, ms=4.5, alpha=0.45, mec="none")
    ax.set_xlabel("actual crossing time (cycles)")
    ax.set_ylabel("predicted crossing time")
    cov = (metrics["coverage"] or 0) * 100
    ax.set_title(f"All {len(caught)} caught events, {n_engines} unseen engines\n"
                 f"{cov:.0f}% inside the band (target {100 * (1 - ALPHA):.0f}%)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    style_axis(ax)

    ax = axes[1]
    edges = [1, 8, 16, 24, 32, 40]
    labels, maes, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (t_act >= a) & (t_act < b) if b < HORIZON else \
               (t_act >= a) & (t_act <= b)
        labels.append(f"{a}-{b}")
        counts.append(int(mask.sum()))
        maes.append(float(np.mean(np.abs(t_pred[mask] - t_act[mask])))
                    if mask.any() else 0.0)
    bars = ax.bar(labels, maes, color=GREEN)
    for b, v, n in zip(bars, maes, counts):
        ax.annotate(f"{v:.1f}\n(n={n})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("how far ahead the event is (cycles)")
    ax.set_ylabel("timing error (cycles)")
    ax.set_title("Accuracy by lead time: know how much to trust each horizon",
                 fontsize=11)
    style_axis(ax)

    fig.suptitle("The trust panel: calibrated, honest, and it tells you its own limits",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=8,
                        help="training epochs (default 8)")
    parser.add_argument("--stride", type=int, default=5,
                        help="audit window stride (default 5)")
    parser.add_argument("--max-engines", type=int, default=110,
                        help="number of audit engines from 151.. (default 110)")
    parser.add_argument("--train-engines", type=int, default=150,
                        help="number of training engines from 1.. (default 150)")
    args = parser.parse_args()

    set_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    train_units = list(range(1, args.train_engines + 1))
    test_units = list(range(151, 151 + args.max_engines))

    df, n_regimes = load_and_normalize(train_units)
    sensors, sign, scores = select_sensors(df, train_units)
    print(f"C-MAPSS FD002 | {n_regimes} operating regimes, per-regime "
          "normalization fit on TRAIN engines only")
    print("selected sensors (trend consistency on train): "
          + ", ".join(f"{s} ({scores[s]:+.3f})" for s in sensors))

    frames = build_frames(df, sensors, sign)
    channels = [TARGET] + sensors[1:]
    threshold = float(np.median([frames[u][TARGET].to_numpy()[-15]
                                 for u in train_units]))
    orient = "oriented increasing" if sign > 0 else "sign-flipped to increase"
    print(f"target: {sensors[0]} ({orient}); threshold > {threshold:.3f} "
          f"(train-engines definition), horizon {HORIZON}, alpha {ALPHA}, "
          f"epochs {args.epochs}")

    forecaster = fit_forecaster(frames, train_units, sensors[1:], args.epochs)
    lo, hi, n_cal = forecaster._crossing_calibration(threshold, "above", ALPHA)
    print(f"crossing-time calibration: n={n_cal}, signed-error window "
          f"[{lo:+.2f}, {hi:+.2f}] cycles")

    records = audit(forecaster, frames, test_units, channels, threshold, args.stride)
    metrics = summarize(records, lo, hi)
    metrics["n_calibration_crossings"] = n_cal

    fmt = lambda v, s=1.0: "   --" if v is None else f"{v * s:6.2f}"
    print(f"\n=== FD002 audit on {len(test_units)} unseen engines "
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
          f"{fmt(metrics['matched_apdtflow_mae'])} cycles")

    # ---- fleet snapshot: one window per unseen engine, varied lead times
    rng = np.random.default_rng(SEED)
    histories, actual_by_unit = {}, {}
    for u in test_units:
        target = frames[u][TARGET].to_numpy()
        lead = int(rng.integers(8, HORIZON))
        origin = len(target) - HISTORY - lead
        if origin < 0:
            continue
        histories[u] = frames[u].iloc[:origin + HISTORY]
        anchor = target[origin + HISTORY - 1]
        future = target[origin + HISTORY:origin + HISTORY + HORIZON]
        actual_by_unit[u] = first_crossing_time(
            np.arange(0.0, len(future) + 1),
            np.concatenate([[anchor], future]), threshold, "above")

    schedule = forecaster.predict_when_fleet(
        histories, threshold=threshold, direction="above", alpha=ALPHA)
    judged = [(row, actual_by_unit[row["asset_id"]])
              for _, row in schedule.iterrows()
              if actual_by_unit.get(row["asset_id"]) is not None]
    act_by_ok = (float(np.mean([row["act_by"] <= t for row, t in judged]))
                 if judged else 0.0)
    fleet_cov = (float(np.mean([row["earliest"] <= t <= row["latest"]
                                for row, t in judged
                                if not row["censored"]])) if judged else 0.0)
    fleet_stats = {
        "n_engines": len(histories),
        "n_with_actual_crossing": len(judged),
        "act_by_early_enough": act_by_ok,
        "window_coverage": fleet_cov,
    }
    print(f"\nfleet snapshot ({len(histories)} engines): act-by early enough for "
          f"{act_by_ok * 100:.0f}% | window coverage {fleet_cov * 100:.0f}%")
    print(schedule.head(10).to_string(index=False,
                                      float_format=lambda v: f"{v:.1f}"))

    payload = {
        "demo": "fd002_robustness",
        "dataset": "NASA C-MAPSS FD002 (260 engines, 6 operating regimes)",
        "config": {
            "selected_sensors": sensors, "target_orientation_sign": sign,
            "threshold": threshold, "direction": "above",
            "history": HISTORY, "horizon": HORIZON, "alpha": ALPHA,
            "epochs": args.epochs, "stride": args.stride,
            "train_engines": args.train_engines, "audit_engines": args.max_engines,
            "n_regimes": n_regimes, "seed": SEED,
        },
        "metrics": metrics,
        "fleet_snapshot": fleet_stats,
        "sensor_importance": {k: float(v)
                              for k, v in forecaster.sensor_importance_.items()},
    }
    out_json = RESULTS_DIR / "fd002_robustness.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_json.relative_to(REPO)}")

    # ---- figures
    candidates = [r for r in records
                  if r["t_act"] is not None and 8 < r["t_act"] <= HORIZON - 12
                  and r["t_pred"] is not None]
    if candidates:
        record = min(candidates, key=lambda r: abs(r["t_act"] - 19))
        plot_robustness(forecaster, frames, channels, record, threshold, lo, hi,
                        metrics, len(test_units),
                        IMAGES_DIR / "apdtflow_fd002_robustness.png")
        print("wrote assets/images/apdtflow_fd002_robustness.png")
    else:
        print("WARNING: no caught event window available; robustness plot skipped")

    plot_fleet_dashboard(schedule, actual_by_unit, len(histories),
                         act_by_ok, fleet_cov,
                         IMAGES_DIR / "apdtflow_fleet_dashboard.png")
    print("wrote assets/images/apdtflow_fleet_dashboard.png")

    plot_trust_panel(records, lo, hi, metrics, len(test_units),
                     IMAGES_DIR / "apdtflow_trust_panel.png")
    print("wrote assets/images/apdtflow_trust_panel.png")


if __name__ == "__main__":
    main()
