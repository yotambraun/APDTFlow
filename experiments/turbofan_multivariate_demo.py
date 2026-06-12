#!/usr/bin/env python
"""C-MAPSS FD001 multivariate: 5-sensor learned health indicator vs univariate s11.

Section 10.7 of the v0.4.0 methodology: the multivariate upgrade, validated.
APDTFlow is fitted with ``fit(feature_cols=...)`` — sensor s11 as the target
plus s12, s4, s7, s15 (the most consistently trending FD001 sensors, selected
on TRAIN engines) fused by the learned health-indicator layer — under the
SAME training budget and the SAME audit as experiments/turbofan_when_demo.py:

  * train engines 1-60, audit engines 61-100, horizon 40, alpha 0.1
  * identical threshold definition (median s11 fifteen cycles before failure,
    train engines only) and identical sliding-window audit

The univariate side of the comparison is read from
experiments/results/turbofan_when.json — run turbofan_when_demo.py first.

Outputs:
  * experiments/results/turbofan_multivariate.json
  * assets/images/apdtflow_multivariate_audit.png (side-by-side bars + learned
    sensor-importance weights)

Usage:
  python experiments/turbofan_multivariate_demo.py        # full run
  python experiments/turbofan_multivariate_demo.py --epochs 1 \
      --train-engines 6 --max-engines 3 --stride 15       # smoke run
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
UNIVARIATE_JSON = RESULTS_DIR / "turbofan_when.json"

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
TARGET = "s11"
FEATURES = ["s12", "s4", "s7", "s15"]  # top trending sensors, selected on train
CHANNELS = [TARGET] + FEATURES
HISTORY = 30
HORIZON = 40
ALPHA = 0.1
SEED = 4

GREEN, PURPLE, RED = "#2e8b57", "#6a5acd", "#d62839"


def load_frames() -> dict:
    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: missing C-MAPSS data file {DATA_FILE.relative_to(REPO)}\n"
            "Download it first:  python dataset_examples/get_cmapss.py"
        )
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    frames = {}
    for u, g in df.groupby("unit"):
        smoothed = {s: g[s].rolling(5, min_periods=1).mean().to_numpy(dtype=float)
                    for s in CHANNELS}
        frames[int(u)] = pd.DataFrame(smoothed)
    return frames


def fit_forecaster(frames: dict, train_units: list, epochs: int) -> APDTFlowForecaster:
    """Fit the multivariate model on the concatenated training engines."""
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
    forecaster.fit(df, target_col=TARGET, feature_cols=FEATURES)
    return forecaster


def linear_eta(window: np.ndarray, threshold: float) -> float | None:
    t = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(t, window, 1)
    if slope <= 0:
        return None
    anchor_value = intercept + slope * (len(window) - 1)
    tau = (threshold - anchor_value) / slope
    return float(tau) if 0 < tau <= HORIZON else None


def audit(forecaster, frames: dict, test_units: list, threshold: float,
          stride: int) -> list:
    grid = np.arange(0.0, HORIZON + 1)
    records, windows = [], []
    for u in test_units:
        matrix = frames[u][CHANNELS].to_numpy(dtype=float).T  # (C, T)
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


def plot_comparison(uni: dict, multi: dict, importance: pd.Series, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Left: side-by-side audit bars.
    ax = axes[0]
    groups = [
        ("Timing MAE\n(cycles)", "mae_full_apdtflow", 1.0, "{:.2f}"),
        ("Event catch\nrate (%)", "catch_rate", 100.0, "{:.0f}"),
        ("False-alarm\nrate (%)", "false_alarm_rate", 100.0, "{:.1f}"),
    ]
    x = np.arange(len(groups))
    width = 0.34
    for j, (label, source, color) in enumerate([
            ("Univariate (1 sensor)", uni, PURPLE),
            ("Multivariate (5-sensor learned health indicator)", multi, GREEN)]):
        vals = [(source.get(key) or 0.0) * scale for _, key, scale, _ in groups]
        bars = ax.bar(x + (j - 0.5) * width, vals, width, color=color, label=label)
        for b, v, (_, _, _, f) in zip(bars, vals, groups):
            ax.annotate(f.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    cov_u = (uni.get("coverage") or 0) * 100
    cov_m = (multi.get("coverage") or 0) * 100
    ax.set_title("NASA C-MAPSS, unseen engines: one fusion layer\n"
                 f"(coverage: {cov_u:.0f}% univariate vs {cov_m:.0f}% multivariate)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.3, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Right: learned health-indicator weights.
    ax = axes[1]
    names = list(importance.index)
    weights = importance.to_numpy(dtype=float)
    colors = [GREEN if w >= 0 else RED for w in weights]
    ax.barh(names, weights, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("weight")
    ax.set_title("What the learned health indicator uses\n"
                 "(fusion weights — interpretable sensor contributions)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle("Multivariate validated: same package, same audit, "
                 "+4 sensors via fit(feature_cols=...)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=25,
                        help="training epochs — keep identical to "
                             "turbofan_when_demo.py (default 25)")
    parser.add_argument("--stride", type=int, default=3,
                        help="audit window stride (default 3)")
    parser.add_argument("--max-engines", type=int, default=40,
                        help="number of audit engines from 61.. (default 40)")
    parser.add_argument("--train-engines", type=int, default=60,
                        help="number of training engines from 1.. (default 60)")
    args = parser.parse_args()

    if not UNIVARIATE_JSON.exists():
        sys.exit(
            "ERROR: univariate reference results not found at "
            f"{UNIVARIATE_JSON.relative_to(REPO)}\n"
            "Run the univariate audit first:  python experiments/turbofan_when_demo.py"
        )
    uni_payload = json.loads(UNIVARIATE_JSON.read_text())
    uni = uni_payload["metrics"]
    uni_cfg = uni_payload.get("config", {})
    if (uni_cfg.get("stride") != args.stride
            or uni_cfg.get("audit_engines") != args.max_engines):
        print("NOTE: univariate run used a different audit configuration "
              f"(stride {uni_cfg.get('stride')}, engines {uni_cfg.get('audit_engines')}); "
              "the comparison is indicative, rerun both with matching flags "
              "for a strict A/B.")

    set_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_frames()
    train_units = list(range(1, args.train_engines + 1))
    test_units = list(range(61, 61 + args.max_engines))

    threshold = float(np.median([frames[u][TARGET].to_numpy()[-15]
                                 for u in train_units]))
    print(f"C-MAPSS FD001 multivariate | target {TARGET} + features {FEATURES}")
    print(f"threshold {TARGET} > {threshold:.3f} (train-engines definition), "
          f"horizon {HORIZON}, alpha {ALPHA}, epochs {args.epochs}")

    forecaster = fit_forecaster(frames, train_units, args.epochs)
    lo, hi, n_cal = forecaster._crossing_calibration(threshold, "above", ALPHA)
    print(f"crossing-time calibration: n={n_cal}, signed-error window "
          f"[{lo:+.2f}, {hi:+.2f}] cycles")

    records = audit(forecaster, frames, test_units, threshold, args.stride)
    metrics = summarize(records, lo, hi)
    metrics["n_calibration_crossings"] = n_cal

    importance = forecaster.sensor_importance_
    fmt = lambda v, s=1.0: "   --" if v is None else f"{v * s:6.2f}"
    print("\n=== FD001 audit: univariate vs multivariate (same unseen engines) ===")
    print(f"{'metric':28s} {'univariate':>11s} {'multivariate':>13s}")
    rows = [
        ("timing MAE, full set (cyc)", "mae_full_apdtflow", 1.0),
        ("timing MAE, caught (cyc)", "mae_caught", 1.0),
        ("event catch rate (%)", "catch_rate", 100.0),
        ("false-alarm rate (%)", "false_alarm_rate", 100.0),
        ("90%-window coverage (%)", "coverage", 100.0),
        ("events audited (n)", "n_events", 1.0),
    ]
    for label, key, scale in rows:
        print(f"{label:28s} {fmt(uni.get(key), scale):>11s} "
              f"{fmt(metrics.get(key), scale):>13s}")
    print("\nLearned sensor importance (fusion weights):")
    for name, weight in importance.items():
        print(f"  {name:5s} {weight:+.4f}")

    payload = {
        "demo": "turbofan_multivariate",
        "dataset": "NASA C-MAPSS FD001 (100 run-to-failure engines)",
        "config": {
            "target": TARGET, "feature_cols": FEATURES, "threshold": threshold,
            "direction": "above", "history": HISTORY, "horizon": HORIZON,
            "alpha": ALPHA, "epochs": args.epochs, "stride": args.stride,
            "train_engines": args.train_engines, "audit_engines": args.max_engines,
            "seed": SEED,
        },
        "metrics": metrics,
        "univariate_reference": uni,
        "sensor_importance": {k: float(v) for k, v in importance.items()},
    }
    out_json = RESULTS_DIR / "turbofan_multivariate.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_json.relative_to(REPO)}")

    plot_comparison(uni, metrics, importance,
                    IMAGES_DIR / "apdtflow_multivariate_audit.png")
    print("wrote assets/images/apdtflow_multivariate_audit.png")


if __name__ == "__main__":
    main()
