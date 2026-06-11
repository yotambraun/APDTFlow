#!/usr/bin/env python
"""NASA battery end-of-life: full leave-one-battery-out audit of predict_when().

Real NASA PCoE Li-ion capacity-degradation data (cells B0005/B0006/B0007), one
row per measured discharge cycle. Question: "when will capacity cross the
1.4 Ah end-of-life threshold?"

Protocol (Section 10.5 of the v0.4.0 methodology):
  * Full leave-one-battery-out cross-validation over the 3 cells.
  * Per fold: fit APDTFlow (continuous decoder + conformal calibration) on the
    two training cells' capacity series; audit every sliding window of the
    held-out cell with the crossing-time machinery behind ``predict_when``.
  * Baselines: persistence (no future crossing unless already crossed) and
    linear extrapolation of the input window.
  * Metrics: full-event-set timing MAE (a censored "no crossing" answer is
    scored at the horizon), caught-event MAE, catch rate, calibrated
    90%-window coverage (alpha=0.1, asymmetric signed-error calibration), and
    censoring honesty on the cell that never reaches EOL inside the horizon.

The cycle ids in the CSVs are irregular (the cycle column jumps by ~2-4);
rows are the model's time grid ("measured cycles") and errors are also
reported in chronological cycles via the median cycle-column spacing.

Outputs:
  * experiments/results/battery_eol.json
  * assets/images/apdtflow_battery_eol.png        (hero panel)
  * assets/images/apdtflow_battery_full_audit.png (RUL convergence + audit bars)

Usage:
  python experiments/battery_eol_demo.py            # full run (~5 min CPU)
  python experiments/battery_eol_demo.py --epochs 1 # smoke run
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

DATA_DIR = REPO / "dataset_examples"
RESULTS_DIR = REPO / "experiments" / "results"
IMAGES_DIR = REPO / "assets" / "images"

CELLS = ["B0005", "B0006", "B0007"]
HERO_CELL = "B0005"
THRESHOLD = 1.4  # Ah, the standard NASA EOL definition
DIRECTION = "below"
HISTORY = 30
HORIZON = 30
ALPHA = 0.1  # 90% crossing-time windows
SEED = 3

GREEN, PURPLE, RED, BLUE = "#2e8b57", "#6a5acd", "#d62839", "#4477aa"


# ----------------------------------------------------------------- data
def load_cells() -> dict:
    missing = [c for c in CELLS if not (DATA_DIR / f"nasa_{c}.csv").exists()]
    if missing:
        sys.exit(
            "ERROR: missing NASA battery data files: "
            + ", ".join(f"dataset_examples/nasa_{c}.csv" for c in missing)
            + "\nDownload them first:  python dataset_examples/get_cmapss.py"
        )
    cells = {}
    for cell in CELLS:
        df = pd.read_csv(DATA_DIR / f"nasa_{cell}.csv")
        cells[cell] = {
            "capacity": df["capacity"].to_numpy(dtype=float),
            "cycle": df["cycle"].to_numpy(dtype=float),
        }
    return cells


# ----------------------------------------------------------------- model
def fit_fold(train_series: list, epochs: int) -> APDTFlowForecaster:
    """Fit one LOBO fold on the concatenation of the two training cells."""
    df = pd.DataFrame({"capacity": np.concatenate(train_series)})
    forecaster = APDTFlowForecaster(
        forecast_horizon=HORIZON,
        history_length=HISTORY,
        num_epochs=epochs,
        batch_size=32,
        hidden_dim=32,
        num_scales=3,
        learning_rate=1e-3,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    forecaster.fit(df, target_col="capacity")
    return forecaster


def dense_trajectory(forecaster, window: np.ndarray, n_points: int = 240):
    """Denormalized mean forecast trajectory for one raw input window."""
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


# ----------------------------------------------------------------- audit
def linear_eta(window: np.ndarray) -> float | None:
    """Crossing time of a least-squares line fitted to the input window."""
    t = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(t, window, 1)
    if (DIRECTION == "below" and slope >= 0) or (DIRECTION == "above" and slope <= 0):
        return None
    anchor_value = intercept + slope * (len(window) - 1)
    tau = (THRESHOLD - anchor_value) / slope
    return float(tau) if 0 < tau <= HORIZON else None


def audit_cell(forecaster, series: np.ndarray, stride: int) -> list:
    """Slide windows over a held-out cell; return one record per window."""
    grid = np.arange(0.0, HORIZON + 1)
    records, windows = [], []
    for i in range(0, len(series) - HISTORY - HORIZON + 1, stride):
        window = series[i:i + HISTORY]
        anchor = window[-1]
        future = series[i + HISTORY:i + HISTORY + HORIZON]
        t_act = first_crossing_time(
            grid, np.concatenate([[anchor], future]), THRESHOLD, DIRECTION
        )
        records.append({
            "origin": i,
            "anchor": float(anchor),
            "t_act": t_act,
            "t_lin": linear_eta(window),
            # persistence: a flat line never crosses unless already crossed
            "t_per": 0.0 if (anchor <= THRESHOLD) == (DIRECTION == "below") else None,
        })
        windows.append(window)

    x_norm = (np.asarray(windows) - forecaster.scaler_mean_) / forecaster.scaler_std_
    t_pred = forecaster._batch_crossing_times(
        torch.tensor(x_norm[:, None, :], dtype=torch.float32),
        None, THRESHOLD, DIRECTION,
    )
    for record, tp in zip(records, t_pred):
        record["t_pred"] = None if np.isnan(tp) else float(tp)
    return records


def summarize(records: list, lo: float, hi: float) -> dict:
    """Event / censoring metrics for one set of audit records."""
    events = [r for r in records if r["t_act"] is not None and r["t_act"] > 0]
    no_cross = [r for r in records if r["t_act"] is None]

    def full_mae(key):
        if not events:
            return None
        errs = [abs((r[key] if r[key] is not None else HORIZON) - r["t_act"])
                for r in events]
        return float(np.mean(errs))

    caught = [r for r in events if r["t_pred"] is not None]
    covered = None
    if caught and np.isfinite(lo) and np.isfinite(hi):
        hits = [
            np.clip(r["t_pred"] - hi, 0, HORIZON)
            <= r["t_act"]
            <= np.clip(r["t_pred"] - lo, 0, HORIZON)
            for r in caught
        ]
        covered = float(np.mean(hits))
    false_alarms = [r for r in no_cross if r["t_pred"] is not None]
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
        "n_correctly_censored": len(no_cross) - len(false_alarms),
        "false_alarm_rate": (len(false_alarms) / len(no_cross)) if no_cross else None,
        "coverage": covered,
    }


# ----------------------------------------------------------------- plots
def style_axis(ax):
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_hero(forecaster, series, record, fleet_row, path: Path):
    origin = record["origin"]
    anchor_idx = origin + HISTORY - 1
    k = min(20, anchor_idx + 1)
    hist_x = np.arange(-(k - 1), 1)
    hist_y = series[anchor_idx - k + 1:anchor_idx + 1]
    future = series[origin + HISTORY:origin + HISTORY + HORIZON]
    taus, traj = dense_trajectory(forecaster, series[origin:origin + HISTORY])

    eta, earliest, latest = fleet_row["eta"], fleet_row["earliest"], fleet_row["latest"]
    t_act = record["t_act"]

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.plot(hist_x, hist_y, "-o", color=BLUE, ms=4, lw=1.6,
            label="Capacity history (real)")
    ax.plot(np.arange(1, len(future) + 1), future, "o", color="black", ms=5,
            label="Actual future capacity")
    ax.plot(taus, traj, color=GREEN, lw=2.4, label="APDTFlow continuous forecast")
    ax.axhline(THRESHOLD, color=RED, ls="--", lw=1.8,
               label=f"End-of-life threshold {THRESHOLD} Ah")
    ax.axvspan(earliest, latest, color=RED, alpha=0.12,
               label="90% crossing-time window")
    ax.axvline(eta, color=RED, lw=2.4)
    ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.plot([t_act], [THRESHOLD], "*", color="black", ms=24,
            label=f"Actual EOL: cycle +{t_act:.0f}")
    ax.annotate(
        f"predict_when(< {THRESHOLD})\n= +{eta:.0f} cycles  [{earliest:.0f}, {latest:.0f}]",
        xy=(eta, THRESHOLD), xytext=(eta + 1.0, ax.get_ylim()[0] +
                                     0.55 * np.ptp(ax.get_ylim())),
        color=RED, fontsize=11,
    )
    ax.set_xlabel("measured cycles from forecast origin")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title(
        f"predict_when() on REAL NASA battery data (held-out cell {HERO_CELL})\n"
        "'When will this battery reach end-of-life?' — leave-one-battery-out, "
        "calibrated 90% window",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_full_audit(hero_records, hero_band, eol_index, per_cell, censor_cell,
                    censor_stats, path: Path):
    lo, hi = hero_band
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left: RUL convergence on the hero cell.
    ax = axes[0]
    events = [r for r in hero_records
              if r["t_act"] is not None and r["t_act"] > 0 and r["t_pred"] is not None]
    ages = np.array([r["origin"] + HISTORY for r in events], dtype=float)
    pred_eol = np.array([r["origin"] + HISTORY + r["t_pred"] for r in events])
    if np.isfinite(lo) and np.isfinite(hi):
        band_lo = ages + np.clip([r["t_pred"] - hi for r in events], 0, HORIZON)
        band_hi = ages + np.clip([r["t_pred"] - lo for r in events], 0, HORIZON)
        ax.fill_between(ages, band_lo, band_hi, color=GREEN, alpha=0.18,
                        label="90% window")
    ax.plot(ages, pred_eol, "-o", color=GREEN, ms=3.5, lw=1.4,
            label="Predicted EOL cycle (predict_when)")
    ax.axhline(eol_index, color="black", ls="--", lw=1.4,
               label=f"Actual EOL ({eol_index:.0f})")
    ax.set_xlabel("forecast origin (battery age, measured cycles)")
    ax.set_ylabel("predicted EOL, measured cycle")
    ax.set_title(f"RUL convergence — held-out cell {HERO_CELL}\n"
                 "prediction sharpens toward the true end-of-life as the battery ages",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    style_axis(ax)

    # Right: per-cell audit bars vs baselines.
    ax = axes[1]
    cells = [c for c in CELLS if c != censor_cell]
    methods = [("APDTFlow predict_when", "mae_full_apdtflow", GREEN),
               ("Linear extrapolation", "mae_full_linear", PURPLE),
               ("Persistence", "mae_full_persistence", RED)]
    x = np.arange(len(cells))
    width = 0.26
    for j, (label, key, color) in enumerate(methods):
        vals = [per_cell[c][key] or 0.0 for c in cells]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(held out)" for c in cells])
    ax.set_ylabel("EOL timing error (measured cycles)")
    ax.set_title(
        "Leave-one-battery-out audit (real NASA cells)\n"
        f"{censor_cell} (no EOL in horizon): "
        f"{censor_stats['n_correctly_censored']}/{censor_stats['n_no_crossing']} "
        f"correctly censored, {censor_stats['n_false_alarms']} false alarms",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    style_axis(ax)

    fig.suptitle("predict_when() on REAL NASA battery data — full cross-validated evidence",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60,
                        help="training epochs per LOBO fold (default 60)")
    parser.add_argument("--stride", type=int, default=1,
                        help="audit window stride on the held-out cell (default 1)")
    args = parser.parse_args()

    set_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cells = load_cells()
    spacing = {c: float(np.median(np.diff(cells[c]["cycle"]))) for c in CELLS}
    print(f"NASA battery LOBO audit | threshold {THRESHOLD} Ah, horizon {HORIZON} "
          f"measured cycles, alpha {ALPHA}, epochs {args.epochs}")
    print(f"median chronological cycle spacing: "
          + ", ".join(f"{c}={spacing[c]:.1f}" for c in CELLS))

    per_cell, calibrations = {}, {}
    hero_assets = None
    for held_out in CELLS:
        train_cells = [c for c in CELLS if c != held_out]
        print(f"\n[fold] held-out {held_out} <- train on {'+'.join(train_cells)}")
        forecaster = fit_fold([cells[c]["capacity"] for c in train_cells], args.epochs)
        lo, hi, n_cal = forecaster._crossing_calibration(THRESHOLD, DIRECTION, ALPHA)
        records = audit_cell(forecaster, cells[held_out]["capacity"], args.stride)
        stats = summarize(records, lo, hi)
        stats["n_calibration_crossings"] = n_cal
        per_cell[held_out] = stats
        calibrations[held_out] = (lo, hi)
        if held_out == HERO_CELL:
            hero_assets = (forecaster, records, (lo, hi))

    # --- pooled metrics over the cells that actually reach EOL
    event_cells = [c for c in CELLS
                   if per_cell[c]["n_events"] > 0 and c != "B0007"] or \
                  [c for c in CELLS if per_cell[c]["n_events"] > 0]
    pooled = {}
    for key in ("mae_full_apdtflow", "mae_full_linear", "mae_full_persistence"):
        num = sum(per_cell[c][key] * per_cell[c]["n_events"] for c in event_cells)
        pooled[key] = num / sum(per_cell[c]["n_events"] for c in event_cells)
    cov_cells = [c for c in event_cells if per_cell[c]["coverage"] is not None]
    pooled["coverage"] = (
        sum(per_cell[c]["coverage"] * per_cell[c]["n_events"] for c in cov_cells)
        / sum(per_cell[c]["n_events"] for c in cov_cells) if cov_cells else None)
    pooled["catch_rate"] = (
        sum(per_cell[c]["catch_rate"] * per_cell[c]["n_events"] for c in event_cells)
        / sum(per_cell[c]["n_events"] for c in event_cells))
    pooled["n_events"] = int(sum(per_cell[c]["n_events"] for c in event_cells))
    mean_spacing = float(np.mean([spacing[c] for c in event_cells]))
    pooled["mae_full_apdtflow_chron_cycles"] = pooled["mae_full_apdtflow"] * mean_spacing

    # --- results table
    fmt = lambda v, scale=1.0: "  --" if v is None else f"{v * scale:6.2f}"
    print("\n=== Leave-one-battery-out audit "
          "(timing errors in measured cycles; full event set) ===")
    print(f"{'cell':8s} {'events':>6s} {'APDTFlow':>9s} {'Linear':>8s} "
          f"{'Persist':>8s} {'catch%':>7s} {'cover%':>7s} {'chron.cycles(APDT)':>19s}")
    for c in CELLS:
        s = per_cell[c]
        chron = (None if s["mae_full_apdtflow"] is None
                 else s["mae_full_apdtflow"] * spacing[c])
        print(f"{c:8s} {s['n_events']:6d} {fmt(s['mae_full_apdtflow']):>9s} "
              f"{fmt(s['mae_full_linear']):>8s} {fmt(s['mae_full_persistence']):>8s} "
              f"{fmt(s['catch_rate'], 100):>7s} {fmt(s['coverage'], 100):>7s} "
              f"{fmt(chron):>19s}")
    print(f"{'POOLED':8s} {pooled['n_events']:6d} {fmt(pooled['mae_full_apdtflow']):>9s} "
          f"{fmt(pooled['mae_full_linear']):>8s} {fmt(pooled['mae_full_persistence']):>8s} "
          f"{fmt(pooled['catch_rate'], 100):>7s} {fmt(pooled['coverage'], 100):>7s} "
          f"{fmt(pooled['mae_full_apdtflow_chron_cycles']):>19s}")
    b7 = per_cell["B0007"]
    print(f"\nCensoring honesty (B0007 never reaches EOL in horizon): "
          f"{b7['n_correctly_censored']}/{b7['n_no_crossing']} windows correctly "
          f"censored, {b7['n_false_alarms']} false alarms "
          f"({(b7['false_alarm_rate'] or 0) * 100:.0f}%)")

    # --- JSON
    payload = {
        "demo": "battery_eol",
        "dataset": "NASA PCoE Li-ion batteries B0005/B0006/B0007",
        "config": {
            "threshold_ah": THRESHOLD, "direction": DIRECTION,
            "history": HISTORY, "horizon": HORIZON, "alpha": ALPHA,
            "epochs": args.epochs, "stride": args.stride, "seed": SEED,
            "units": "measured cycles (rows); chronological cycles = "
                     "measured * median cycle spacing",
            "median_cycle_spacing": spacing,
        },
        "per_cell": per_cell,
        "pooled": pooled,
        "censoring_honesty": {
            "cell": "B0007",
            "n_no_crossing_windows": b7["n_no_crossing"],
            "n_correctly_censored": b7["n_correctly_censored"],
            "n_false_alarms": b7["n_false_alarms"],
            "false_alarm_rate": b7["false_alarm_rate"],
        },
    }
    out_json = RESULTS_DIR / "battery_eol.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_json.relative_to(REPO)}")

    # --- plots
    forecaster, hero_records, hero_band = hero_assets
    hero_series = cells[HERO_CELL]["capacity"]
    candidates = [r for r in hero_records
                  if r["t_act"] is not None and 0 < r["t_act"] <= HORIZON - 2
                  and r["t_pred"] is not None]
    if candidates:
        record = min(candidates, key=lambda r: abs(r["t_act"] - 20))
        fleet = forecaster.predict_when_fleet(
            {HERO_CELL: hero_series[:record["origin"] + HISTORY]},
            threshold=THRESHOLD, direction=DIRECTION, alpha=ALPHA,
        ).iloc[0]
        plot_hero(forecaster, hero_series, record, fleet,
                  IMAGES_DIR / "apdtflow_battery_eol.png")
        print(f"wrote assets/images/apdtflow_battery_eol.png "
              f"(origin {record['origin']}, actual EOL +{record['t_act']:.0f})")
    else:
        print("WARNING: no caught event window on the hero cell; hero plot skipped")

    eol_index = first_crossing_time(
        np.arange(len(hero_series), dtype=float), hero_series, THRESHOLD, DIRECTION)
    plot_full_audit(hero_records, hero_band, eol_index or len(hero_series),
                    per_cell, "B0007", b7,
                    IMAGES_DIR / "apdtflow_battery_full_audit.png")
    print("wrote assets/images/apdtflow_battery_full_audit.png")


if __name__ == "__main__":
    main()
