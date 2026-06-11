"""Adversarial audit harness for ``predict_when`` on any univariate CSV.

SHIPPING RULE: a predict_when domain demo is publishable only if it beats
ALL of (a) persistence, (b) linear extrapolation, (c) seasonal-naive/
climatology where the series is seasonal — on both event capture and timing
MAE, on held-out data, using this harness.

What it does
------------
Fits the production ``APDTFlowForecaster(decoder_type='continuous',
use_conformal=True)`` on the first ``--train-frac`` of the series, then
slides windows over the REMAINING held-out data and, for every window, asks
"when does the series first cross ``--threshold``?":

    - model:      first crossing of the continuous mean trajectory
                  (``_batch_crossing_times``) plus the calibrated
                  asymmetric time window from ``_crossing_calibration``
    - persistence: flat continuation of the last observed value
    - linear:      straight line fit on the last K = min(history, 12) points
    - seasonal-naive: repeat the last season (only when ``--season`` given)

Ground truth comes from the held-out target grid via
``apdtflow.event_time.batch_first_crossing_times``, anchored at offset 0
with the last input value. Windows whose last observed value is already
past the threshold are excluded (the event is not in the future there).

Metrics per method: event catch rate, timing MAE on caught events,
false-alarm rate on no-crossing windows; plus calibrated time-window
coverage for the model. Reference results produced with this methodology:
spec Sections 10.4-10.6 (battery EOL: model 3.9 cycles vs linear 9.3 vs
persistence 15.3).

Outputs a comparison table to stdout and JSON into ``experiments/results/``.

Example (smoke):
    python experiments/audit_predict_when.py \
        --csv dataset_examples/monthly-sunspots.csv --value-col Sunspots \
        --threshold 80 --epochs 1 --stride 50
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from apdtflow import APDTFlowForecaster  # noqa: E402
from apdtflow.event_time import batch_first_crossing_times  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS_DIR = REPO / "experiments" / "results"


# --------------------------------------------------------------------------
# Baseline crossing-time predictors (per held-out window, original scale)
# --------------------------------------------------------------------------

def persistence_crossings(X, threshold, direction, horizon):
    """Flat continuation of the last value: crosses only if already past
    the threshold (those windows are excluded upstream), so ~never."""
    grid = np.arange(0, horizon + 1, dtype=float)
    traj = np.repeat(X[:, -1:], horizon + 1, axis=1)
    return batch_first_crossing_times(grid, traj, threshold, direction)


def linear_crossings(X, threshold, direction, horizon, k):
    """Line through the last value with the least-squares slope of the
    last ``k`` points, extrapolated over (0, horizon]."""
    t_fit = np.arange(-k + 1, 1, dtype=float)
    tail = X[:, -k:]
    slope = np.polyfit(t_fit, tail.T, 1)[0]  # (n,)
    grid = np.arange(0, horizon + 1, dtype=float)
    traj = X[:, -1:] + slope[:, None] * grid[None, :]
    return batch_first_crossing_times(grid, traj, threshold, direction)


def seasonal_crossings(X, threshold, direction, horizon, season):
    """Repeat the last observed season, anchored at the last value."""
    n = len(X)
    grid = np.arange(0, horizon + 1, dtype=float)
    traj = np.empty((n, horizon + 1))
    traj[:, 0] = X[:, -1]
    for step in range(1, horizon + 1):
        traj[:, step] = X[:, -season + ((step - 1) % season)]
    return batch_first_crossing_times(grid, traj, threshold, direction)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def score(t_pred, t_actual, window_lo=None, window_hi=None):
    """Catch rate / timing MAE / false-alarm rate (+ optional coverage)."""
    is_event = ~np.isnan(t_actual)
    caught = is_event & ~np.isnan(t_pred)
    n_events = int(is_event.sum())
    n_quiet = int((~is_event).sum())
    out = {
        "n_events": n_events,
        "n_caught": int(caught.sum()),
        "catch_rate": float(caught.sum() / n_events) if n_events else float("nan"),
        "timing_mae": (float(np.abs(t_pred[caught] - t_actual[caught]).mean())
                       if caught.any() else float("nan")),
        "n_no_crossing_windows": n_quiet,
        "false_alarms": int((~is_event & ~np.isnan(t_pred)).sum()),
        "false_alarm_rate": (float((~is_event & ~np.isnan(t_pred)).sum() / n_quiet)
                             if n_quiet else float("nan")),
    }
    if window_lo is not None and caught.any():
        inside = ((t_actual[caught] >= window_lo[caught])
                  & (t_actual[caught] <= window_hi[caught]))
        out["window_coverage"] = float(inside.mean())
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="path to the CSV file")
    parser.add_argument("--value-col", required=True,
                        help="column holding the series values")
    parser.add_argument("--date-col", default=None,
                        help="optional date column (informational; the audit "
                             "is computed in step offsets)")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--direction", choices=["above", "below"],
                        default="above")
    parser.add_argument("--history", type=int, default=36)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1,
                        help="stride between held-out audit windows")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="miscoverage level of the calibrated time window")
    parser.add_argument("--season", type=int, default=None,
                        help="seasonal period for the seasonal-naive baseline "
                             "(omit for non-seasonal series)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--verbose", action="store_true",
                        help="show training progress bars")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = REPO / csv_path
    df = pd.read_csv(csv_path)
    if args.value_col not in df.columns:
        parser.error(f"column {args.value_col!r} not in {list(df.columns)}")
    if args.date_col:
        df = df.sort_values(args.date_col)
    values = pd.to_numeric(df[args.value_col], errors="coerce").dropna()
    series = values.values.astype(float)
    n = len(series)
    split = int(n * args.train_frac)
    if split < args.history + args.horizon + 10:
        parser.error("training fraction too small for the chosen "
                     "history/horizon")
    if args.season is not None and args.season > args.history:
        parser.error("--season must be <= --history (the seasonal-naive "
                     "baseline reads one season from the input window)")

    print(f"Series: {csv_path.name} [{args.value_col}] — {n} points, "
          f"train on first {split}, audit on the rest "
          f"(threshold {args.threshold} {args.direction})")

    # ---- fit the production forecaster on the train fraction -------------
    forecaster = APDTFlowForecaster(
        decoder_type="continuous",
        use_conformal=True,
        forecast_horizon=args.horizon,
        history_length=args.history,
        hidden_dim=32,
        num_scales=3,
        filter_size=5,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=args.epochs,
        verbose=args.verbose,
    )
    t0 = time.time()
    forecaster.fit(pd.DataFrame({"y": series[:split]}), target_col="y")
    print(f"Fitted in {time.time() - t0:.1f}s")

    # ---- held-out audit windows ------------------------------------------
    starts = list(range(split, n - args.history - args.horizon, args.stride))
    if not starts:
        parser.error("no held-out windows; reduce --history/--horizon "
                     "or --train-frac")
    X = np.stack([series[i:i + args.history] for i in starts])
    Y = np.stack([series[i + args.history:i + args.history + args.horizon]
                  for i in starts])

    # Exclude windows already past the threshold at forecast time: the
    # "when" question is only meaningful for future crossings.
    sign = 1.0 if args.direction == "above" else -1.0
    future_event = sign * (X[:, -1] - args.threshold) < 0
    n_excluded = int((~future_event).sum())
    X, Y = X[future_event], Y[future_event]
    print(f"{len(X)} audit windows ({n_excluded} excluded: already past "
          f"threshold at forecast time)")

    # ---- ground truth: first crossing of the actual target grid ----------
    grid = np.arange(0, args.horizon + 1, dtype=float)
    actual_grid = np.concatenate([X[:, -1:], Y], axis=1)  # offset-0 anchor
    t_actual = batch_first_crossing_times(
        grid, actual_grid, args.threshold, args.direction)

    # ---- model predictions + calibrated windows ---------------------------
    X_norm = (X - forecaster.scaler_mean_) / forecaster.scaler_std_
    X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)
    t_model = forecaster._batch_crossing_times(
        X_t, None, args.threshold, args.direction)
    lo, hi, n_calib = forecaster._crossing_calibration(
        args.threshold, args.direction, args.alpha)
    print(f"Calibration: {n_calib} crossings on the calibration split, "
          f"signed-error quantiles [{lo:.2f}, {hi:.2f}] "
          f"(alpha={args.alpha})")
    win_lo = t_model - hi  # earliest
    win_hi = t_model - lo  # latest

    # ---- baselines ---------------------------------------------------------
    k = min(args.history, 12)
    preds = {
        "APDTFlow predict_when": t_model,
        "persistence": persistence_crossings(
            X, args.threshold, args.direction, args.horizon),
        "linear extrapolation": linear_crossings(
            X, args.threshold, args.direction, args.horizon, k),
    }
    if args.season:
        preds["seasonal-naive"] = seasonal_crossings(
            X, args.threshold, args.direction, args.horizon, args.season)

    scores = {}
    for name, t_pred in preds.items():
        if name == "APDTFlow predict_when":
            scores[name] = score(t_pred, t_actual, win_lo, win_hi)
        else:
            scores[name] = score(t_pred, t_actual)

    # Matched subset (spec 10.6 nuance): windows where BOTH the model and
    # linear extrapolation produce an estimate on a true event.
    t_lin = preds["linear extrapolation"]
    matched = (~np.isnan(t_actual) & ~np.isnan(t_model) & ~np.isnan(t_lin))
    matched_stats = None
    if matched.any():
        matched_stats = {
            "n": int(matched.sum()),
            "model_mae": float(np.abs(t_model[matched] - t_actual[matched]).mean()),
            "linear_mae": float(np.abs(t_lin[matched] - t_actual[matched]).mean()),
        }

    # ---- table -------------------------------------------------------------
    print("\n" + "=" * 86)
    print(f"predict_when audit — {csv_path.name}, threshold "
          f"{args.threshold} {args.direction}, horizon {args.horizon}, "
          f"alpha {args.alpha}")
    print("=" * 86)
    header = (f"{'method':24s}{'events':>8s}{'caught':>8s}{'catch%':>9s}"
              f"{'timing MAE':>12s}{'false al.':>10s}{'FA%':>8s}{'coverage':>10s}")
    print(header)
    print("-" * len(header))
    for name, s in scores.items():
        cov = s.get("window_coverage")
        print(f"{name:24s}{s['n_events']:>8d}{s['n_caught']:>8d}"
              f"{100 * s['catch_rate']:>8.1f}%"
              f"{s['timing_mae']:>12.2f}"
              f"{s['false_alarms']:>10d}"
              f"{100 * s['false_alarm_rate']:>7.1f}%"
              + (f"{100 * cov:>9.1f}%" if cov is not None else f"{'—':>10s}"))
    print("-" * len(header))
    if matched_stats:
        print(f"matched subset (n={matched_stats['n']}): model MAE "
              f"{matched_stats['model_mae']:.2f} vs linear "
              f"{matched_stats['linear_mae']:.2f}")

    # ---- shipping-rule verdict ----------------------------------------------
    model_s = scores["APDTFlow predict_when"]
    beats_all = True
    for name, s in scores.items():
        if name == "APDTFlow predict_when":
            continue
        better_catch = (model_s["catch_rate"] >= s["catch_rate"]) or np.isnan(s["catch_rate"])
        better_mae = (np.isnan(s["timing_mae"])
                      or (not np.isnan(model_s["timing_mae"])
                          and model_s["timing_mae"] <= s["timing_mae"]))
        if not (better_catch and better_mae):
            beats_all = False
            print(f"shipping rule: model does NOT beat {name} "
                  f"(catch {100 * model_s['catch_rate']:.0f}% vs "
                  f"{100 * s['catch_rate']:.0f}%, timing MAE "
                  f"{model_s['timing_mae']:.2f} vs {s['timing_mae']:.2f})")
    print("SHIPPING-RULE VERDICT: "
          + ("PASS — publishable as a domain demo" if beats_all else
             "FAIL — do not publish this domain demo (see the shipping rule "
             "in this script's docstring)"))

    # ---- JSON ----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / (
        f"audit_predict_when_{csv_path.stem}_"
        f"{args.direction}{args.threshold:g}.json")
    payload = {
        "meta": {
            "script": "experiments/audit_predict_when.py",
            "csv": str(csv_path),
            "value_col": args.value_col,
            "threshold": args.threshold,
            "direction": args.direction,
            "history": args.history,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "stride": args.stride,
            "train_frac": args.train_frac,
            "alpha": args.alpha,
            "season": args.season,
            "seed": args.seed,
            "n_audit_windows": int(len(X)),
            "n_excluded_already_past": n_excluded,
            "calibration": {"lo": float(lo), "hi": float(hi),
                            "n_crossings": int(n_calib)},
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "scores": scores,
        "matched_subset_vs_linear": matched_stats,
        "shipping_rule_pass": beats_all,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
