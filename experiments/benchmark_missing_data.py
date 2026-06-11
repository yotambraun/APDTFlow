"""Missing-data benchmark: plain imputation vs missingness-aware features.

PURPOSE: keep a NEGATIVE result reproducible. During the v0.4.0
investigation (spec APDTFlow_MASTER_TRANSFORMATION.md, Section 11, rule 1),
feeding the model an observation mask plus a time-since-last-observation
("delta-t") channel as exogenous features was tested at ~30% and ~50% bursty
missingness and made held-out accuracy WORSE than plain forward-fill
imputation (1.70 vs 1.66 MAE at 30%; 2.08 vs 1.98 at 50%) while LOWERING the
training loss — i.e. the extra channels enable overfitting, not skill. The
feature was therefore rejected and must not be implemented or claimed. This
script re-runs that experiment against the production API so the finding can
be re-checked whenever the model changes.

Setup, per missingness level:
    - synthetic hourly-like series: trend + daily + weekly cycles + noise
    - bursty missingness (random gap starts, geometric gap lengths)
    - variant A "impute-only":  forward-filled series -> APDTFlowForecaster
    - variant B "missingness-aware": same series, plus ``obs_mask`` and
      ``time_since_obs`` exogenous channels via the production
      ``fit(..., exog_cols=[...])`` gated-fusion path
    - baselines: seasonal-naive and linear regression on the filled series
    - all variants scored on held-out windows against the TRUE
      (pre-masking) series

The expected (negative) outcome is variant B >= variant A in MAE.

Outputs a results table to stdout and JSON to
``experiments/results/benchmark_missing_data.json``.

Usage:
    python experiments/benchmark_missing_data.py              # full run
    python experiments/benchmark_missing_data.py --epochs 1   # smoke run
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

warnings.filterwarnings("ignore")

N = 1200
T_IN, T_OUT = 36, 12
SEASON = 24
RESULTS_DIR = REPO / "experiments" / "results"

# (gap-start probability, geometric gap-length parameter) tuned so the
# realized missing fraction lands near the labelled level.
MISSINGNESS_LEVELS = {
    "30%": (0.08, 0.18),
    "50%": (0.10, 0.10),
}


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------

def make_series(rng):
    """Trend + daily + weekly cycle + noise (the Section 11.1 series)."""
    t = np.arange(N)
    return (0.02 * t + 4 * np.sin(2 * np.pi * t / 24)
            + 2 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.5, N))


def make_bursty_mask(rng, p_start, p_geom):
    """Bursty missingness: random gap starts, geometric gap lengths."""
    obs = np.ones(N, dtype=bool)
    i = 0
    while i < N:
        if rng.random() < p_start:
            gap = rng.geometric(p_geom)
            obs[i:i + gap] = False
            i += gap
        else:
            i += 1
    return obs


def forward_fill(true, obs_mask):
    """Forward-fill plus the time-since-last-observation channel."""
    filled = np.empty(N)
    delta = np.zeros(N)
    last, d = true[0], 0
    for i in range(N):
        if obs_mask[i]:
            last, d = true[i], 0
        else:
            d += 1
        filled[i], delta[i] = last, d
    return filled, delta


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def make_windows(arr, lo, hi, extra_channels=None):
    """Input windows from ``arr`` with targets from the TRUE series grid."""
    X, E = [], []
    for i in range(lo, hi - T_IN - T_OUT):
        X.append(arr[i:i + T_IN])
        if extra_channels is not None:
            E.append(np.stack([c[i:i + T_IN] for c in extra_channels]))
    X = np.asarray(X, dtype=np.float32)
    E = np.asarray(E, dtype=np.float32) if extra_channels is not None else None
    return X, E


def target_windows(true, lo, hi):
    return np.asarray(
        [true[i + T_IN:i + T_IN + T_OUT] for i in range(lo, hi - T_IN - T_OUT)],
        dtype=float)


def batch_forecast(forecaster, X_norm, exog=None):
    """Run the fitted production model over normalized test windows."""
    model = forecaster.model
    device = forecaster.device
    t_span = torch.linspace(0, 1, steps=T_IN, device=device)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_norm), 256):
            xb = torch.tensor(X_norm[start:start + 256], dtype=torch.float32,
                              device=device).unsqueeze(1)
            eb = None
            if exog is not None:
                eb = torch.tensor(exog[start:start + 256],
                                  dtype=torch.float32, device=device)
            p, _ = model(xb, t_span, exog=eb)
            preds.append(p.squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# --------------------------------------------------------------------------
# One missingness level
# --------------------------------------------------------------------------

def run_level(label, p_start, p_geom, epochs, seed, verbose=False):
    rng = np.random.default_rng(seed)
    true = make_series(rng)
    obs_mask = make_bursty_mask(rng, p_start, p_geom)
    filled, delta = forward_fill(true, obs_mask)
    missing_frac = 1 - obs_mask.mean()

    split = int(N * 0.8)
    df = pd.DataFrame({
        "y": filled,
        "obs_mask": obs_mask.astype(float),
        "time_since_obs": delta / 10.0,  # scaled as in the prototype
    })
    df_train = df.iloc[:split]
    A = target_windows(true, split, N)  # held-out truth

    common = dict(
        forecast_horizon=T_OUT, history_length=T_IN, hidden_dim=32,
        num_scales=3, filter_size=5, learning_rate=1e-3, batch_size=64,
        num_epochs=epochs, verbose=verbose,
    )
    res = {}

    # --- variant A: plain forward-fill imputation -------------------------
    m_imp = APDTFlowForecaster(**common)
    m_imp.fit(df_train, target_col="y")
    norm = (filled - m_imp.scaler_mean_) / m_imp.scaler_std_
    Xte, _ = make_windows(norm, split, N)
    P = batch_forecast(m_imp, Xte) * m_imp.scaler_std_ + m_imp.scaler_mean_
    res["APDTFlow impute-only"] = float(np.abs(P - A).mean())

    # --- variant B: mask + time-since-observation exogenous channels ------
    m_aw = APDTFlowForecaster(**common)
    m_aw.fit(df_train, target_col="y", exog_cols=["obs_mask", "time_since_obs"])
    norm = (filled - m_aw.scaler_mean_) / m_aw.scaler_std_
    exog_raw = df[["obs_mask", "time_since_obs"]].values
    exog_norm = ((exog_raw - m_aw.exog_mean_) / m_aw.exog_std_).T  # (2, N)
    Xte, Ete = make_windows(norm, split, N, extra_channels=list(exog_norm))
    P = batch_forecast(m_aw, Xte, exog=Ete) * m_aw.scaler_std_ + m_aw.scaler_mean_
    res["APDTFlow mask+delta-t"] = float(np.abs(P - A).mean())

    # --- baselines on the forward-filled series ----------------------------
    mean, std = filled[:split].mean(), filled[:split].std()
    nf = (filled - mean) / std
    Xtr, _ = make_windows(nf, 0, split)
    Ytr = target_windows(filled, 0, split)  # filled targets: same info as A/B
    Ytr = (Ytr - mean) / std
    Xte, _ = make_windows(nf, split, N)
    W, *_ = np.linalg.lstsq(
        np.hstack([Xtr, np.ones((len(Xtr), 1))]), Ytr, rcond=None)
    Pl = (np.hstack([Xte, np.ones((len(Xte), 1))]) @ W) * std + mean
    res["linear (ffill)"] = float(np.abs(Pl - A).mean())

    sn = np.stack([
        (Xte[i, -SEASON:-SEASON + T_OUT] * std + mean) for i in range(len(Xte))
    ])
    res["seasonal-naive (ffill)"] = float(np.abs(sn - A).mean())

    return {"missing_fraction": float(missing_frac), "mae_vs_true": res,
            "n_test_windows": int(len(Xte))}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40,
                        help="training epochs per variant (default 40)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true",
                        help="show per-epoch training progress bars")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    for label, (p_start, p_geom) in MISSINGNESS_LEVELS.items():
        t0 = time.time()
        results[label] = run_level(
            label, p_start, p_geom, args.epochs, args.seed,
            verbose=args.verbose)
        print(f"[{label} missingness] done in {time.time() - t0:.1f}s "
              f"(realized {results[label]['missing_fraction']:.1%})")

    print("\n" + "=" * 72)
    print(f"Missing-data benchmark — MAE vs TRUE series on held-out windows "
          f"(epochs {args.epochs})")
    print("=" * 72)
    methods = ["APDTFlow impute-only", "APDTFlow mask+delta-t",
               "seasonal-naive (ffill)", "linear (ffill)"]
    header = "method".ljust(26) + "".join(
        f"{lbl + ' missing':>18s}" for lbl in results)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = m.ljust(26)
        row += "".join(f"{results[lbl]['mae_vs_true'][m]:>18.3f}"
                       for lbl in results)
        print(row)
    print("-" * len(header))

    for lbl, r in results.items():
        imp = r["mae_vs_true"]["APDTFlow impute-only"]
        aw = r["mae_vs_true"]["APDTFlow mask+delta-t"]
        verdict = ("NEGATIVE RESULT CONFIRMED: mask features did not help"
                   if aw >= imp else
                   "NOTE: mask features helped on this run — re-examine "
                   "spec Section 11.1 before drawing conclusions")
        print(f"{lbl}: impute-only {imp:.3f} vs mask+delta-t {aw:.3f} -> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_missing_data.json"
    payload = {
        "meta": {
            "script": "experiments/benchmark_missing_data.py",
            "purpose": "reproduce the Section 11.1 negative result "
                       "(missingness features rejected)",
            "horizon": T_OUT,
            "history": T_IN,
            "epochs": args.epochs,
            "seed": args.seed,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
