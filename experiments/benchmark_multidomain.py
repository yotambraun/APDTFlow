"""Multi-domain forecast accuracy benchmark for APDTFlow v0.4.0.

Measures 12-step-ahead MAE of the production ``APDTFlowForecaster`` on six
datasets spanning real and synthetic domains, and reports every number
RELATIVE TO SEASONAL-NAIVE (rel < 1.00 means "beats seasonal-naive"):

    1. electric (real, monthly)        - US electric production, yearly cycle
    2. temperature (real, daily)       - Melbourne daily minimum temperatures
    3. retail-like (mult. seasonal)    - multiplicative seasonality + trend
    4. regime-switching                - nonlinear regime shifts + cycle
    5. trend+dual-seasonality          - the Section 10.1 bug-fix series
    6. random walk (stochastic)        - honesty case: nothing should beat
                                         naive by much

Baselines: seasonal-naive, naive-last, linear (least-squares window
regression), Holt-Winters (statsmodels ExponentialSmoothing, refit per
sampled test origin). This is the production port of the v0.4.0 prototype
``benchmark_multidomain.py``; the prototype's ``fast_solver`` monkeypatch is
replaced by the production ``ode_method='rk4'`` default. Reference numbers:
spec Section 10.2 of APDTFlow_MASTER_TRANSFORMATION.md.

Outputs a results table to stdout and JSON to
``experiments/results/benchmark_multidomain.json``.

Usage:
    python experiments/benchmark_multidomain.py                  # full run
    python experiments/benchmark_multidomain.py --epochs 1 \
        --datasets "random walk (stochastic)"                    # smoke run
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

T_OUT = 12
RESULTS_DIR = REPO / "experiments" / "results"


# --------------------------------------------------------------------------
# Datasets (diverse domains; generators copied from the v0.4.0 prototype)
# --------------------------------------------------------------------------

def load_datasets():
    """Return ``{name: (series, season)}`` for all six benchmark datasets."""
    ds = {}

    # 1. REAL: monthly US electric production (trend + strong yearly cycle)
    d = pd.read_csv(REPO / "dataset_examples" / "Electric_Production.csv")
    ds["electric (real, monthly)"] = (
        d["IPG2211A2N"].values.astype(float), 12)

    # 2. REAL: daily minimum temperatures, Melbourne (noisy, yearly cycle)
    d2 = pd.read_csv(
        REPO / "dataset_examples" / "daily-minimum-temperatures-in-me_clean.csv")
    vals = pd.to_numeric(
        d2["Daily minimum temperatures"], errors="coerce").dropna().values
    ds["temperature (real, daily)"] = (vals.astype(float)[:1500], 365)

    # 3. SYNTH: multiplicative seasonality + trend (retail-like)
    rng = np.random.default_rng(0)
    n = 800
    t = np.arange(n)
    ds["retail-like (mult. seasonal)"] = (
        (50 + 0.05 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 30))
        + rng.normal(0, 2, n), 30)

    # 4. SYNTH: regime switching + noise (hard nonlinear)
    rng = np.random.default_rng(1)
    n = 800
    t = np.arange(n)
    regime = (np.sin(2 * np.pi * t / 200) > 0).astype(float)
    ds["regime-switching"] = (
        10 + regime * 8 + 3 * np.sin(2 * np.pi * t / 24) * (1 + regime)
        + rng.normal(0, 1, n), 24)

    # 5. SYNTH: trend + dual seasonality (the Section 10.1 bug-fix series)
    rng = np.random.default_rng(2)
    n = 1200
    t = np.arange(n)
    ds["trend+dual-seasonality"] = (
        0.02 * t + 4 * np.sin(2 * np.pi * t / 24)
        + 2 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.5, n), 24)

    # 6. SYNTH: random walk (honesty case)
    rng = np.random.default_rng(3)
    ds["random walk (stochastic)"] = (
        np.cumsum(rng.normal(0, 1, 800)) + 100, 1)

    return ds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def make_windows(arr, lo, hi, t_in, stride=1):
    """Sliding (input, target) windows whose inputs start in [lo, hi)."""
    X, Y = [], []
    for i in range(lo, hi - t_in - T_OUT, stride):
        X.append(arr[i:i + t_in])
        Y.append(arr[i + t_in:i + t_in + T_OUT])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def batch_forecast(forecaster, X_norm):
    """Run the fitted production model over normalized windows (n, T_in)."""
    model = forecaster.model
    device = forecaster.device
    t_span = torch.linspace(0, 1, steps=X_norm.shape[1], device=device)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_norm), 256):
            xb = torch.tensor(
                X_norm[start:start + 256], dtype=torch.float32,
                device=device).unsqueeze(1)
            p, _ = model(xb, t_span)
            preds.append(p.squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_dataset(name, series, season, epochs, verbose=False):
    series = np.asarray(series, dtype=float)
    n = len(series)
    split = int(n * 0.8)
    t_in = min(36, max(24, season if season <= 36 else 36))

    # --- APDTFlow via the production sklearn-style API -------------------
    forecaster = APDTFlowForecaster(
        forecast_horizon=T_OUT,
        history_length=t_in,
        hidden_dim=32,
        num_scales=3,
        filter_size=5,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=epochs,
        verbose=verbose,
    )
    forecaster.fit(pd.DataFrame({"y": series[:split]}), target_col="y")

    mean, std = forecaster.scaler_mean_, forecaster.scaler_std_
    norm = (series - mean) / std
    Xte, Yte = make_windows(norm, split, n, t_in, stride=1)
    if len(Xte) == 0:
        raise ValueError(f"{name}: not enough held-out data for evaluation")

    def denorm(a):
        return a * std + mean

    A = denorm(Yte)
    P = denorm(batch_forecast(forecaster, Xte))
    out = {"APDTFlow": float(np.abs(P - A).mean())}

    # --- seasonal-naive: repeat the last observed season ------------------
    m_ = min(season, t_in)
    sn = np.stack([
        denorm(np.array([Xte[i, -m_ + (k % m_)] for k in range(T_OUT)]))
        for i in range(len(Xte))
    ])
    out["seasonal-naive"] = float(np.abs(sn - A).mean())

    # --- naive-last: flat continuation of the last value ------------------
    nl = np.repeat(denorm(Xte[:, -1:]), T_OUT, axis=1)
    out["naive-last"] = float(np.abs(nl - A).mean())

    # --- linear: least-squares regression window -> horizon ---------------
    stride = 2 if n > 1000 else 1
    Xtr, Ytr = make_windows(norm, 0, split, t_in, stride)
    W, *_ = np.linalg.lstsq(
        np.hstack([Xtr, np.ones((len(Xtr), 1))]), Ytr, rcond=None)
    Pl = np.hstack([Xte, np.ones((len(Xte), 1))]) @ W
    out["linear"] = float(np.abs(denorm(Pl) - A).mean())

    # --- Holt-Winters: refit per sampled test origin on true history ------
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hw_errs = []
    for i in range(0, len(Xte), max(1, len(Xte) // 15)):
        hist = series[:split + i + t_in]
        try:
            sp = season if (1 < season <= 60 and season < len(hist) // 2) else None
            hw = ExponentialSmoothing(
                hist, trend="add", seasonal="add" if sp else None,
                seasonal_periods=sp).fit()
            hw_errs.append(np.abs(hw.forecast(T_OUT) - A[i]).mean())
        except Exception:
            pass
    out["Holt-Winters"] = float(np.mean(hw_errs)) if hw_errs else float("nan")

    return out, t_in, len(Xte)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

METHODS = ["APDTFlow", "seasonal-naive", "naive-last", "linear", "Holt-Winters"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30,
                        help="training epochs per dataset (default 30)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset of dataset names to run (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="show per-epoch training progress bars")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_ds = load_datasets()
    names = args.datasets or list(all_ds)
    unknown = [d for d in names if d not in all_ds]
    if unknown:
        parser.error(f"unknown datasets {unknown}; available: {list(all_ds)}")

    results = {}
    for name in names:
        series, season = all_ds[name]
        t0 = time.time()
        res, t_in, n_test = run_dataset(
            name, series, season, args.epochs, verbose=args.verbose)
        results[name] = {
            "mae": res,
            "rel_to_seasonal_naive": {
                k: v / res["seasonal-naive"] for k, v in res.items()},
            "T_in": t_in,
            "n_points": len(series),
            "n_test_windows": n_test,
            "season": season,
        }
        print(f"[{name}] done in {time.time() - t0:.1f}s "
              f"(T_in={t_in}, {n_test} test windows)")

    # ---------------- results table ----------------
    width = max(len(n) for n in names)
    print("\n" + "=" * (width + 16 * len(METHODS)))
    print(f"Multi-domain benchmark - MAE relative to seasonal-naive "
          f"(horizon {T_OUT}, epochs {args.epochs}; lower is better, "
          f"<1.00 beats seasonal-naive)")
    print("=" * (width + 16 * len(METHODS)))
    header = "dataset".ljust(width) + "".join(f"{m:>16s}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for name in names:
        rel = results[name]["rel_to_seasonal_naive"]
        row = name.ljust(width)
        row += "".join(f"{rel[m]:>16.2f}" for m in METHODS)
        print(row)
    print("-" * len(header))
    print("absolute MAE:")
    for name in names:
        mae = results[name]["mae"]
        row = name.ljust(width)
        row += "".join(f"{mae[m]:>16.3f}" for m in METHODS)
        print(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_multidomain.json"
    payload = {
        "meta": {
            "script": "experiments/benchmark_multidomain.py",
            "horizon": T_OUT,
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
