"""Battery remaining-useful-life playground.

Pick a forecast origin anywhere along a real NASA battery's life and see
the capacity forecast, the predicted end-of-life window, and how the RUL
estimate converges as the origin advances. Every number comes from models
trained on the *other* two cells (leave-one-battery-out) — nothing is
simulated.

Run:  python examples/battery_rul_playground.py --origin 90
      python examples/battery_rul_playground.py --cell B0006 --origin 120
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apdtflow import APDTFlowForecaster, set_seed

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "dataset_examples"
CELLS = ["B0005", "B0006", "B0007"]
EOL_THRESHOLD = 1.4  # Ah
HISTORY, HORIZON = 30, 30


def load_capacity(cell):
    df = pd.read_csv(DATA / f"nasa_{cell}.csv")
    return df["capacity"].to_numpy(dtype=float)


def fit_on_other_cells(held_out, epochs):
    train_series = np.concatenate([load_capacity(c) for c in CELLS if c != held_out])
    model = APDTFlowForecaster(
        forecast_horizon=HORIZON,
        history_length=HISTORY,
        num_epochs=epochs,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    model.fit(pd.DataFrame({"capacity": train_series}), target_col="capacity")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="B0005", choices=CELLS,
                        help="Held-out battery cell to explore.")
    parser.add_argument("--origin", type=int, default=100,
                        help="Forecast origin (cycle row index >= history length).")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    set_seed(0)
    capacity = load_capacity(args.cell)
    origin = int(np.clip(args.origin, HISTORY, len(capacity) - 1))

    print(f"Training on the other two cells (leave-{args.cell}-out)...")
    model = fit_on_other_cells(args.cell, args.epochs)

    # Forecast from the chosen origin.
    window = capacity[origin - HISTORY:origin]
    z = (window - model.scaler_mean_) / model.scaler_std_
    import torch
    X = torch.tensor(z, dtype=torch.float32).reshape(1, 1, -1)
    taus = np.linspace(0.5, HORIZON, 120)
    with torch.no_grad():
        traj, _ = model.model.forward_at(
            X, torch.linspace(0, 1, HISTORY), torch.tensor(taus, dtype=torch.float32)
        )
    traj = traj.squeeze().numpy() * model.scaler_std_ + model.scaler_mean_

    t_pred = model._batch_crossing_times(X, None, EOL_THRESHOLD, "below")[0]
    lo, hi, n_calib = model._crossing_calibration(EOL_THRESHOLD, "below", alpha=0.1)

    # Actual EOL after the origin, if it happens.
    future = capacity[origin:]
    below = np.nonzero(future < EOL_THRESHOLD)[0]
    actual_rul = int(below[0]) if len(below) else None

    print(f"\nCell {args.cell}, origin at measurement #{origin} "
          f"(capacity {capacity[origin - 1]:.3f} Ah)")
    if np.isnan(t_pred):
        print(f"  predicted: no EOL crossing within {HORIZON} measurements (censored)")
    else:
        print(f"  predicted EOL in {t_pred:.1f} measurements "
              f"(90% window {max(t_pred - hi, 0):.1f} .. {t_pred - lo:.1f}, "
              f"act_by {max(t_pred - hi, 0):.1f})")
    print(f"  actual: {'EOL in %d measurements' % actual_rul if actual_rul is not None else 'no EOL in the data'}")

    # RUL convergence as the origin advances.
    origins = list(range(HISTORY, len(capacity) - 1, 5))
    preds, actuals = [], []
    windows = np.stack([
        (capacity[o - HISTORY:o] - model.scaler_mean_) / model.scaler_std_
        for o in origins
    ])
    Xb = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
    t_preds = model._batch_crossing_times(Xb, None, EOL_THRESHOLD, "below")
    for o, tp in zip(origins, t_preds):
        fut = capacity[o:]
        b = np.nonzero(fut < EOL_THRESHOLD)[0]
        actuals.append(b[0] if len(b) else np.nan)
        preds.append(tp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(np.arange(len(capacity)), capacity, color="gray", lw=1, label="capacity")
    ax.plot(origin + taus, traj, color="C0", lw=2, label="forecast")
    ax.axhline(EOL_THRESHOLD, color="C3", lw=1, label=f"EOL {EOL_THRESHOLD} Ah")
    if not np.isnan(t_pred):
        ax.axvspan(origin + max(t_pred - hi, 0), origin + t_pred - lo,
                   color="C2", alpha=0.2, label="90% EOL window")
    if actual_rul is not None:
        ax.plot(origin + actual_rul, EOL_THRESHOLD, "k*", ms=14, label="actual EOL")
    ax.axvline(origin, color="k", ls=":", lw=1)
    ax.set_title(f"{args.cell}: forecast from origin #{origin}")
    ax.set_xlabel("measurement #")
    ax.set_ylabel("capacity (Ah)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(origins, actuals, "k--", lw=1, label="actual RUL")
    ax.plot(origins, preds, "o-", color="C0", ms=3, lw=1, label="predicted RUL")
    ax.set_title("RUL convergence as the origin advances")
    ax.set_xlabel("forecast origin (measurement #)")
    ax.set_ylabel("measurements until EOL")
    ax.legend(fontsize=8)

    out = REPO / "examples" / f"battery_rul_playground_{args.cell}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
