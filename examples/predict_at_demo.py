"""Forecast at ANY moment in time with predict_at.

Most forecasting tools answer "what will the value be at step k?".
APDTFlow's continuous-time decoder answers "what will the value be at
14:37 next Tuesday?" — fractional steps, arbitrary timestamps, and times
beyond the trained horizon — from one trained model.

Run:  python examples/predict_at_demo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apdtflow import APDTFlowForecaster, set_seed

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "dataset_examples" / "daily-minimum-temperatures-in-me_clean.csv"


def main():
    set_seed(0)
    df = pd.read_csv(CSV)
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])

    model = APDTFlowForecaster(
        forecast_horizon=14,
        history_length=60,
        num_epochs=20,
        decoder_type="continuous",   # enables predict_at / predict_when
        use_conformal=True,          # calibrated uncertainty
        verbose=True,
    )
    model.fit(df, target_col=value_col, date_col=date_col)

    # 1. The familiar grid forecast.
    grid = model.predict()

    # 2. Values at arbitrary timestamps — including *between* days and
    #    *beyond* the trained 14-day horizon.
    last_date = df[date_col].iloc[-1]
    query_times = [
        last_date + pd.Timedelta(hours=36),     # one and a half days out
        last_date + pd.Timedelta(days=7.25),    # quarter past a week
        last_date + pd.Timedelta(days=16),      # beyond the trained horizon
    ]
    values, lower, upper = model.predict_at(query_times)
    print("\nForecasts at arbitrary moments:")
    for t, v, lo, hi in zip(query_times, values, lower, upper):
        print(f"  {t}: {v:.2f}  (95% interval {lo:.2f} .. {hi:.2f})")

    # 3. A dense continuous trajectory for plotting.
    dense_offsets = np.linspace(0.1, 16, 200)
    dense, dense_lo, dense_hi = model.predict_at(dense_offsets)

    fig, ax = plt.subplots(figsize=(10, 5))
    hist = df[value_col].to_numpy()[-90:]
    hist_dates = df[date_col].to_numpy()[-90:]
    ax.plot(hist_dates, hist, color="gray", lw=1, label="history")
    dense_dates = [last_date + pd.Timedelta(days=float(o)) for o in dense_offsets]
    ax.plot(dense_dates, dense, color="C0", lw=2, label="continuous forecast")
    ax.fill_between(dense_dates, dense_lo, dense_hi, color="C0", alpha=0.2,
                    label="95% conformal band")
    grid_dates = [last_date + pd.Timedelta(days=k) for k in range(1, 15)]
    ax.plot(grid_dates, grid, "o", ms=4, color="C1", label="grid forecast (predict)")
    ax.axvline(grid_dates[-1], color="k", ls=":", lw=1, label="trained horizon")
    ax.set_title("predict_at: one model, any moment in time")
    ax.legend(loc="upper left", fontsize=8)
    out = REPO / "examples" / "predict_at_demo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
