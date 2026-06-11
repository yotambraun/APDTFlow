"""Forecast WHEN a threshold will be crossed with predict_when.

"When will solar activity rise above 80?" — predict_when answers with a
calibrated time window, not just a point guess, and returns an act_by
date (the window's early edge) to schedule against.

Run:  python examples/predict_when_demo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apdtflow import APDTFlowForecaster, set_seed

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "dataset_examples" / "monthly-sunspots.csv"
THRESHOLD = 80.0


def main():
    set_seed(0)
    df = pd.read_csv(CSV)
    df.columns = ["month", "sunspots"]

    # Fit up to the start of a rising solar cycle so a crossing lies ahead.
    origin = 2740
    train = df.iloc[origin - 1100:origin]  # ~90 years of monthly data

    model = APDTFlowForecaster(
        forecast_horizon=24,
        history_length=48,
        num_epochs=8,
        batch_size=128,
        decoder_type="continuous",
        use_conformal=True,
        verbose=True,
    )
    model.fit(train, target_col="sunspots")

    result = model.predict_when(THRESHOLD, direction="above", alpha=0.1)
    print(f"\nWhen will sunspots rise above {THRESHOLD:.0f}?")
    print(f"  point estimate (eta): t+{result.eta:.1f} months")
    print(f"  90% time window:      t+{result.earliest:.1f} .. t+{result.latest:.1f}")
    print(f"  act_by:               t+{result.act_by:.1f}  <- schedule by this, not eta")
    print(f"  censored:             {result.censored}")

    # Risk mode: from when is the crossing *plausible*?
    risk = model.predict_when(THRESHOLD, direction="above", mode="risk", alpha=0.1)
    print(f"  plausible from (risk mode): t+{risk.eta:.1f}")

    # Plot trajectory + window + what actually happened.
    taus = np.linspace(0.1, 24, 200)
    traj = model.predict_at(taus)
    traj = traj[0] if isinstance(traj, tuple) else traj
    actual = df["sunspots"].to_numpy()[origin:origin + 24]

    fig, ax = plt.subplots(figsize=(10, 5))
    hist = train["sunspots"].to_numpy()[-72:]
    ax.plot(np.arange(-72, 0), hist, color="gray", lw=1, label="history")
    ax.plot(taus, traj, color="C0", lw=2, label="forecast trajectory")
    ax.plot(np.arange(1, len(actual) + 1), actual, color="k", lw=1, ls="--",
            label="what actually happened")
    ax.axhline(THRESHOLD, color="C3", ls="-", lw=1, label=f"threshold {THRESHOLD:.0f}")
    if not result.censored:
        ax.axvspan(result.earliest, result.latest, color="C2", alpha=0.2,
                   label="90% time window")
        ax.axvline(result.eta, color="C2", ls=":", lw=2)
    ax.set_xlabel("months from forecast origin")
    ax.set_title(f'predict_when: "when does it cross {THRESHOLD:.0f}?" — calibrated window')
    ax.legend(loc="upper left", fontsize=8)
    out = REPO / "examples" / "predict_when_demo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
