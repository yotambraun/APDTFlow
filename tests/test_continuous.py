"""Tests for the continuous-time decoder: predict_at and predict_when.

The cycle-expressiveness test is a RELEASE BLOCKER: the prototype decoder
collapsed to level-only output on cyclic data; the production decoder must
express within-horizon cycles before any release.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from apdtflow import APDTFlowForecaster
from apdtflow.event_time import batch_first_crossing_times

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset_examples")


def _make_weekly_df(n=400, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture(scope="module")
def fitted_continuous():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _make_weekly_df()
    m = APDTFlowForecaster(
        forecast_horizon=14,
        history_length=28,
        num_epochs=15,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    m.fit(df, target_col="y", date_col="ds")
    return m, df


def test_predict_at_api(fitted_continuous):
    m, df = fitted_continuous

    # Fractional and beyond-horizon float offsets return finite values.
    values, lower, upper = m.predict_at([0.5, 3.7, 14.0, 17.5])
    assert np.isfinite(values).all()
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert (lower <= values).all() and (values <= upper).all()

    # Datetime queries work when fitted with a date column.
    last = pd.to_datetime(df["ds"].iloc[-1])
    v2, _, _ = m.predict_at([last + pd.Timedelta(days=2), last + pd.Timedelta(hours=36)])
    assert np.isfinite(v2).all()

    # Grid offsets match predict() shape semantics.
    grid = np.arange(1, m.forecast_horizon + 1, dtype=float)
    v_grid, _, _ = m.predict_at(grid)
    p = m.predict()
    assert v_grid.shape == p.shape

    # Querying at or before the end of the series is an error.
    with pytest.raises(ValueError):
        m.predict_at([0.0, 1.0])


def test_predict_at_requires_continuous_decoder():
    m = APDTFlowForecaster(decoder_type="transformer", num_epochs=1, verbose=False)
    df = _make_weekly_df(n=120)
    m.fit(df, target_col="y", date_col="ds")
    with pytest.raises(RuntimeError, match="decoder_type='continuous'"):
        m.predict_at([1.5])
    with pytest.raises(RuntimeError, match="decoder_type='continuous'"):
        m.predict_when(threshold=12.0)


def test_cycle_expressiveness(fitted_continuous):
    """MANDATORY: the decoder must express within-horizon cycles."""
    m, df = fitted_continuous
    preds = m.predict()
    # The true signal's within-horizon std (amplitude-5 weekly sine).
    t_future = np.arange(len(df), len(df) + m.forecast_horizon)
    truth = 10 + 5 * np.sin(2 * np.pi * t_future / 7)
    assert preds.std() >= 0.6 * truth.std(), (
        f"Forecast std {preds.std():.2f} < 60% of target std {truth.std():.2f} — "
        f"decoder collapsed to level-only output on cyclic data"
    )


@pytest.mark.slow
def test_predict_when_calibration_sunspots():
    """Time-window coverage >= 85% on held-out sunspots crossings."""
    torch.manual_seed(0)
    np.random.seed(0)
    csv = os.path.join(DATA_DIR, "monthly-sunspots.csv")
    df = pd.read_csv(csv)
    df.columns = ["month", "sunspots"]
    series = df["sunspots"].to_numpy(dtype=float)

    history, horizon, threshold, alpha = 48, 24, 80.0, 0.1
    n_train = 1700
    train_df = df.iloc[:n_train].copy()

    m = APDTFlowForecaster(
        forecast_horizon=horizon,
        history_length=history,
        num_epochs=8,
        batch_size=128,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    m.fit(train_df, target_col="sunspots")

    # Held-out windows.
    z = (series - m.scaler_mean_) / m.scaler_std_
    starts = range(n_train, len(series) - history - horizon, 4)
    X = torch.tensor(
        np.stack([z[i:i + history] for i in starts]), dtype=torch.float32
    ).unsqueeze(1)
    anchors = np.array([series[i + history - 1] for i in starts]).reshape(-1, 1)
    targets = np.stack([series[i + history:i + history + horizon] for i in starts])

    t_pred = m._batch_crossing_times(X, None, threshold, "above")
    grid_times = np.arange(0, horizon + 1, dtype=float)
    t_actual = batch_first_crossing_times(
        grid_times, np.concatenate([anchors, targets], axis=1), threshold, "above"
    )

    lo, hi, n_calib = m._crossing_calibration(threshold, "above", alpha)
    assert n_calib >= 20, f"only {n_calib} calibration crossings"

    both = ~np.isnan(t_pred) & ~np.isnan(t_actual)
    assert both.sum() >= 10, f"only {both.sum()} held-out crossings to score"
    covered = (
        (t_actual[both] >= t_pred[both] - hi) & (t_actual[both] <= t_pred[both] - lo)
    )
    coverage = covered.mean()
    assert coverage >= 0.85, f"time-window coverage {coverage:.0%} < 85%"


def test_predict_when_censored(fitted_continuous):
    m, _ = fitted_continuous
    r = m.predict_when(threshold=1e6, direction="above")
    assert r.censored
    eta, earliest, latest = r
    assert r.act_by == earliest
