"""Tests for the v0.4.0 product layer: multivariate input, fleet API,
robust persistence, and per-regime normalization."""
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from apdtflow import APDTFlowForecaster
from apdtflow.preprocessing import regime_normalize


def _degradation_df(n=260, seed=0, sensors=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    health = 100 - 0.25 * t + rng.normal(0, 0.8, n)
    df = pd.DataFrame({"health": health})
    if sensors:
        df["s1"] = health * 0.5 + rng.normal(0, 0.5, n)
        df["s2"] = rng.normal(0, 1.0, n)  # pure noise sensor
        df["s3"] = -health * 0.3 + rng.normal(0, 0.5, n)
    return df


@pytest.fixture(scope="module")
def fitted_when_model():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _degradation_df(sensors=False)
    m = APDTFlowForecaster(
        forecast_horizon=20,
        history_length=30,
        num_epochs=10,
        decoder_type="continuous",
        use_conformal=True,
        verbose=False,
    )
    m.fit(df, target_col="health")
    return m


class TestMultivariate:
    def test_fit_predict_and_importance(self):
        torch.manual_seed(0)
        df = _degradation_df()
        m = APDTFlowForecaster(
            forecast_horizon=10,
            history_length=20,
            num_epochs=3,
            decoder_type="continuous",
            use_conformal=True,
            verbose=False,
        )
        m.fit(df, target_col="health", feature_cols=["s1", "s2", "s3"])
        preds = m.predict()
        assert preds.shape == (10,)
        assert np.isfinite(preds).all()

        importance = m.sensor_importance_
        assert list(importance.index) == ["health", "s1", "s2", "s3"]
        assert np.isfinite(importance.to_numpy()).all()

        # predict_at / predict_when work on the multivariate model.
        v = m.predict_at([1.5, 5.0])
        assert np.isfinite(np.asarray(v[0] if isinstance(v, tuple) else v)).all()
        r = m.predict_when(threshold=df["health"].min() - 5, direction="below")
        assert r.eta is not None

    def test_feature_cols_validation(self):
        df = _degradation_df()
        m = APDTFlowForecaster(num_epochs=1, verbose=False)
        with pytest.raises(ValueError, match="not found"):
            m.fit(df, target_col="health", feature_cols=["nope"])
        with pytest.raises(ValueError, match="cannot be combined"):
            m.fit(df, target_col="health", feature_cols=["s1"], exog_cols=["s2"])

    def test_sensor_importance_requires_multivariate(self, fitted_when_model):
        with pytest.raises(RuntimeError, match="feature_cols"):
            fitted_when_model.sensor_importance_


class TestFleetAPI:
    def test_schedule_schema_and_sorting(self, fitted_when_model):
        m = fitted_when_model
        rng = np.random.default_rng(1)
        t = np.arange(120, dtype=float)
        fleet = {
            "fast-degrader": 60 - 0.8 * t[-40:] + rng.normal(0, 0.5, 40),
            "slow-degrader": 80 - 0.3 * t[-40:] + rng.normal(0, 0.5, 40),
            "healthy": 90 + rng.normal(0, 0.5, 40) * 0 + np.zeros(40) + 90,
        }
        schedule = m.predict_when_fleet(fleet, threshold=20.0, direction="below")

        assert list(schedule.columns) == [
            "asset_id", "eta", "earliest", "latest", "act_by", "censored", "confidence",
        ]
        assert len(schedule) == 3
        # Sorted by act_by with censored assets last.
        non_censored = schedule[~schedule["censored"]]
        assert non_censored["act_by"].is_monotonic_increasing
        if schedule["censored"].any():
            assert schedule["censored"].to_numpy()[-1]
        # act_by is the window's early edge.
        assert (schedule["act_by"] == schedule["earliest"]).all()
        # Export hooks for CMMS ingestion.
        assert isinstance(schedule.to_dict(orient="records"), list)

    def test_short_series_rejected(self, fitted_when_model):
        with pytest.raises(ValueError, match="history_length"):
            fitted_when_model.predict_when_fleet({"a": np.ones(5)}, threshold=0.0)


class TestPersistence:
    def test_roundtrip_with_calibration(self, fitted_when_model, tmp_path):
        m = fitted_when_model
        threshold = 30.0
        r_before = m.predict_when(threshold=threshold, direction="below")
        path = str(tmp_path / "model.pkl")
        m.save(path)

        loaded = APDTFlowForecaster.load(path)
        # Predictions identical after reload.
        np.testing.assert_allclose(m.predict(), loaded.predict(), rtol=1e-5)
        # Calibration state survived: same predict_when window.
        r_after = loaded.predict_when(threshold=threshold, direction="below")
        assert float(r_after.eta) == pytest.approx(float(r_before.eta), rel=1e-5)
        assert float(r_after.earliest) == pytest.approx(float(r_before.earliest), rel=1e-5)
        # Cached threshold quantiles persisted too.
        assert loaded._when_cache_

    def test_pre_04_checkpoint_rejected(self, tmp_path):
        path = tmp_path / "old.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model_type": "apdtflow"}, f)  # no apdtflow_version
        with pytest.raises(ValueError, match="0.3.x"):
            APDTFlowForecaster.load(str(path))


class TestRegimeNormalize:
    def test_removes_regime_jumps(self):
        rng = np.random.default_rng(0)
        n = 300
        regime = rng.integers(0, 3, n)
        offsets = np.array([0.0, 50.0, -30.0])[regime]
        df = pd.DataFrame({
            "op1": regime.astype(float),
            "s1": offsets + rng.normal(0, 1, n),
        })
        normed, stats = regime_normalize(df, ["op1"], ["s1"])
        # After per-regime normalization the regime jumps are gone.
        means = normed.groupby(df["op1"])["s1"].mean()
        assert np.abs(means.to_numpy()).max() < 0.2
        assert len(stats) == 3

        # Reusing training stats on new data works; unseen regime raises.
        df2 = df.iloc[:50].copy()
        normed2, _ = regime_normalize(df2, ["op1"], ["s1"], stats=stats)
        assert np.isfinite(normed2["s1"]).all()
        df_bad = pd.DataFrame({"op1": [9.0], "s1": [1.0]})
        with pytest.raises(ValueError, match="not present"):
            regime_normalize(df_bad, ["op1"], ["s1"], stats=stats)
