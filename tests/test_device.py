"""Device selection and device-parametrized end-to-end tests.

Every test parametrized over ``DEVICES`` runs on CPU everywhere; the CUDA and
MPS variants skip unless that hardware is present.
"""

import io
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torchdiffeq import odeint

from apdtflow import APDTFlowForecaster
from apdtflow.device import mps_is_available, resolve_device
from apdtflow.forecaster import _CPUUnpickler
from apdtflow.models.dynamics import NeuralDynamics, solve_latent_ode
from apdtflow.models.fixed_grid import rk4_fixed_step, rk4_on_grid

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
requires_mps = pytest.mark.skipif(not mps_is_available(), reason="MPS not available")

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=requires_cuda),
    pytest.param("mps", marks=requires_mps),
]


# ---------------------------------------------------------------- resolve_device


def test_auto_prefers_cuda_then_mps_then_cpu():
    expected = "cuda" if torch.cuda.is_available() else ("mps" if mps_is_available() else "cpu")
    assert resolve_device("auto").type == expected
    assert resolve_device(None).type == expected
    assert APDTFlowForecaster().device.type == expected


def test_explicit_cpu_and_torch_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_device(" CPU ") == torch.device("cpu")


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="device"):
        resolve_device("tpu")
    with pytest.raises(ValueError):
        resolve_device("not-a-device")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_requesting_missing_cuda_raises_helpfully():
    with pytest.raises(ValueError, match="CUDA build of PyTorch"):
        APDTFlowForecaster(device="cuda")


@pytest.mark.skipif(mps_is_available(), reason="MPS is available here")
def test_requesting_missing_mps_raises_helpfully():
    with pytest.raises(ValueError, match="MPS"):
        resolve_device("mps")


@requires_cuda
def test_cuda_index_out_of_range_raises():
    with pytest.raises(ValueError, match="CUDA device"):
        resolve_device(f"cuda:{torch.cuda.device_count()}")


# ------------------------------------------- solver: bit-identical to torchdiffeq


@pytest.mark.parametrize("device", DEVICES)
def test_rk4_on_grid_matches_torchdiffeq(device):
    torch.manual_seed(0)
    dynamics = NeuralDynamics(hidden_dim=8, input_dim=1).to(device)
    t_span = torch.linspace(0, 1, steps=17, device=device)
    x_seq = torch.randn(4, 17, 1, device=device)

    def func(t, state):
        h, _ = state
        idx = (torch.abs(t_span - t)).argmin()
        return dynamics(t, (h, x_seq[:, idx, :]))

    h0 = torch.randn(4, 8, device=device, requires_grad=True)
    lv0 = torch.zeros(4, 8, device=device)
    ref = odeint(func, (h0, lv0), t_span, method="rk4")
    new = rk4_on_grid(func, (h0, lv0), t_span)
    for a, b in zip(ref, new):
        assert a.shape == b.shape
        assert torch.equal(a, b)
    g_ref = torch.autograd.grad(ref[0].sum(), h0)[0]
    g_new = torch.autograd.grad(new[0].sum(), h0)[0]
    assert torch.allclose(g_ref, g_new, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_encoder_solve_matches_torchdiffeq_nearest_point_lookup(device):
    # 0.4 looked up the conditioning value nearest to each RK4 stage time
    # with an argmin on the device; 0.5 resolves it on the host. Same bits.
    torch.manual_seed(2)
    dynamics = NeuralDynamics(hidden_dim=16, input_dim=1).to(device)
    t_span = torch.linspace(0, 1, steps=30, device=device)
    x_seq = torch.randn(5, 30, 1, device=device)

    def func(t, state):
        h, _ = state
        idx = (torch.abs(t_span - t)).argmin()
        return dynamics(t, (h, x_seq[:, idx, :]))

    h0 = torch.zeros(5, 16, device=device)
    ref = odeint(func, (h0, h0.clone()), t_span, method="rk4")
    h_new, lv_new = solve_latent_ode(dynamics, h0, h0.clone(), t_span, x_seq)
    assert torch.equal(ref[0].transpose(0, 1), h_new)
    assert torch.equal(ref[1].transpose(0, 1), lv_new)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("end", [14.0, 13.7, 0.3, 41.0])
def test_rk4_fixed_step_matches_torchdiffeq(device, end):
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(9, 16), torch.nn.Tanh(),
                              torch.nn.Linear(16, 8)).to(device)

    def func(t, h):
        return net(torch.cat([h, t.reshape(1, 1).expand(h.size(0), 1)], dim=-1))

    h0 = torch.randn(3, 8, device=device)
    offsets = torch.sort(torch.unique(torch.cat([
        torch.rand(12) * end, torch.tensor([0.25, 1.0, 2.0, end]),
    ]).clamp(1e-3, end)))[0].to(device)
    t_eval = torch.cat([torch.zeros(1, device=device), offsets])
    ref = odeint(func, h0, t_eval, method="rk4", options={"step_size": 0.25})
    new = rk4_fixed_step(func, h0, t_eval, step_size=0.25)
    assert ref.shape == new.shape
    assert torch.equal(ref, new)


# ------------------------------------------------------------ end-to-end on device


def _series(n=260, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 10 + np.sin(t / 6.0) * 2 + t * 0.02 + rng.normal(0, 0.1, n)


def _assert_on(model, device):
    expected = resolve_device(device).type
    assert model.device.type == expected
    assert all(p.device.type == expected for p in model.model.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_pipeline_runs_on_device(device, tmp_path):
    torch.manual_seed(0)
    y = _series()
    model = APDTFlowForecaster(
        forecast_horizon=10, history_length=24, num_epochs=2, batch_size=32,
        decoder_type="continuous", use_conformal=True, device=device, verbose=False,
    )
    model.fit(pd.DataFrame({"y": y}), target_col="y")
    _assert_on(model, device)

    preds = model.predict()
    assert preds.shape == (10,) and np.isfinite(preds).all()
    lo, mid, hi = model.predict(return_intervals="conformal")
    assert (lo <= hi).all()
    values, v_lo, v_hi = model.predict_at([0.5, 3.25, 12.0])
    assert values.shape == (3,) and np.isfinite(values).all()
    result = model.predict_when(threshold=float(np.median(y)), direction="above")
    assert result.earliest <= result.latest
    fleet = model.predict_when_fleet({"a": y[-50:], "b": y[-80:-20]},
                                     threshold=float(np.median(y)))
    assert list(fleet["asset_id"]) and set(fleet["asset_id"]) == {"a", "b"}
    report = model.score_recent(y[-60:])
    assert np.isfinite(report["mae"])

    # Checkpoints hold CPU tensors and load onto any device.
    path = tmp_path / "m.pkl"
    model.save(str(path))
    with open(path, "rb") as f:
        state = pickle.load(f)
    assert all(v.device.type == "cpu" for v in state["model_state_dict"].values())
    for target in ("cpu", device):
        loaded = APDTFlowForecaster.load(str(path), device=target)
        _assert_on(loaded, target)
        np.testing.assert_allclose(loaded.predict(), preds, rtol=1e-4, atol=1e-4)

    # Moving a fitted model keeps its predictions.
    model.to("cpu")
    _assert_on(model, "cpu")
    np.testing.assert_allclose(model.predict(), preds, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_multivariate_and_exog_on_device(device):
    torch.manual_seed(0)
    y = _series(220)
    df = pd.DataFrame({"y": y, "s1": y * 0.5 + 1, "s2": np.cos(np.arange(220) / 5)})
    mv = APDTFlowForecaster(forecast_horizon=6, history_length=20, num_epochs=1,
                            decoder_type="continuous", use_conformal=True,
                            device=device, verbose=False)
    mv.fit(df, target_col="y", feature_cols=["s1", "s2"])
    _assert_on(mv, device)
    assert mv.sensor_importance_.shape == (3,)
    assert mv.predict().shape == (6,)

    ex = APDTFlowForecaster(forecast_horizon=6, history_length=20, num_epochs=1,
                            device=device, verbose=False)
    ex.fit(df, target_col="y", exog_cols=["s2"], future_exog_cols=["s2"])
    _assert_on(ex, device)
    assert ex.predict(exog_future=df[["s2"]].tail(6)).shape == (6,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_type", ["transformer", "tcn"])
def test_grid_models_on_device(device, model_type):
    torch.manual_seed(0)
    model = APDTFlowForecaster(model_type=model_type, forecast_horizon=5,
                               history_length=20, num_epochs=1, device=device,
                               verbose=False)
    model.fit(_series(150))
    _assert_on(model, device)
    assert model.predict().shape == (5,)


@pytest.mark.parametrize("device", DEVICES)
def test_dopri5_adjoint_on_device(device):
    torch.manual_seed(0)
    dynamics = NeuralDynamics(hidden_dim=4, input_dim=1).to(device)
    h0 = torch.zeros(2, 4, device=device)
    t_span = torch.linspace(0, 1, steps=6, device=device)
    x_seq = torch.randn(2, 6, 1, device=device)
    h_sol, _ = solve_latent_ode(dynamics, h0, h0.clone(), t_span, x_seq,
                                ode_method="dopri5_adjoint")
    assert h_sol.shape == (2, 6, 4) and h_sol.device.type == resolve_device(device).type


@requires_cuda
def test_cuda_tensors_in_old_checkpoints_map_to_cpu():
    # apdtflow <= 0.4 pickled state dicts as-is, so GPU-trained checkpoints
    # held CUDA tensors that failed to load on CPU-only machines.
    buf = io.BytesIO()
    pickle.dump({"w": torch.ones(3, device="cuda")}, buf)
    buf.seek(0)
    restored = _CPUUnpickler(buf).load()
    assert restored["w"].device.type == "cpu"
    assert torch.equal(restored["w"], torch.ones(3))


# ------------------------------------------------- held-out conformal calibration


def _n_windows(n, history, horizon):
    return n - history - horizon + 1


def test_calibration_split_holds_out_recent_windows():
    n, hist, hor = 260, 24, 10
    model = APDTFlowForecaster(forecast_horizon=hor, history_length=hist, num_epochs=1,
                               decoder_type="continuous", use_conformal=True,
                               calibration_split=0.25, device="cpu", verbose=False)
    model.fit(_series(n))
    windows = _n_windows(n, hist, hor)
    assert len(model._calib_X_) == round(windows * 0.25)
    # The calibration windows are the most recent ones, in order.
    y = (_series(n) - model.scaler_mean_) / model.scaler_std_
    np.testing.assert_allclose(model._calib_X_[-1, 0].numpy(), y[n - hor - hist:n - hor],
                               rtol=1e-5, atol=1e-5)


def test_calibration_is_in_sample_by_default():
    # The protocol of every published audit; held-out calibration is opt-in.
    assert APDTFlowForecaster().calibration_split == 0.0


def test_calibration_split_zero_is_in_sample():
    n, hist, hor = 200, 20, 5
    model = APDTFlowForecaster(forecast_horizon=hor, history_length=hist, num_epochs=1,
                               use_conformal=True, calibration_split=0.0,
                               device="cpu", verbose=False)
    model.fit(_series(n))
    assert len(model._calib_X_) == _n_windows(n, hist, hor)


def test_calibration_split_falls_back_when_data_is_short():
    model = APDTFlowForecaster(forecast_horizon=20, history_length=20, num_epochs=1,
                               use_conformal=True, calibration_split=0.45,
                               device="cpu", verbose=False)
    with pytest.warns(UserWarning, match="in-sample"):
        model.fit(_series(60))
    assert model.conformal_predictor is not None


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_calibration_split_validated(bad):
    model = APDTFlowForecaster(forecast_horizon=5, history_length=20, num_epochs=1,
                               use_conformal=True, calibration_split=bad,
                               device="cpu", verbose=False)
    with pytest.raises(ValueError, match="calibration_split"):
        model.fit(_series(120))


def test_conformal_with_unsupported_model_fails_before_training():
    model = APDTFlowForecaster(model_type="tcn", use_conformal=True, num_epochs=500,
                               device="cpu", verbose=False)
    with pytest.raises(ValueError, match="Conformal prediction is only supported"):
        model.fit(_series(120))


def test_backtest_retrain_folds_keep_configuration_and_device(monkeypatch):
    seen = []
    original_init = APDTFlowForecaster.__init__

    def spy(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.append((self.hidden_dim, self.learning_rate, self.device.type))

    monkeypatch.setattr(APDTFlowForecaster, "__init__", spy)
    model = APDTFlowForecaster(forecast_horizon=5, history_length=20, num_epochs=1,
                               hidden_dim=8, learning_rate=0.005, device="cpu",
                               verbose=False)
    df = pd.DataFrame({"y": _series(220)})
    model.historical_forecasts(df, target_col="y", start=0.85, stride=10, retrain=True)
    assert len(seen) > 1
    assert all(entry == (8, 0.005, "cpu") for entry in seen)
