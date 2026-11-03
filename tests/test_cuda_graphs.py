"""CUDA-graph training: same arithmetic as eager training, without launch overhead.

The CPU tests check the sync-free code paths the capture relies on; the CUDA
tests check that replaying a captured step trains the model exactly like the
eager loop does.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from apdtflow import APDTFlowForecaster
from apdtflow.device import mps_is_available
from apdtflow.models.apdtflow import APDTFlow
from apdtflow.models.cuda_graphs import CUDAGraphTrainStep, graph_capture_supported
from apdtflow.models.fixed_grid import rk4_fixed_step

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=requires_cuda),
    pytest.param("mps", marks=pytest.mark.skipif(not mps_is_available(), reason="MPS not available")),
]
HISTORY, HORIZON = 12, 6


def _model(decoder_type="continuous", seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = APDTFlow(num_scales=3, input_channels=1, filter_size=5, hidden_dim=8,
                     output_dim=1, forecast_horizon=HORIZON, history_length=HISTORY,
                     decoder_type=decoder_type).to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0  # deterministic, so graph and eager runs can be compared
    return model


# ---------------------------------------------------------------- sync-free paths


@pytest.mark.parametrize("device", DEVICES)
def test_forward_train_matches_forward(device):
    model = _model(device=device).eval()
    x = torch.randn(5, 1, HISTORY, device=device)
    t_span = torch.linspace(0, 1, HISTORY, device=device)
    fused, x_orig = model._encode(x, t_span)
    grid = torch.arange(1, HORIZON + 1, dtype=torch.float32, device=device)
    taus = torch.tensor([0.2, 3.5, 1.0, 5.99, 0.001, 3.5], device=device)  # incl. duplicates
    offsets = torch.cat([grid, taus])
    with torch.no_grad():
        expected, expected_lv = model.decoder(fused[:, -1, :], x_orig.squeeze(1), offsets)
        got, got_lv = model.decoder.forward_train(fused[:, -1, :], x_orig.squeeze(1), offsets)
    # Same math, but forward() solves the deduplicated offsets and forward_train()
    # the static grid, so float32 rounding may differ in the last bit on some CPUs.
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_lv, expected_lv, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_rk4_fixed_step_static_grid_matches(device):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).to(device)
    h0 = torch.randn(3, 4, device=device)
    t = torch.tensor([0.0, 0.7, 2.0, 6.0], device=device)
    func = lambda tt, h: net(h) * torch.cos(tt)  # noqa: E731
    expected = rk4_fixed_step(func, h0, t, step_size=0.25)
    got = rk4_fixed_step(func, h0, t, step_size=0.25, n_grid=25)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_graph_capture_supported_rules():
    cuda = torch.device("cuda")
    assert graph_capture_supported(cuda, "rk4", has_exog=False)
    assert not graph_capture_supported(cuda, "dopri5_adjoint", has_exog=False)
    assert not graph_capture_supported(cuda, "rk4", has_exog=True)
    assert not graph_capture_supported(torch.device("cpu"), "rk4", has_exog=False)
    assert not graph_capture_supported(torch.device("mps"), "rk4", has_exog=False)


def test_graph_step_requires_capturable_optimizer():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="capturable=True"):
        CUDAGraphTrainStep(lambda x, y: x.sum(), optimizer, batch_size=4)


def test_cuda_graphs_is_a_constructor_parameter():
    model = APDTFlowForecaster(cuda_graphs=False, device="cpu", verbose=False)
    assert model.get_params()["cuda_graphs"] is False
    assert model.set_params(cuda_graphs=True).cuda_graphs is True


# ---------------------------------------------------------------- CUDA


def _train(decoder_type, use_graph, batches, monkeypatch):
    device = "cuda"
    fixed_taus = torch.tensor([0.4, 1.7, 2.5, 3.9, 4.2, 5.8], device=device)
    monkeypatch.setattr(APDTFlow, "_training_taus", staticmethod(lambda horizon, dev: fixed_taus))
    model = _model(decoder_type, device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, capturable=True)
    t_span = torch.linspace(0, 1, HISTORY, device=device)

    def loss_fn(x, y):
        if decoder_type == "continuous":
            return model._continuous_training_loss(x, y, t_span)
        preds, _ = model(x, t_span)
        return ((preds - y.transpose(1, 2)) ** 2).mean()

    step = CUDAGraphTrainStep(loss_fn, optimizer, batch_size=batches[0][0].shape[0]) if use_graph else None
    losses = []
    for x, y in batches:
        if step is not None:
            losses.append(step(x, y))
        else:
            optimizer.zero_grad()
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
    torch.cuda.synchronize()
    return model, torch.stack(losses), step


@requires_cuda
@pytest.mark.parametrize("decoder_type", ["continuous", "transformer"])
def test_graph_training_matches_eager(decoder_type, monkeypatch):
    gen = torch.Generator().manual_seed(1)
    batches = []
    for i in range(10):
        n = 3 if i in (4, 9) else 8  # partial batches before and after capture
        batches.append((torch.randn(n, 1, HISTORY, generator=gen).cuda(),
                        torch.randn(n, 1, HORIZON, generator=gen).cuda()))
    eager, eager_losses, _ = _train(decoder_type, False, batches, monkeypatch)
    graphed, graph_losses, step = _train(decoder_type, True, batches, monkeypatch)
    assert step.captured
    torch.testing.assert_close(graph_losses, eager_losses, rtol=1e-4, atol=1e-6)
    for (name, p_eager), p_graph in zip(eager.named_parameters(), graphed.parameters()):
        torch.testing.assert_close(p_graph, p_eager, rtol=1e-4, atol=1e-6, msg=name)


def _weekly(n=240, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n, freq="D"), "y": y})


@requires_cuda
def test_forecaster_trains_with_cuda_graphs(monkeypatch):
    calls = {"captured": False}
    original = CUDAGraphTrainStep._capture

    def spy(self, x, y):
        calls["captured"] = True
        return original(self, x, y)

    monkeypatch.setattr(CUDAGraphTrainStep, "_capture", spy)
    df = _weekly()
    model = APDTFlowForecaster(forecast_horizon=7, history_length=14, num_epochs=4,
                               batch_size=16, device="cuda", decoder_type="continuous",
                               use_conformal=True, early_stopping=True, patience=10,
                               verbose=False)
    model.fit(df, target_col="y", date_col="ds")
    assert calls["captured"]
    assert np.isfinite(model.predict()).all()
    assert all(p.grad is None for p in model.model.parameters())  # training memory released
    r = model.predict_when(threshold=11.0, direction="above", alpha=0.1)
    assert r.censored in (True, False)


@requires_cuda
def test_forecaster_falls_back_to_eager_when_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CUDAGraphTrainStep must not be used with cuda_graphs=False")

    monkeypatch.setattr(CUDAGraphTrainStep, "__init__", fail)
    model = APDTFlowForecaster(forecast_horizon=7, history_length=14, num_epochs=1,
                               device="cuda", decoder_type="continuous", cuda_graphs=False,
                               verbose=False)
    model.fit(_weekly(), target_col="y", date_col="ds")
    assert np.isfinite(model.predict()).all()
