"""Learning regression tests - these catch the class of bug fixed in v0.4.0."""
import numpy as np
import pytest
import torch

from apdtflow.models.apdtflow import APDTFlow


def _make_model(T_in=24, T_out=6):
    return APDTFlow(num_scales=3, input_channels=1, filter_size=5, hidden_dim=16,
                    output_dim=1, forecast_horizon=T_out, use_embedding=True,
                    history_length=T_in)


def test_predictions_depend_on_input():
    torch.manual_seed(0)
    m = _make_model()
    m.eval()
    t = torch.linspace(0, 1, 24)
    x1 = torch.sin(torch.linspace(0, 12, 24)).reshape(1, 1, 24)
    x2 = torch.randn(1, 1, 24) * 100
    with torch.no_grad():
        p1, _ = m(x1, t)
        p2, _ = m(x2, t)
    assert not torch.allclose(p1, p2), "Model output is independent of its input!"


@pytest.mark.slow
def test_model_learns_synthetic_signal():
    torch.manual_seed(0)
    np.random.seed(0)
    N, T_in, T_out = 600, 24, 6
    t = np.arange(N)
    s = 0.02 * t + 4 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.3, N)
    mu, sd = s[:480].mean(), s[:480].std()
    z = (s - mu) / sd
    X = torch.tensor(np.stack([z[i:i + T_in] for i in range(480 - T_in - T_out)]),
                     dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.stack([z[i + T_in:i + T_in + T_out] for i in range(480 - T_in - T_out)]),
                     dtype=torch.float32).unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y),
                                         batch_size=64, shuffle=True)
    m = _make_model(T_in, T_out)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(50):
        tot = 0.0
        for xb, yb in loader:
            ts = torch.linspace(0, 1, T_in)
            opt.zero_grad()
            p, _ = m(xb, ts)
            loss = ((p - yb.transpose(1, 2)) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(X))
    assert losses[-1] < 0.5 * losses[0]
    m.eval()  # dropout must be off for evaluation
    Xe = torch.tensor(np.stack([z[i:i + T_in] for i in range(480, N - T_in - T_out)]),
                      dtype=torch.float32).unsqueeze(1)
    Ye = np.stack([z[i + T_in:i + T_in + T_out] for i in range(480, N - T_in - T_out)])
    with torch.no_grad():
        P, _ = m(Xe, torch.linspace(0, 1, T_in))
    mae = np.abs(P.squeeze(-1).numpy() - Ye).mean()
    snaive = np.stack([Xe[i, 0, -24:-24 + T_out].numpy() for i in range(len(Xe))])
    assert mae / np.abs(snaive - Ye).mean() < 1.0, "Does not beat seasonal-naive"


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = _make_model()
    m.eval()
    x = torch.randn(2, 1, 24)
    t = torch.linspace(0, 1, 24)
    with torch.no_grad():
        p1, _ = m(x, t)
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), path)
    m2 = _make_model()
    m2.load_state_dict(torch.load(path))
    m2.eval()
    with torch.no_grad():
        p2, _ = m2(x, t)
    assert torch.allclose(p1, p2)
