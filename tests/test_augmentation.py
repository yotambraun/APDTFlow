import torch
from apdtflow.augmentation import jitter, scaling, time_warp


def test_jitter_no_noise():
    x = torch.ones(2, 1, 50)
    x_jit = jitter(x, sigma=0.0)
    assert torch.allclose(x, x_jit)


def test_scaling_no_change():
    x = torch.ones(2, 1, 50)
    x_scaled = scaling(x, sigma=0.0)
    assert torch.allclose(x, x_scaled)


def test_time_warp_output():
    x = torch.arange(50, dtype=torch.float32).view(1, 1, 50)
    x_warp = time_warp(x, max_warp=0.1)
    assert x_warp.shape == x.shape
    assert torch.isfinite(x_warp).all()
