import torch
import numpy as np
from apdtflow.augmentation import jitter, scaling, time_warp

def test_jitter():
    x = torch.ones(2, 1, 50)
    x_jittered = jitter(x, sigma=0.0)
    assert torch.allclose(x, x_jittered)

def test_scaling():
    x = torch.ones(2, 1, 50)
    x_scaled = scaling(x, sigma=0.0)
    assert torch.allclose(x, x_scaled)

def test_time_warp():
    x = torch.arange(50, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    x_warped = time_warp(x, max_warp=0.1)
    assert x_warped.shape == x.shape
    assert torch.isfinite(x_warped).all()
