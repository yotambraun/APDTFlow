import torch
import numpy as np


def jitter(x, sigma=0.03):
    """Add random Gaussian noise."""
    return x + sigma * torch.randn_like(x)


def scaling(x, sigma=0.1):
    """Multiply by a random scaling factor."""
    factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
    return x * factor


def time_warp(x, max_warp=0.2):
    """Time warp via re–interpolation of the series."""
    batch_size, channels, length = x.size()
    warp = (
        torch.linspace(0, 1, steps=length, device=x.device)
        + (torch.rand(length, device=x.device) - 0.5) * max_warp
    )
    warp, _ = torch.sort(warp)
    orig_idx = torch.linspace(0, 1, steps=length, device=x.device)
    x_warped = torch.zeros_like(x)
    for b in range(batch_size):
        for c in range(channels):
            x_np = x[b, c, :].cpu().numpy()
            warp_np = warp.cpu().numpy()
            orig_np = orig_idx.cpu().numpy()
            interp_np = np.interp(orig_np, warp_np, x_np)
            x_warped[b, c, :] = torch.tensor(interp_np, device=x.device)
    return x_warped
