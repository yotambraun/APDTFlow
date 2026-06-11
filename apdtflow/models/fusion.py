import torch
import torch.nn as nn


class ProbScaleFusion(nn.Module):
    """Uncertainty-weighted attention fusion across scales.

    Accepts per-scale tensors of shape ``(B, H)`` or full latent
    trajectories of shape ``(B, T, H)`` and fuses across the scale
    dimension, returning ``(B, H)`` or ``(B, T, H)`` respectively.
    """

    def __init__(self, hidden_dim, num_scales):
        super(ProbScaleFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.num_scales = num_scales

    def forward(self, latent_means, latent_logvars):
        means_stack = torch.stack(latent_means, dim=1)  # (B,S,H) or (B,S,T,H)
        hidden_dim = means_stack.size(-1)
        scores = self.attention(means_stack.reshape(-1, hidden_dim))
        scores = scores.view(*means_stack.shape[:-1], 1)
        uncert_stack = (
            torch.stack(latent_logvars, dim=1).exp().mean(dim=-1, keepdim=True)
        )
        epsilon = 1e-6
        weights = torch.softmax(scores / (uncert_stack + epsilon), dim=1)
        fused = torch.sum(weights * means_stack, dim=1)  # (B,H) or (B,T,H)
        return fused
