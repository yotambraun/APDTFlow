import torch
import torch.nn as nn


class ProbScaleFusion(nn.Module):
    def __init__(self, hidden_dim, num_scales):
        super(ProbScaleFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.num_scales = num_scales

    def forward(self, latent_means, latent_logvars):
        batch_size, hidden_dim = latent_means[0].size()
        means_stack = torch.stack(latent_means, dim=1)
        scores = self.attention(means_stack.view(-1, hidden_dim))
        scores = scores.view(batch_size, self.num_scales, 1)
        uncert_stack = (
            torch.stack(latent_logvars, dim=1).exp().mean(dim=-1, keepdim=True)
        )
        epsilon = 1e-6
        weights = torch.softmax(scores / (uncert_stack + epsilon), dim=1)
        fused = torch.sum(weights * means_stack, dim=1)
        return fused
