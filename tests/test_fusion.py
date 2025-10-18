import torch
from apdtflow.models.fusion import ProbScaleFusion


def test_prob_scale_fusion():
    batch_size = 3
    num_scales = 4
    hidden_dim = 10
    latent_means = [torch.ones(batch_size, hidden_dim) * i for i in range(num_scales)]
    latent_logvars = [torch.zeros(batch_size, hidden_dim) for _ in range(num_scales)]
    fusion = ProbScaleFusion(hidden_dim, num_scales)
    fused = fusion(latent_means, latent_logvars)
    assert fused.shape == (batch_size, hidden_dim)
