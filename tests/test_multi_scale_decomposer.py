import torch
from apdtflow.models.multi_scale_decomposer import ResidualMultiScaleDecomposer


def test_decomposer_output():
    batch_size = 2
    input_channels = 1
    length = 50
    num_scales = 3
    filter_size = 5
    x = torch.randn(batch_size, input_channels, length)
    decomposer = ResidualMultiScaleDecomposer(num_scales, input_channels, filter_size)
    outputs = decomposer(x)
    assert isinstance(outputs, list)
    assert len(outputs) == num_scales
    for out in outputs:
        assert out.shape == (batch_size, 1, length)
