import torch
from apdtflow.models.dynamic_dilation_conv import DynamicDilationConv


def test_dynamic_dilation_conv_output_shape():
    x = torch.randn(2, 3, 50)
    kernel_size = 3
    conv_module = DynamicDilationConv(
        in_channels=3, out_channels=5, kernel_size=kernel_size
    )
    out = conv_module(x)
    assert out.shape == (2, 5, 50)
