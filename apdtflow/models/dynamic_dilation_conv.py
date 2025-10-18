import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDilationConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DynamicDilationConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.log_dilation = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dilation = int(
            torch.clamp(torch.exp(self.log_dilation), min=1.0).round().item()
        )
        padding = (self.kernel_size - 1) * dilation // 2
        return F.conv1d(
            x, self.conv.weight, self.conv.bias, padding=padding, dilation=dilation
        )
