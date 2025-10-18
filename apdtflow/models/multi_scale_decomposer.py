import torch.nn as nn
from .dynamic_dilation_conv import DynamicDilationConv


class ResidualMultiScaleDecomposer(nn.Module):
    def __init__(self, num_scales, input_channels, filter_size):
        super(ResidualMultiScaleDecomposer, self).__init__()
        self.num_scales = num_scales
        self.paths = nn.ModuleList()
        for _ in range(num_scales):
            self.paths.append(
                nn.Sequential(
                    DynamicDilationConv(input_channels, input_channels, filter_size),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU(),
                    nn.Conv1d(input_channels, 1, filter_size, padding=filter_size // 2),
                    nn.BatchNorm1d(1),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        return [path(x) for path in self.paths]
