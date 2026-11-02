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
        self._dilation_cache = (-1, 1)  # (parameter version, dilation)

    def _dilation(self) -> int:
        # The dilation is an integer, so reading it needs a host copy; on a
        # GPU that is a synchronisation point. Recompute it only when the
        # parameter has changed (load_state_dict, .to(), manual edits):
        # rounding to an int gives the parameter no gradient, so training
        # never moves it.
        param = self.log_dilation
        key = (param._version, param.data_ptr())
        cache = getattr(self, '_dilation_cache', (None, 1))
        if key != cache[0]:
            cache = (key, int(torch.clamp(torch.exp(param), min=1.0).round().item()))
            self._dilation_cache = cache
        return cache[1]

    def forward(self, x):
        dilation = self._dilation()
        padding = (self.kernel_size - 1) * dilation // 2
        return F.conv1d(
            x, self.conv.weight, self.conv.bias, padding=padding, dilation=dilation
        )
