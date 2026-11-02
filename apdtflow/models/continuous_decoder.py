"""Continuous-time ODE decoder.

Decodes the encoder's final latent state into forecasts at arbitrary
real-valued time offsets by integrating a small neural ODE forward in
forecast time. This is what powers ``predict_at`` (forecast at any
moment) and ``predict_when`` (event-time forecasting).
"""

import math

import torch
import torch.nn as nn

from .fixed_grid import rk4_fixed_step

# RK4 step size of the decoder ODE, in forecast steps.
_STEP_SIZE = 0.25


def _cubic_hermite_interpolate(grid_values, query, horizon):
    """Differentiably interpolate per-step grid values at fractional offsets.

    Catmull-Rom cubic Hermite interpolation over the integer grid
    ``1..horizon`` with linear extrapolation outside it.

    Args:
        grid_values: Tensor of shape ``(B, H)`` — values at offsets 1..H.
        query: 1D tensor of offsets (any positive floats).
        horizon: H, the number of grid offsets.

    Returns:
        Tensor of shape ``(B, len(query))``.
    """
    batch_size = grid_values.size(0)
    # Pad one virtual point on each side (linear continuation) so that the
    # spline is defined on the full [1, H] range and extrapolates linearly.
    left = 2 * grid_values[:, :1] - grid_values[:, 1:2]
    right = 2 * grid_values[:, -1:] - grid_values[:, -2:-1]
    padded = torch.cat([left, grid_values, right], dim=1)  # offsets 0..H+1

    q = query.clamp(min=1.0, max=float(horizon))
    base = q.floor().clamp(max=horizon - 1)  # in [1, H-1]
    frac = (q - base).unsqueeze(0).expand(batch_size, -1)
    idx = base.long()  # offset k corresponds to padded index k

    p0 = padded.gather(1, (idx - 1).unsqueeze(0).expand(batch_size, -1))
    p1 = padded.gather(1, idx.unsqueeze(0).expand(batch_size, -1))
    p2 = padded.gather(1, (idx + 1).unsqueeze(0).expand(batch_size, -1))
    p3 = padded.gather(1, (idx + 2).unsqueeze(0).expand(batch_size, -1))

    t = frac
    t2 = t * t
    t3 = t2 * t
    interp = (
        0.5 * ((2 * p1)
               + (-p0 + p2) * t
               + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
               + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    )

    # Linear extrapolation outside [1, H]. Applied unconditionally with
    # torch.where (no `if mask.any()`), so the function never waits on the
    # device and can run inside a captured CUDA graph.
    below = (query < 1.0).unsqueeze(0).expand(batch_size, -1)
    slope = grid_values[:, 1:2] - grid_values[:, :1]
    delta = (query - 1.0).unsqueeze(0).expand(batch_size, -1)
    interp = torch.where(below, grid_values[:, :1] + slope * delta, interp)
    above = (query > float(horizon)).unsqueeze(0).expand(batch_size, -1)
    slope = grid_values[:, -1:] - grid_values[:, -2:-1]
    delta = (query - float(horizon)).unsqueeze(0).expand(batch_size, -1)
    interp = torch.where(above, grid_values[:, -1:] + slope * delta, interp)
    return interp


class ContinuousODEDecoder(nn.Module):
    """Decode a latent state into forecasts at arbitrary time offsets.

    A small MLP ``dyn: (h, t) -> dh/dt`` is integrated with a fixed-step
    RK4 solver from forecast time 0 through the (sorted) query offsets; a
    linear readout decodes each latent state. Two production additions
    over the prototype design:

    1. A per-step linear skip ``nn.Linear(history_length, forecast_horizon)``
       evaluated on the raw input window at grid offsets and smoothly
       (cubic Hermite) interpolated across continuous offsets — without it
       the decoder collapses to level-only output on cyclic data.
    2. Randomized query training is supported by accepting arbitrary
       ``query_offsets`` so the training loop can sample off-grid times.

    Offsets are expressed in forecast steps: ``1.0`` is one step after the
    end of the input window; values beyond ``forecast_horizon`` are allowed
    (extrapolation beyond the trained horizon).
    """

    def __init__(self, hidden_dim, output_dim, history_length, forecast_horizon):
        super(ContinuousODEDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon

        self.dyn = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        # Per-step skip: raw window -> values at integer offsets 1..H,
        # interpolated across continuous offsets in forward().
        self.skip = nn.Linear(history_length, forecast_horizon * output_dim)

    def _dynamics(self, t, h):
        t_feature = t.reshape(1, 1).expand(h.size(0), 1)
        return self.dyn(torch.cat([h, t_feature], dim=-1))

    def forward(self, h_T, x_window, query_offsets=None):
        """
        Args:
            h_T: Final encoder latent state, shape ``(B, hidden_dim)``.
            x_window: Raw input window, shape ``(B, history_length)``.
            query_offsets: Optional 1D tensor of positive float offsets (in
                forecast steps). Defaults to the integer grid
                ``1..forecast_horizon``.

        Returns:
            Tuple ``(values, logvars)`` of shape
            ``(B, len(query_offsets), output_dim)``, ordered like the input
            offsets.
        """
        device = h_T.device
        if query_offsets is None:
            query_offsets = torch.arange(
                1, self.forecast_horizon + 1, dtype=torch.float32, device=device
            )
        query_offsets = query_offsets.to(device=device, dtype=torch.float32)
        if (query_offsets <= 0).any():
            raise ValueError("query_offsets must be strictly positive")

        sorted_offsets, sort_idx = torch.sort(query_offsets)
        # Duplicate offsets are integrated once.
        unique_offsets, inverse = torch.unique(sorted_offsets, return_inverse=True)
        values, logvars = self._decode_sorted(h_T, x_window, unique_offsets)

        # Map back from unique/sorted offsets to the caller's order.
        values = values[:, inverse, :]
        logvars = logvars[:, inverse, :]
        return self._unsort(values, logvars, sort_idx)

    def forward_train(self, h_T, x_window, query_offsets):
        """Decode at training-time offsets without any host synchronisation.

        For offsets in ``(0, forecast_horizon]`` whose largest value is exactly
        ``forecast_horizon`` (the training loss always includes the integer
        grid ``1..forecast_horizon``), this returns the same values as
        :meth:`forward`. It skips input validation and de-duplication and
        sizes the solver grid from the horizon instead of reading the largest
        offset back from the device, so a whole training step can be captured
        as a CUDA graph.
        """
        sorted_offsets, sort_idx = torch.sort(query_offsets)
        n_grid = math.ceil(self.forecast_horizon / _STEP_SIZE + 1)
        values, logvars = self._decode_sorted(h_T, x_window, sorted_offsets, n_grid=n_grid)
        return self._unsort(values, logvars, sort_idx)

    def _decode_sorted(self, h_T, x_window, offsets, n_grid=None):
        """Values and log-variances at increasing, positive ``offsets``."""
        t_eval = torch.cat([offsets.new_zeros(1), offsets])
        # Fixed step size keeps integration accuracy independent of how
        # sparse the query offsets are (e.g. a single far-out timestamp).
        # (Same arithmetic as torchdiffeq.odeint(method='rk4',
        # options={'step_size': 0.25}), without per-step host syncs.)
        states = rk4_fixed_step(
            self._dynamics, h_T, t_eval, step_size=_STEP_SIZE, n_grid=n_grid
        )  # (T+1, B, H)
        states = states[1:]  # drop tau=0

        values = self.readout(states).permute(1, 0, 2)    # (B, U, out)
        logvars = self.logvar_head(states).permute(1, 0, 2)

        # Per-step skip, interpolated across continuous offsets.
        grid = self.skip(x_window).view(
            x_window.size(0), self.forecast_horizon, self.output_dim
        )
        skip_vals = torch.stack(
            [
                _cubic_hermite_interpolate(grid[:, :, d], offsets, self.forecast_horizon)
                for d in range(self.output_dim)
            ],
            dim=-1,
        )  # (B, U, out)
        return values + skip_vals, logvars

    @staticmethod
    def _unsort(values, logvars, sort_idx):
        unsort = torch.empty_like(sort_idx)
        unsort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        return values[:, unsort, :], logvars[:, unsort, :]
