"""Fixed-grid RK4 integration without per-step host synchronisation.

``torchdiffeq.odeint(..., method='rk4')`` walks its time grid with Python
comparisons of tensor values (``while t1 >= t[j]``). On a GPU every such
comparison copies a value to the host and waits for all queued kernels, once
per solver step, so the CPU can never run ahead of the GPU. The functions here
perform the same arithmetic (torchdiffeq's 3/8-rule RK4 step and its linear
interpolation between grid points) with the step bookkeeping done on the host,
so a forward pass synchronises a handful of times instead of once per step.

They are drop-in replacements for the two ``odeint`` calls APDTFlow makes and
are tested to produce bit-identical results to torchdiffeq on CPU and CUDA
(``tests/test_device.py``).
"""

from typing import Callable, Optional, Sequence, Tuple

import torch

_ONE_THIRD = 1 / 3
_TWO_THIRDS = 2 / 3


def _rk4_step(func, t0, dt, t1, y, extra=None):
    """One 3/8-rule RK4 step on a tuple state (torchdiffeq ``rk4``).

    ``extra``, if given, is a sequence of four per-stage arguments passed to
    ``func`` as a third positional argument.
    """
    args = [()] * 4 if extra is None else [(e,) for e in extra]
    k1 = func(t0, y, *args[0])
    k2 = func(t0 + dt * _ONE_THIRD,
              tuple(y_ + dt * k1_ * _ONE_THIRD for y_, k1_ in zip(y, k1)), *args[1])
    k3 = func(t0 + dt * _TWO_THIRDS,
              tuple(y_ + dt * (k2_ - k1_ * _ONE_THIRD)
                    for y_, k1_, k2_ in zip(y, k1, k2)), *args[2])
    k4 = func(t1, tuple(y_ + dt * (k1_ - k2_ + k3_)
                        for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)), *args[3])
    return tuple(
        y_ + (k1_ + 3 * (k2_ + k3_) + k4_) * dt * 0.125
        for y_, k1_, k2_, k3_, k4_ in zip(y, k1, k2, k3, k4)
    )


def rk4_on_grid(
    func: Callable[..., Sequence[torch.Tensor]],
    y0: Tuple[torch.Tensor, ...],
    t: torch.Tensor,
    stage_input: Optional[Callable[[int, int], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    """Integrate with one RK4 step between consecutive points of ``t``.

    Equivalent to ``torchdiffeq.odeint(func, y0, t, method='rk4')`` for a
    strictly increasing 1D ``t`` and a tuple state. Returns a tuple of tensors
    of shape ``(len(t), *y_i.shape)``.

    If ``stage_input(k, stage)`` is given, its result is passed to ``func`` as
    a third argument for RK4 stage ``stage`` (0-3) of step ``k``; this lets
    a caller supply time-indexed conditioning without looking it up from a
    device tensor inside ``func``.
    """
    states = [y0]
    y = y0
    for k in range(t.shape[0] - 1):
        t0 = t[k]
        t1 = t[k + 1]
        extra = None if stage_input is None else [stage_input(k, s) for s in range(4)]
        y = _rk4_step(func, t0, t1 - t0, t1, y, extra)
        states.append(y)
    return tuple(torch.stack(parts) for parts in zip(*states))


def rk4_fixed_step(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    step_size: float,
    n_grid: Optional[int] = None,
) -> torch.Tensor:
    """Fixed step-size RK4 evaluated at the (increasing) times ``t``.

    Equivalent to ``torchdiffeq.odeint(func, y0, t, method='rk4',
    options={'step_size': step_size})``: states on the grid
    ``t[0], t[0] + step_size, ...`` (last point replaced by ``t[-1]``) and
    linear interpolation between grid points. Returns ``(len(t), *y0.shape)``.

    ``n_grid`` is the number of grid points, ``ceil((t[-1] - t[0]) /
    step_size + 1)``. Pass it when the caller already knows ``t[-1]`` to skip
    the one host synchronisation this function otherwise needs, which makes
    the call capturable in a CUDA graph.
    """
    start = t[0]
    end = t[-1]
    if n_grid is None:
        # The grid length is the only value needed on the host (one sync per
        # call, as in torchdiffeq).
        n_grid = int(torch.ceil((end - start) / step_size + 1).item())
    grid = torch.arange(0, n_grid, dtype=t.dtype, device=t.device) * step_size + start
    grid[-1] = end

    def tuple_func(tt, state):
        return (func(tt, state[0]),)

    (states,) = rk4_on_grid(tuple_func, (y0,), grid)  # (G, *y0.shape)

    # Map each query time to the grid step that torchdiffeq would use: the
    # first step whose right end is >= the query.
    idx = torch.searchsorted(grid, t).clamp(1, n_grid - 1)
    t0 = grid[idx - 1]
    t1 = grid[idx]
    y_lo = states[idx - 1]
    y_hi = states[idx]
    view = (-1,) + (1,) * y0.dim()
    slope = ((t - t0) / (t1 - t0)).reshape(view)
    out = y_lo + slope * (y_hi - y_lo)
    out = torch.where((t == t1).reshape(view), y_hi, out)
    out = torch.where((t == t0).reshape(view), y_lo, out)
    # The first requested time is the initial state itself.
    out = torch.cat([y0.unsqueeze(0), out[1:]], dim=0)
    return out
