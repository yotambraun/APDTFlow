"""Event-time forecasting support: threshold crossings of forecast trajectories.

Used by :meth:`APDTFlowForecaster.predict_when`. Times are expressed in
forecast steps (1.0 = one step after the end of the input window) unless
converted to timestamps by the caller.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd

TimeLike = Union[float, pd.Timestamp]


@dataclass
class PredictWhenResult:
    """Result of an event-time forecast.

    Iterating the result yields ``(eta, earliest, latest)`` so it can be
    unpacked like a tuple.

    Attributes:
        eta: First crossing of the mean trajectory (``mode='expected'``) or
            the earliest plausible crossing (``mode='risk'``).
        earliest: Early edge of the calibrated time window.
        latest: Late edge of the calibrated time window.
        act_by: Operational deadline — equal to ``earliest``. The point
            estimate has a systematic late bias on degradation data;
            schedule by ``act_by``, never by ``eta``.
        censored: True when no crossing occurs within the horizon at this
            confidence; ``eta`` is then the horizon.
        low_confidence: True when fewer than 20 calibration crossings were
            available and the window fell back to value-space banding.
        mode: 'expected' or 'risk'.
        threshold: The threshold queried.
        direction: 'above' or 'below'.
    """

    eta: TimeLike
    earliest: TimeLike
    latest: TimeLike
    act_by: TimeLike = field(default=None)  # type: ignore[assignment]
    censored: bool = False
    low_confidence: bool = False
    mode: str = "expected"
    threshold: float = float("nan")
    direction: str = "above"

    def __post_init__(self):
        if self.act_by is None:
            self.act_by = self.earliest

    def __iter__(self) -> Iterator[TimeLike]:
        return iter((self.eta, self.earliest, self.latest))


def first_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str = "above",
) -> Optional[float]:
    """First time a piecewise-linear trajectory crosses a threshold.

    Args:
        times: 1D array of strictly increasing time points.
        values: 1D array of trajectory values at ``times``.
        threshold: Threshold level.
        direction: 'above' (crossing upward) or 'below' (crossing downward).

    Returns:
        The (linearly interpolated) crossing time, or None if the
        trajectory never crosses within ``times``. A trajectory already
        past the threshold at the first time point counts as crossing at
        ``times[0]``.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    sign = 1.0 if direction == "above" else -1.0
    excess = sign * (np.asarray(values, dtype=float) - threshold)

    if excess[0] >= 0:
        return float(times[0])
    crossings = np.nonzero((excess[:-1] < 0) & (excess[1:] >= 0))[0]
    if len(crossings) == 0:
        return None
    i = int(crossings[0])
    # Linear interpolation between the bracketing points.
    t0, t1 = float(times[i]), float(times[i + 1])
    v0, v1 = float(excess[i]), float(excess[i + 1])
    if v1 == v0:
        return t1
    return t0 + (t1 - t0) * (0.0 - v0) / (v1 - v0)


def batch_first_crossing_times(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str = "above",
) -> np.ndarray:
    """Vectorized :func:`first_crossing_time` over rows of ``values``.

    Args:
        times: 1D array of time points, shared across rows.
        values: 2D array ``(n_series, len(times))``.

    Returns:
        1D float array of crossing times with ``np.nan`` where a row never
        crosses.
    """
    out = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        t = first_crossing_time(times, values[i], threshold, direction)
        if t is not None:
            out[i] = t
    return out
