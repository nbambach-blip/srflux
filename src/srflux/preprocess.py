"""Block preparation shared by every ramp detector.

A surface-renewal estimate is made on one averaging block (typically 30 min) of a high-rate
scalar time series. Both detectors in this package expect that block to be gap-free, finite
and free of the slow trend that would otherwise dominate the ramp signal, so the same three
steps are applied before either one runs:

  1. range clip   values outside a physical range become NaN (dead sensors rail, loggers
                  write sentinels such as -8190 or -3.5e8)
  2. gap fill     short gaps are interpolated, edges are filled; a block with too few valid
                  samples is rejected rather than fabricated
  3. detrend      the signal minus its centred moving mean, which is what makes the ramp
                  amplitude a property of the microfront rather than of the diurnal cycle

The detrend window differs per detector (300 s for the Haar front picker, none for Van
Atta, which is already differencing), so it is applied inside each detector, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Physically plausible range for an air or surface temperature block [deg C].
TEMP_RANGE = (-40.0, 75.0)


@dataclass(frozen=True)
class BlockQC:
    """Outcome of :func:`prepare_block`."""

    values: np.ndarray | None  #: cleaned series, or None when the block is rejected
    n_valid: int  #: finite samples before gap filling
    n_clipped: int  #: samples removed by the range clip
    ok: bool  #: whether the block passed QC

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def moving_average(v, window_s: float, fs: float = 1.0) -> np.ndarray:
    """Centred moving mean over ``window_s`` seconds.

    ``min_periods`` is half the window, so the first and last half-window are still defined
    (with a wider effective averaging period) instead of turning into NaN and truncating the
    block.
    """
    w = max(1, int(round(window_s * fs)))
    return (pd.Series(np.asarray(v, float))
            .rolling(w, center=True, min_periods=max(1, w // 2))
            .mean().to_numpy())


def detrend(v, window_s: float, fs: float = 1.0) -> np.ndarray:
    """Remove the centred moving mean -- a high-pass that keeps the ramp band."""
    v = np.asarray(v, float)
    return v - moving_average(v, window_s, fs)


def prepare_block(v, fs: float = 1.0, value_range=TEMP_RANGE, min_valid_frac: float = 0.5,
                  max_gap_s: float = 60.0) -> BlockQC:
    """Clip, gap-fill and validate one block.

    Parameters
    ----------
    v : array-like
        Raw block at ``fs`` Hz.
    value_range : tuple
        Values outside this range are treated as missing.
    min_valid_frac : float
        Minimum fraction of finite samples required, checked BEFORE gap filling so a block
        cannot be rescued by interpolating across most of itself.
    max_gap_s : float
        Longest gap that may be interpolated; a block still holding NaNs afterwards is
        rejected rather than edge-filled across a long outage.
    """
    v = np.asarray(v, float)
    n = len(v)
    if n == 0:
        return BlockQC(None, 0, 0, False)

    lo, hi = value_range
    inside = (v >= lo) & (v <= hi)
    n_clipped = int(np.isfinite(v).sum() - inside.sum())
    v = np.where(inside, v, np.nan)

    n_valid = int(np.isfinite(v).sum())
    if n_valid < min_valid_frac * n:
        return BlockQC(None, n_valid, n_clipped, False)

    limit = max(1, int(round(max_gap_s * fs)))
    s = pd.Series(v)
    # limit_area="inside" keeps interpolation between real observations; the edge fills are
    # capped at the same limit. An unqualified ffill()/bfill() would defeat max_gap_s by
    # extending the last good value across an arbitrarily long outage.
    filled = (s.interpolate(limit=limit, limit_area="inside")
               .ffill(limit=limit).bfill(limit=limit).to_numpy())
    if not np.isfinite(filled).all():
        return BlockQC(None, n_valid, n_clipped, False)
    return BlockQC(filled, n_valid, n_clipped, True)


def increment_skewness(v, fs: float = 1.0, detrend_s: float = 180.0, lag: int = 1) -> float:
    """Skewness of the lag-``lag`` increments of the detrended signal.

    This is the quantity the ramp-direction convention is built on: a warm (upward-flux)
    ramp rises gradually and falls sharply, which leaves the increment distribution
    negatively skewed. See :mod:`srflux.sign`.
    """
    d = detrend(v, detrend_s, fs)
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return float("nan")
    inc = d[lag:] - d[:-lag]
    m2 = float(np.mean(inc ** 2))
    if m2 <= 0:
        return float("nan")
    return float(np.mean(inc ** 3) / m2 ** 1.5)
