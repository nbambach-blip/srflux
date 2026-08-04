"""Flux direction.

A ramp detector measures how big the ramp is, not whether heat is leaving the surface, so
the sign must come from somewhere else.

:func:`sign_from_skewness` is the temperature-only option. A warm (upward-flux) ramp rises
gradually and collapses sharply, leaving the increment distribution negatively skewed. Two
parameters control it and both must be fitted on a calibration window with
:func:`fit_skewness_sign`:

* ``lag_s`` -- the increment lag, which must match the timescale of the fronts. The 1 s
  default suits a fast air sensor; a radiometric skin temperature needs several seconds.
* ``tau`` -- the flip threshold, which is not zero. Near the daytime H -> 0 transition the
  raw sign flips early, and a small positive tau corrects it.

:func:`sign_from_stability` uses ``-sign(zeta)`` where an eddy-covariance system is present.
"""
from __future__ import annotations

import numpy as np

from .preprocess import increment_skewness

#: Default flip threshold. Fit it per site and channel; this is only a starting point.
SKEW_TAU = 0.0


def sign_from_skewness(v, fs: float = 1.0, tau: float = SKEW_TAU,
                       detrend_s: float = 180.0, lag_s: float = 1.0) -> float:
    """Ramp direction from the increment skewness of the scalar itself.

    Returns +1 (upward flux), -1 (downward), or nan when the block is too short.

    Examples
    --------
    >>> from srflux.synthetic import ramp_series
    >>> sign_from_skewness(ramp_series(n=1800, period_s=60, amplitude=1.0, warm=True))
    1.0
    """
    sk = increment_skewness(v, fs=fs, detrend_s=detrend_s,
                            lag=max(1, int(round(lag_s * fs))))
    if not np.isfinite(sk):
        return float("nan")
    return 1.0 if sk < tau else -1.0


def fit_skewness_sign(blocks, reference, fs: float = 1.0, detrend_s: float = 180.0,
                      lags_s=(1, 2, 3, 4, 6, 8, 10, 15, 20, 30), n_tau: int = 81):
    """Choose the increment lag and flip threshold together on a calibration set.

    ``blocks`` is a sequence of prepared blocks, ``reference`` the matching measured fluxes.
    Returns ``(lag_s, tau, accuracy)``. The two parameters interact, so sweep them jointly,
    and fit on days you are not scoring.
    """
    ref = np.sign(np.asarray(reference, float))
    prepared = [np.asarray(b, float) for b in blocks]
    best = (float("nan"), float("nan"), -np.inf)
    for lag_s in lags_s:
        lag = max(1, int(round(lag_s * fs)))
        sk = np.array([increment_skewness(b, fs=fs, detrend_s=detrend_s, lag=lag)
                       for b in prepared])
        m = np.isfinite(sk) & np.isfinite(ref) & (ref != 0)
        if m.sum() < 30:
            continue
        lo, hi = np.nanpercentile(sk[m], [2, 98])
        for tau in np.linspace(lo, hi, n_tau):
            acc = float(np.mean(np.where(sk[m] < tau, 1.0, -1.0) == ref[m]))
            if acc > best[2]:
                best = (float(lag_s), float(tau), acc)
    return best


def sign_from_stability(zeta) -> np.ndarray:
    """Ramp direction from the stability parameter: unstable (zeta < 0) means upward flux."""
    z = np.asarray(zeta, float)
    s = -np.sign(z)
    return np.where(s == 0, 1.0, s)
