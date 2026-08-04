"""Which way is the flux going?

Neither detector's amplitude is signed in a usable way on its own -- both measure how big
the ramp is, not whether heat is leaving the surface. Four conventions are provided, in
increasing order of how much non-surface-renewal information they need.

1. :func:`sign_from_skewness` -- pure surface renewal, from the ramp shape. A warm
   (upward-flux) ramp rises gradually and collapses sharply, so the increment skewness is
   negative. The flip threshold is NOT zero: near the daytime H -> 0 transition the raw sign
   flips early, and a small positive tau (site-calibrated, ~0-0.3) fixes most of it.
   Temperature-only, no other instrument. On AIR series it is the strongest of the
   temperature-only options (92-99 % correct across three sites); on a radiometric SURFACE
   temperature it is unreliable (51-88 %), because skin temperature decouples from the flux
   direction.
2. :func:`sign_from_gradient` -- **the surface-air temperature difference**, and the most
   useful option when only a radiometer is available. An IRT reports two channels: the
   TARGET (canopy skin, Ts) and its own housing thermistor, whose temperature equilibrates
   with the AIR rather than the canopy (validated against block-mean sonic temperature:
   r = 0.979, bias -0.69 K). Their difference dT = Ts - Ta is a direct thermodynamic
   statement about the flux direction -- surface warmer than air means heat goes up -- and
   it needs no time-domain statistics and no sonic. Across three sites it was 80-87 %
   correct with a fitted offset (73-85 % at zero offset), against 61-78 % for the ramp
   skewness of the same skin temperature.
3. :func:`sign_from_stability` -- ``-sign(zeta)``. Needs a sonic for the Obukhov length but
   is ~98 % correct, because zeta encodes the buoyancy-flux sign directly. Use it whenever
   an eddy-covariance system is present and the question is not "can surface renewal stand
   alone".
4. Reference sign -- ``np.sign(H_reference)``. Only for diagnostics; it leaks the answer
   into the estimate.

Note what the housing channel is good for. Its MEAN temperature is a usable air proxy
(option 2); its RAMPS are not -- the body ramp amplitude sits at the detector noise floor
(0.014-0.021 K against 0.15-0.35 K on the target channel), and using its ramp skewness as a
sign source scored ~0.59 with a polarity that was not even consistent between sites. Use the
body temperature, not the body ramps.

The offset in :func:`sign_from_gradient` is not cosmetic. The housing is not a ventilated
air sensor: it is shaded or radiatively loaded depending on the mount, so dT crosses zero at
a site-specific value (-0.4 to -2.7 K observed). Fit it with :func:`fit_gradient_offset` on
one subset and score on another, as with any calibrated threshold.
"""
from __future__ import annotations

import numpy as np

from .preprocess import increment_skewness

#: Default flip threshold for the skewness convention. Site-calibrated values of 0.0-0.3
#: have been observed; 0.0 is the uncalibrated textbook choice.
SKEW_TAU = 0.0


def sign_from_skewness(v, fs: float = 1.0, tau: float = SKEW_TAU,
                       detrend_s: float = 180.0, lag_s: float = 1.0) -> float:
    """Ramp direction from the increment skewness of the scalar itself.

    Returns +1 (upward flux), -1 (downward) or nan when the block is too short.

    ``lag_s`` is not a detail. The increment lag must match the timescale of the fronts in
    the signal, and the default of 1 s is right only for a fast air sensor. On a radiometric
    SKIN temperature, whose ramps are an order of magnitude slower, lag-1 increments sample
    noise: over a fortnight at one orchard the skin skewness agreed with the flux direction
    76 % of the time at 1 s and 88-90 % at 4-8 s, and the day/night medians separate only at
    the longer lags (-0.27 / +0.21 at 15 s against -0.06 / +0.01 at 1 s). A skin channel that
    looks like it carries no direction information at all usually just needs a longer lag.

    Choose ``lag_s`` and ``tau`` together on a calibration window -- see
    :func:`fit_skewness_sign` -- never on the block you are predicting.
    """
    sk = increment_skewness(v, fs=fs, detrend_s=detrend_s,
                            lag=max(1, int(round(lag_s * fs))))
    if not np.isfinite(sk):
        return float("nan")
    return 1.0 if sk < tau else -1.0


def fit_skewness_sign(blocks, reference, fs: float = 1.0, detrend_s: float = 180.0,
                      lags_s=(1, 2, 3, 4, 6, 8, 10, 15, 20, 30), n_tau: int = 81):
    """Choose the increment lag and flip threshold together on a calibration set.

    ``blocks`` is a sequence of prepared 1 Hz blocks and ``reference`` the matching measured
    fluxes. Returns ``(lag_s, tau, accuracy)`` maximising agreement with ``sign(reference)``.

    Fit on days you are not scoring. The two parameters interact -- the best threshold shifts
    with the lag -- so sweeping them jointly matters; and the optimum is channel-specific
    (2-4 s for a fine wire, 3-8 s for a skin temperature in the records tested).
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


def surface_air_difference(ts, ta) -> np.ndarray:
    """dT = Ts - Ta, the surface-to-air temperature difference.

    ``ta`` is normally the IRT's own housing (body) thermistor, which equilibrates with the
    air; ``ts`` is the target (canopy skin) channel of the same instrument. Taking both from
    one instrument is what makes this a radiometer-only diagnostic.
    """
    return np.asarray(ts, float) - np.asarray(ta, float)


def sign_from_gradient(ts, ta, offset: float = 0.0) -> np.ndarray:
    """Ramp direction from the surface-air temperature difference.

    Returns +1 where ``Ts - Ta > offset`` (surface warmer than air, so heat moves upward),
    -1 below it, and nan where either input is missing.

    Examples
    --------
    >>> import numpy as np
    >>> sign_from_gradient([25.0, 18.0], [20.0, 20.0])
    array([ 1., -1.])
    """
    dt = surface_air_difference(ts, ta)
    s = np.where(dt > offset, 1.0, -1.0)
    return np.where(np.isfinite(dt), s, np.nan)


def fit_gradient_offset(ts, ta, reference_sign, grid=None) -> float:
    """Calibrate the dT flip threshold against a reference direction.

    Fit on one subset of days and score on another -- an offset fitted and scored on the
    same blocks will flatter itself. Returns the offset maximising agreement, or nan when
    there is nothing usable to fit on.
    """
    dt = surface_air_difference(ts, ta)
    ref = np.sign(np.asarray(reference_sign, float))
    m = np.isfinite(dt) & np.isfinite(ref) & (ref != 0)
    if m.sum() < 10:
        return float("nan")
    if grid is None:
        lo, hi = np.nanpercentile(dt[m], [2, 98])
        grid = np.linspace(lo, hi, 201)
    scores = [np.mean(np.where(dt[m] > o, 1.0, -1.0) == ref[m]) for o in grid]
    return float(grid[int(np.argmax(scores))])


def stability_from_gradient(ts, ta, cuts=(0.0, 0.0)) -> np.ndarray:
    """Coarse stability class from dT, as a sonic-free stand-in for zeta.

    ``cuts = (stable_below, unstable_above)`` in kelvin. Returns an array of
    ``"unstable" | "neutral" | "stable"``. Discrimination against zeta < -0.1 reached
    AUC 0.72-0.85 on dT alone across three sites, better than the ramp activity (0.65-0.82)
    and close to a logistic combination of both (0.72-0.90).
    """
    dt = surface_air_difference(ts, ta)
    lo, hi = cuts
    out = np.full(dt.shape, "neutral", dtype=object)
    out[dt > hi] = "unstable"
    out[dt < lo] = "stable"
    out[~np.isfinite(dt)] = None
    return out


def sign_from_stability(zeta) -> np.ndarray:
    """Ramp direction from the stability parameter: unstable (zeta<0) means upward flux."""
    z = np.asarray(zeta, float)
    s = -np.sign(z)
    return np.where(s == 0, 1.0, s)


def apply_sign(values, sign):
    """Attach a direction to unsigned ramp fluxes, propagating nan."""
    return np.asarray(sign, float) * np.asarray(values, float)
