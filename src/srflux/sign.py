"""Which way is the flux going?

Neither detector's amplitude is signed in a usable way on its own -- both measure how big
the ramp is, not whether heat is leaving the surface. Three conventions are provided, in
increasing order of reliability and of how much non-surface-renewal information they need.

1. :func:`sign_from_skewness` -- pure surface renewal. A warm (upward-flux) ramp rises
   gradually and collapses sharply, so the increment skewness is negative. The flip
   threshold is NOT zero: near the daytime H -> 0 transition the raw sign flips early, and
   a small positive tau (site-calibrated, ~0-0.3) fixes most of it. Temperature-only, no
   other instrument, ~70-80 % correct in the transition band.
2. :func:`sign_from_stability` -- ``-sign(zeta)``. Needs a sonic for the Obukhov length but
   is ~98 % correct, because zeta encodes the buoyancy-flux sign directly. Use it whenever
   an eddy-covariance system is present and the question is not "can surface renewal stand
   alone".
3. Reference sign -- ``np.sign(H_reference)``. Only for diagnostics; it leaks the answer
   into the estimate.

On a radiometric SURFACE temperature the skewness convention is unreliable (51-88 % correct
against 92-99 % for air), because skin temperature decouples from the flux direction. If an
air series exists, take the sign from the air and apply it to the surface channel.
"""
from __future__ import annotations

import numpy as np

from .preprocess import increment_skewness

#: Default flip threshold for the skewness convention. Site-calibrated values of 0.0-0.3
#: have been observed; 0.0 is the uncalibrated textbook choice.
SKEW_TAU = 0.0


def sign_from_skewness(v, fs: float = 1.0, tau: float = SKEW_TAU,
                       detrend_s: float = 180.0) -> float:
    """Ramp direction from the increment skewness of the scalar itself.

    Returns +1 (upward flux), -1 (downward) or nan when the block is too short.
    """
    sk = increment_skewness(v, fs=fs, detrend_s=detrend_s)
    if not np.isfinite(sk):
        return float("nan")
    return 1.0 if sk < tau else -1.0


def sign_from_stability(zeta) -> np.ndarray:
    """Ramp direction from the stability parameter: unstable (zeta<0) means upward flux."""
    z = np.asarray(zeta, float)
    s = -np.sign(z)
    return np.where(s == 0, 1.0, s)


def apply_sign(values, sign):
    """Attach a direction to unsigned ramp fluxes, propagating nan."""
    return np.asarray(sign, float) * np.asarray(values, float)
