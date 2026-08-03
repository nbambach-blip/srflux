"""SR-WL: the Haar wavelet ramp-front detector.

The microfront at the trailing edge of a surface-renewal ramp is a step in the scalar
series, so detecting ramps is edge detection. Convolving the detrended block with a Haar
step kernel of width ``scale_s`` turns each front into a local extremum of the coefficient
series, and the ramp amplitude is read off as the median |coefficient| at those extrema.

Threshold. A fixed absolute threshold does not travel between sites or scalars, because the
ramp amplitude of a canopy skin temperature is an order of magnitude smaller than that of
air temperature. The threshold here is therefore sigma-RELATIVE: ``k * std(c)`` of the
coefficient series itself, which makes the detector scale-free. ``k = 0.75`` is the value
used throughout the calibration work this package comes from; ``k`` interacts with the
count but only weakly with the amplitude.

Scale. A sweep over 8-128 s kernels showed a broad flat optimum at 20-48 s for reproducing
eddy-covariance H, with 32 s at or within 0.1 % of the peak; the wavelet SHAPE matters even
less (Haar and a first-derivative-of-Gaussian tie, and both beat symmetric or oscillatory
kernels, which detect ramp centres rather than fronts). The defaults below are those
optima, and neither is worth tuning per site.

References
----------
Collineau & Brunet (1993) Boundary-Layer Meteorol. 65, 357-379 -- wavelet detection of
coherent structures.
Paw U et al. (1995) Boundary-Layer Meteorol. 74, 119-137 -- surface renewal for scalar
fluxes.
"""
from __future__ import annotations

import numpy as np

from ..preprocess import detrend
from .base import RampStats

#: Defaults from the wavelet scale/shape sweep (see module docstring).
HAAR_SCALE_S = 32.0
HAAR_K = 0.75
HAAR_DEDUP_S = 15.0
HAAR_DETREND_S = 300.0


def haar_kernel(scale_s: float, fs: float = 1.0) -> np.ndarray:
    """Normalised Haar step kernel spanning ``scale_s`` seconds.

    The two halves are +1/h and -1/h, so convolving a clean step of height ``a`` returns a
    coefficient of magnitude ``a`` -- the coefficient IS the amplitude, in the scalar's own
    units, with no further scaling.
    """
    h = max(1, int(round(scale_s * fs / 2)))
    return np.concatenate([np.ones(h), -np.ones(h)]) / h


def _fronts(coef: np.ndarray, threshold: float, dedup: int) -> list[int]:
    """Indices of local maxima of |coef| above ``threshold``, thinned by ``dedup`` samples.

    Thinning keeps the LARGEST coefficient within each dedup window rather than the first,
    so a front is not split into several detections by ringing on either side of it.
    """
    fronts: list[int] = []
    last = -(10 ** 9)
    a = np.abs(coef)
    for i in range(2, len(coef) - 2):
        if a[i] > threshold and a[i] >= a[i - 1] and a[i] > a[i + 1]:
            if i - last < dedup:
                if fronts and a[i] > a[fronts[-1]]:
                    fronts[-1] = i
                continue
            fronts.append(i)
            last = i
    return fronts


class HaarDetector:
    """Sigma-relative Haar front picker (SR-WL).

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    scale_s, k, dedup_s, detrend_s : float
        Kernel width, sigma-relative threshold factor, minimum front spacing and high-pass
        window, all in seconds except ``k``.
    threshold : float, optional
        Absolute threshold in scalar units. Setting it overrides the sigma-relative rule;
        provided for reproducing older fixed-threshold analyses, not recommended otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from srflux.synthetic import ramp_series
    >>> v = ramp_series(n=1800, period_s=60, amplitude=1.0, seed=0)
    >>> det = HaarDetector(fs=1.0)
    >>> res = det.detect(v)
    >>> res.count > 10 and res.amplitude > 0
    True
    """

    name = "haar"

    def __init__(self, fs: float = 1.0, scale_s: float = HAAR_SCALE_S, k: float = HAAR_K,
                 dedup_s: float = HAAR_DEDUP_S, detrend_s: float = HAAR_DETREND_S,
                 threshold: float | None = None):
        self.fs = float(fs)
        self.scale_s = float(scale_s)
        self.k = float(k)
        self.dedup_s = float(dedup_s)
        self.detrend_s = float(detrend_s)
        self.threshold = threshold
        self._kernel = haar_kernel(self.scale_s, self.fs)

    def coefficients(self, v) -> np.ndarray:
        """Haar coefficient series of the detrended block."""
        d = detrend(np.asarray(v, float), self.detrend_s, self.fs)
        return np.convolve(d, self._kernel, mode="same")

    def detect(self, v) -> RampStats:
        """Detect ramp fronts in one prepared block."""
        v = np.asarray(v, float)
        block_s = len(v) / self.fs
        coef = self.coefficients(v)
        finite = np.isfinite(coef)
        if finite.sum() < 60:
            return RampStats(0, float("nan"), float("nan"), self.name, {})

        if self.threshold is not None:
            thr = float(self.threshold)
        else:
            sigma = float(np.std(coef[finite]))
            if sigma <= 0:
                return RampStats(0, float("nan"), float("nan"), self.name, {})
            thr = self.k * sigma

        dedup = max(1, int(round(self.dedup_s * self.fs)))
        idx = _fronts(coef, thr, dedup)
        if not idx:
            return RampStats(0, float("nan"), float("nan"), self.name,
                             {"threshold": thr})
        amp = float(np.median(np.abs(coef[idx])))
        period = block_s / len(idx)
        return RampStats(len(idx), amp, period, self.name,
                         {"threshold": thr, "front_index": np.asarray(idx),
                          "block_s": block_s})
