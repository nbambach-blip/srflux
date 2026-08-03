"""SR-VA: the Van Atta structure-function method.

Rather than finding individual ramps, Van Atta (1977) fits an idealised ramp model to the
scalar's structure functions. For increments ``inc(r) = x(t+r) - x(t)`` and
``Sn = <inc^n>``, the ramp amplitude ``A`` solves the cubic

    A^3 + p A + q = 0 ,     p = 10 S2 - S5/S3 ,     q = 10 S3

and the total ramp period follows as ``tau = A^2 r / S2``. The root is chosen to match
sign(S3), which is what ties the amplitude to the flux direction: a warm ramp rises slowly
and collapses sharply, giving S3 < 0.

Lag selection is the whole game. A fixed 1 s lag is the textbook choice for a 20 Hz sonic,
but it COLLAPSES on a canopy skin temperature -- a smooth ~90 s surface ramp has almost no
1 s signal, and the fitted amplitude and period both go to noise. The Chen et al. (1997a)
criterion adapts the lag per block by taking the first global maximum of |S3(r)|/r, and it
recovers sensible amplitudes on both air (r* ~ 2 s) and surface (r* ~ 65 s) series. Use
``lag="chen"`` unless you are reproducing a fixed-lag analysis.

References
----------
Van Atta (1977) Arch. Mech. 29, 161-171.
Chen, Novak, Black & Yang (1997a) Boundary-Layer Meteorol. 84, 99-124 -- lag criterion.
Castellvi (2004) Agric. For. Meteorol. 122, 121-135; Shapland et al. (2012)
Boundary-Layer Meteorol. 145, 5-25 -- structure-function surface renewal in practice.
"""
from __future__ import annotations

import numpy as np

from .base import RampStats

#: Ceiling of the Chen lag search [s]. Surface ramps need tens of seconds; beyond ~2 min
#: the increments stop being a ramp signal and start being the diurnal trend.
RMAX_S = 120.0


def structure_functions(x: np.ndarray, r: int) -> tuple[float, float, float]:
    """Second, third and fifth order structure functions at integer lag ``r`` [samples]."""
    inc = x[r:] - x[:-r]
    inc2 = inc * inc
    return (float(inc2.mean()), float((inc2 * inc).mean()),
            float((inc2 * inc2 * inc).mean()))


def solve_cubic(x: np.ndarray, r: int) -> tuple[float, float]:
    """Van Atta amplitude and period at lag ``r``; returns ``(A, tau_samples)``.

    Cardano's formula, taking the real root whose sign matches S3 when three real roots
    exist. Root selection matters: picking per-lag roots without the sign constraint makes
    the solution bifurcate between branches and destroys the correlation with the flux.
    """
    if r < 1 or r >= len(x) // 4:
        return (float("nan"), float("nan"))
    S2, S3, S5 = structure_functions(x, r)
    if S3 == 0 or not np.isfinite(S3):
        return (float("nan"), float("nan"))

    p = 10 * S2 - S5 / S3
    q = 10 * S3
    disc = (q / 2) ** 2 + (p / 3) ** 3
    if disc >= 0:
        sd = np.sqrt(disc)
        A = np.cbrt(-q / 2 + sd) + np.cbrt(-q / 2 - sd)
    else:
        m = np.sqrt(-((p / 3) ** 3))
        th = np.arccos(np.clip(-q / (2 * m), -1, 1))
        roots = [2 * np.cbrt(m) * np.cos((th + 2 * np.pi * k) / 3) for k in range(3)]
        same = [rt for rt in roots if np.sign(rt) == np.sign(S3)]
        A = same[0] if same else roots[int(np.argmin(np.abs(roots)))]

    tau = (A ** 2 * r / S2) if (S2 > 0 and A != 0) else float("nan")
    return (float(A), float(tau))


def chen_lag(x: np.ndarray, fs: float = 1.0, rmax_s: float = RMAX_S) -> int:
    """Chen et al. (1997a) optimal lag: first global maximum of |S3(r)|/r."""
    n = len(x)
    rmax = min(int(rmax_s * fs), n // 4)
    if rmax < 1:
        return 0
    metric = np.full(rmax + 1, -np.inf)
    for r in range(1, rmax + 1):
        inc = x[r:] - x[:-r]
        metric[r] = abs(float((inc ** 3).mean())) / r
    if not np.isfinite(metric[1:]).any():
        return 0
    return 1 + int(np.argmax(metric[1:]))       # argmax returns the FIRST maximum


class VanAttaDetector:
    """Structure-function surface renewal (SR-VA).

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    lag : {"chen", float}
        ``"chen"`` selects the lag per block by the Chen criterion (recommended, and
        essential for skin temperature). A number is a fixed lag in seconds.
    rmax_s : float
        Ceiling of the Chen lag search [s].

    Examples
    --------
    >>> from srflux.synthetic import ramp_series
    >>> v = ramp_series(n=1800, period_s=60, amplitude=1.0, seed=0)
    >>> res = VanAttaDetector(fs=1.0).detect(v)
    >>> res.amplitude > 0 and res.period > 0
    True
    """

    name = "van_atta"

    def __init__(self, fs: float = 1.0, lag="chen", rmax_s: float = RMAX_S):
        self.fs = float(fs)
        self.lag = lag
        self.rmax_s = float(rmax_s)

    def detect(self, v) -> RampStats:
        """Solve the Van Atta cubic on one prepared block."""
        x = np.asarray(v, float)
        block_s = len(x) / self.fs
        if self.lag == "chen":
            r = chen_lag(x, self.fs, self.rmax_s)
        else:
            r = max(1, int(round(float(self.lag) * self.fs)))
        if r < 1:
            return RampStats(0, float("nan"), float("nan"), self.name, {})

        A, tau_samples = solve_cubic(x, r)
        if not (np.isfinite(A) and np.isfinite(tau_samples)) or tau_samples <= 0:
            return RampStats(0, float("nan"), float("nan"), self.name,
                             {"lag_s": r / self.fs})
        tau = tau_samples / self.fs
        count = int(round(block_s / tau)) if tau > 0 else 0
        return RampStats(count, abs(A), tau, self.name,
                         {"lag_s": r / self.fs, "signed_amplitude": A,
                          "block_s": block_s})
