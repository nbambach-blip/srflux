"""From ramp statistics to a calibrated flux, and from flux to daily ET.

Both detectors produce an UNCALIBRATED flux that has the right shape but the wrong scale,
because a point sensor sees only part of the renewed volume and the idealised ramp is not
the real one:

    SR-WL / count form   F = rho cp z  N A / dt        (amplitude x how often it renews)
    SR-VA / period form  F = rho cp z  A / tau         (amplitude / renewal period)

A single dimensionless coefficient ``alpha`` closes the gap, fitted through the origin
against a reference flux. Through-origin is deliberate: it is the minimum-RMSE estimator
and leaves no intercept for noise to hide in, whereas variance matching gives a slope near
one at the cost of a bias that can reach 100 W m-2.

The length scale ``z`` is a convention, not a measurement, and only the product ``alpha*z``
is identifiable from data. For an AIR temperature it is the measurement height; for a
radiometric SURFACE temperature the renewed volume is the canopy layer, so canopy height is
the physical choice. Two studies that pick differently cannot compare alphas -- compare
``alpha*z`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CP_AIR = 1005.0  #: specific heat of air at constant pressure [J kg-1 K-1]
RHO_AIR = 1.19  #: default air density [kg m-3]
LAMBDA_V = 2.45e6  #: latent heat of vaporisation [J kg-1]
BLOCK_S = 1800.0  #: default averaging period [s]


def ramp_flux(amplitude, count=None, period=None, z: float = 1.0, rho: float = RHO_AIR,
              cp: float = CP_AIR, block_s: float = BLOCK_S):
    """Uncalibrated surface-renewal flux [W m-2] for either ramp form.

    Give ``count`` for the count form (``N A / block``) or ``period`` for the period form
    (``A / tau``). The two are the same quantity when ``period = block/count``; they differ
    in practice because Van Atta's tau is a fitted per-block quantity and noisier than a
    count.
    """
    amplitude = np.asarray(amplitude, float)
    if (count is None) == (period is None):
        raise ValueError("give exactly one of count= or period=")
    rate = (np.asarray(count, float) / block_s if count is not None
            else 1.0 / np.asarray(period, float))
    return np.asarray(rho, float) * cp * z * amplitude * rate


def calibrate_alpha(F, reference) -> float:
    """Through-origin calibration ``alpha = sum(F*H) / sum(F^2)``.

    ``F`` is the uncalibrated flux, ``reference`` the measured flux to match (eddy
    covariance, ideally energy-balance-closure corrected). Pairs with a non-finite entry are
    dropped. Fit on a single well-defined regime -- convective daytime blocks of one sign --
    so that the ramp direction never enters the calibration.
    """
    F = np.asarray(F, float)
    H = np.asarray(reference, float)
    m = np.isfinite(F) & np.isfinite(H)
    if m.sum() < 2:
        return float("nan")
    denom = float(np.sum(F[m] ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(F[m] * H[m]) / denom)


def sensible_heat(F, alpha: float, sign=1.0):
    """Calibrated sensible heat flux ``H = sign * alpha * F`` [W m-2].

    ``sign`` carries the ramp direction (+1 upward). Apply it here, not inside the
    calibration: fitting alpha on unsigned F and then multiplying by the sign flips
    stable-regime blocks twice.
    """
    return np.asarray(sign, float) * alpha * np.asarray(F, float)


def scores(F, reference, alpha: float) -> dict:
    """Skill of ``alpha * F`` against a reference flux.

    Reports ``nse`` (Nash-Sutcliffe efficiency, ``1 - MSE/var``) alongside r, RMSE and bias.
    Read NSE: r is blind to scale, so an estimate with the right shape but the wrong magnitude
    still scores well, while NSE is zero for a prediction no better than the mean of the
    observations and negative for one that is worse.

    Examples
    --------
    >>> import numpy as np
    >>> H = np.r_[np.full(10, 150.0), np.full(10, -50.0)]     # day, then night
    >>> F = np.sign(H) * 1e-4                                 # right sign, no magnitude
    >>> sc = scores(F, H, 1.0)
    >>> bool(sc["r"] > 0.99), bool(sc["nse"] < 0)
    (True, True)
    """
    F = np.asarray(F, float)
    H = np.asarray(reference, float)
    e = alpha * F - H
    var = float(np.var(H))
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mbe=float(np.mean(e)),
                r=float(np.corrcoef(F, H)[0, 1]) if len(F) > 2 else float("nan"),
                nse=float(1.0 - np.mean(e ** 2) / var) if var > 0 else float("nan"))


@dataclass(frozen=True)
class AlphaFit:
    """Result of :func:`fit_and_score`."""

    alpha: float
    n: int
    r: float
    rmse: float
    mbe: float
    nse: float = float("nan")

    def apply(self, F, sign=1.0):
        """Calibrated flux for new blocks."""
        return sensible_heat(F, self.alpha, sign)


def fit_and_score(F, reference) -> AlphaFit:
    """Calibrate alpha and report how well it reproduces the reference."""
    F = np.asarray(F, float)
    H = np.asarray(reference, float)
    m = np.isfinite(F) & np.isfinite(H)
    a = calibrate_alpha(F[m], H[m])
    sc = scores(F[m], H[m], a)
    return AlphaFit(a, int(m.sum()), sc["r"], sc["rmse"], sc["mbe"], sc["nse"])


def latent_heat_residual(net_radiation, ground_heat, sensible):
    """Latent heat as the energy-balance residual ``LE = Rn - G - H`` [W m-2]."""
    return (np.asarray(net_radiation, float) - np.asarray(ground_heat, float)
            - np.asarray(sensible, float))


def daily_et(latent, block_s: float = BLOCK_S, lambda_v: float = LAMBDA_V) -> float:
    """Daily evapotranspiration [mm] from one day of latent-heat blocks.

    Sum, do not average: a day with missing blocks returns the ET of the blocks it has, so
    filter days by coverage before comparing them.
    """
    le = np.asarray(latent, float)
    return float(np.nansum(le) * block_s / lambda_v)
