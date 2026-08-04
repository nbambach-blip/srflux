"""srflux -- surface-renewal sensible heat flux from a high-rate scalar series.

Two ramp detectors, one calibration, and the energy-balance step to evapotranspiration:

    SR-WL  :class:`~srflux.detectors.haar.HaarDetector`
           Haar wavelet front picker with a sigma-relative threshold. Counts microfronts
           and measures their amplitude directly.

    SR-VA  :class:`~srflux.detectors.vanatta.VanAttaDetector`
           Van Atta structure-function cubic with the Chen adaptive lag. Fits an idealised
           ramp instead of counting.

Quick start
-----------
>>> import numpy as np
>>> from srflux import HaarDetector, VanAttaDetector, prepare_block, ramp_flux
>>> from srflux.synthetic import ramp_series
>>> block = prepare_block(ramp_series(n=1800, period_s=60, amplitude=1.0, seed=1))
>>> wl = HaarDetector(fs=1.0).detect(block.values)
>>> va = VanAttaDetector(fs=1.0).detect(block.values)
>>> F_wl = ramp_flux(wl.amplitude, count=wl.count, z=2.0)      # W m-2, uncalibrated
>>> F_va = ramp_flux(va.amplitude, period=va.period, z=2.0)
>>> bool(F_wl > 0 and F_va > 0)
True

Calibrate ``alpha`` against a reference flux with :func:`~srflux.flux.calibrate_alpha`,
then ``H = alpha * F``. See the README for what alpha does and does not transfer across.
"""
from __future__ import annotations

from .detectors.base import RampStats
from .detectors.haar import HaarDetector, haar_kernel
from .detectors.vanatta import VanAttaDetector, chen_lag, solve_cubic
from .flux import (AlphaFit, calibrate_alpha, daily_et, fit_and_score,
                   latent_heat_residual, ramp_flux, scores, sensible_heat)
from .preprocess import BlockQC, detrend, increment_skewness, moving_average, prepare_block
from .sign import (fit_gradient_offset, sign_from_gradient, sign_from_skewness,
                   sign_from_stability, stability_from_gradient, surface_air_difference)

__version__ = "0.1.0"

__all__ = [
    "HaarDetector", "VanAttaDetector", "RampStats", "haar_kernel", "chen_lag",
    "solve_cubic", "prepare_block", "BlockQC", "moving_average", "detrend",
    "increment_skewness", "ramp_flux", "calibrate_alpha", "sensible_heat", "fit_and_score",
    "scores",
    "AlphaFit", "latent_heat_residual", "daily_et", "sign_from_skewness",
    "sign_from_stability", "sign_from_gradient", "fit_gradient_offset",
    "stability_from_gradient", "surface_air_difference", "__version__",
]
