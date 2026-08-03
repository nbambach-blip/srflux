"""Ramp detectors: SR-WL (Haar front picker) and SR-VA (Van Atta structure functions)."""
from .base import RampStats
from .haar import HaarDetector, haar_kernel
from .vanatta import VanAttaDetector, chen_lag, solve_cubic

__all__ = ["RampStats", "HaarDetector", "haar_kernel", "VanAttaDetector", "chen_lag",
           "solve_cubic"]
