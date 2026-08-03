"""Synthetic ramp series, for tests and for checking a detector before trusting it on data.

The canonical surface-renewal signal is a sawtooth: a gradual rise of amplitude ``a`` over
most of the period, then a sharp collapse (the microfront). Reversing the asymmetry gives a
downward-flux ramp, which is what makes the sign conventions testable.
"""
from __future__ import annotations

import numpy as np


def ramp_series(n: int = 1800, fs: float = 1.0, period_s: float = 60.0,
                amplitude: float = 1.0, rise_fraction: float = 0.9,
                noise: float = 0.0, warm: bool = True, trend: float = 0.0,
                seed: int | None = None) -> np.ndarray:
    """Sawtooth ramp train.

    Parameters
    ----------
    n : int
        Number of samples.
    period_s : float
        Ramp repetition period [s].
    amplitude : float
        Peak-to-trough ramp amplitude, in the scalar's units.
    rise_fraction : float
        Fraction of the period spent rising. 0.9 gives the strongly asymmetric shape of a
        real warm ramp; 0.5 gives a symmetric triangle with no microfront.
    noise : float
        Standard deviation of additive white noise.
    warm : bool
        True for an upward-flux ramp (gradual rise, sharp fall).
    trend : float
        Linear drift added over the whole series, in scalar units, to exercise detrending.
    seed : int, optional
        Seed for the noise.
    """
    t = np.arange(n) / fs
    phase = np.mod(t, period_s) / period_s
    rise = np.clip(rise_fraction, 1e-3, 1 - 1e-3)
    saw = np.where(phase < rise, phase / rise, (1.0 - phase) / (1.0 - rise))
    v = amplitude * (saw - saw.mean())
    if not warm:
        v = -v
    if trend:
        v = v + np.linspace(0, trend, n)
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, n)
    return v


def step_series(n: int = 1800, fs: float = 1.0, step_s: float = 100.0,
                amplitude: float = 1.0) -> np.ndarray:
    """Square wave of known step height -- the cleanest possible test of an edge detector."""
    t = np.arange(n) / fs
    return amplitude * np.where(np.mod(t, 2 * step_s) < step_s, 0.5, -0.5)
