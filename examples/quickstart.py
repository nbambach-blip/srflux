#!/usr/bin/env python
"""End-to-end example: ramps -> flux -> calibration -> daily ET, on synthetic data.

Runs the whole chain both ways (SR-WL and SR-VA) against a synthetic "reference" flux, so
it can be executed without any field data:

    python examples/quickstart.py

Replace `synthetic_day()` with your own 1 Hz blocks and a measured H to use it for real.
"""
from __future__ import annotations

import numpy as np

from srflux import (HaarDetector, VanAttaDetector, calibrate_alpha, daily_et,
                    fit_and_score, latent_heat_residual, prepare_block, ramp_flux,
                    sensible_heat, sign_from_stability)
from srflux.synthetic import ramp_series

FS = 1.0            # Hz
BLOCK_S = 1800.0    # 30-min blocks
CANOPY_H = 2.0      # m -- length scale for a SURFACE (skin) temperature
TRUE_ALPHA = 3.0    # what we will try to recover


def synthetic_day(n_blocks: int = 24, seed: int = 0):
    """A day of blocks whose ramp amplitude tracks a prescribed sensible heat flux."""
    rng = np.random.default_rng(seed)
    hours = np.linspace(6, 18, n_blocks)
    H_true = 250 * np.sin(np.pi * (hours - 6) / 12) ** 2 + rng.normal(0, 12, n_blocks)
    blocks, refs = [], []
    for H in H_true:
        # bigger flux -> bigger ramps, plus block-to-block scatter the detector must survive
        amp = 0.05 + 0.004 * max(H, 0) * rng.uniform(0.85, 1.15)
        period = 60.0
        v = 20.0 + ramp_series(int(BLOCK_S * FS), FS, period, amp, noise=0.02,
                               seed=int(rng.integers(1e6)))
        blocks.append(v)
        refs.append(H)
    return blocks, np.asarray(refs)


def main() -> None:
    blocks, H_ref = synthetic_day()
    detectors = {"SR-WL (Haar)": HaarDetector(fs=FS),
                 "SR-VA (Van Atta)": VanAttaDetector(fs=FS, lag="chen")}

    fluxes = {}
    for label, det in detectors.items():
        F = []
        for raw in blocks:
            qc = prepare_block(raw, fs=FS)
            if not qc.ok:
                F.append(np.nan)
                continue
            res = det.detect(qc.values)
            if not res.valid:
                F.append(np.nan)
                continue
            # count form for the front picker, period form for the structure-function fit
            F.append(ramp_flux(res.amplitude, count=res.count, z=CANOPY_H, block_s=BLOCK_S)
                     if det.name == "haar"
                     else ramp_flux(res.amplitude, period=res.period, z=CANOPY_H))
        F = np.asarray(F)
        fluxes[det.name] = F

        fit = fit_and_score(F, H_ref)
        print(f"{label:>18}:  alpha = {fit.alpha:6.2f}   r = {fit.r:4.2f}   "
              f"RMSE = {fit.rmse:5.1f} W/m2   n = {fit.n}")

        # energy balance -> ET, with the direction taken from stability (all daytime here)
        zeta = np.full_like(H_ref, -0.5)
        H_sr = sensible_heat(F, fit.alpha, sign=sign_from_stability(zeta))
        Rn, G = np.full_like(H_ref, 600.0), np.full_like(H_ref, 70.0)
        et = daily_et(latent_heat_residual(Rn, G, H_sr), block_s=BLOCK_S)
        et_ref = daily_et(latent_heat_residual(Rn, G, H_ref), block_s=BLOCK_S)
        print(f"{'':>18}   daily ET = {et:4.2f} mm  (reference {et_ref:4.2f} mm, "
              f"error {et - et_ref:+.2f})")

    # What a borrowed calibration costs: apply an alpha that is 50 % too large.
    F_wl = fluxes["haar"]
    a_local = calibrate_alpha(F_wl, H_ref)
    Rn, G = np.full_like(H_ref, 600.0), np.full_like(H_ref, 70.0)
    et_local = daily_et(latent_heat_residual(Rn, G, sensible_heat(F_wl, a_local)))
    et_borrow = daily_et(latent_heat_residual(Rn, G, sensible_heat(F_wl, 1.5 * a_local)))
    print(f"\na 50 % alpha error moves daily ET by {et_borrow - et_local:+.2f} mm "
          f"({100 * (et_borrow - et_local) / et_local:+.0f} %)")


if __name__ == "__main__":
    main()
