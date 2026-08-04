#!/usr/bin/env python
"""Fit the OLA surface-renewal calibration, for the notebook to apply to a single day.

Publishes only the COEFFICIENTS: alpha per regime, the detector configuration they were
fitted with, and the flux-direction convention. The archive stays where it is.

Window. Alpha needs 10-21 days spread over a range of conditions, so a fortnight is about
the minimum. It is also the maximum here: the fine wire is stable through mid-May 2023 and
drifts afterwards, and a calibration window should be bounded by sensor behaviour rather
than by the calendar. The demo day is excluded, so the notebook is out-of-sample.

Regimes. Alpha is fitted separately for convective daytime and stable nighttime, each with a
known flux direction, so the sign never enters the calibration:

    unstable_day    Rn > 50 W m-2 and H > 0     flux upward,   sign +1
    stable_night    Rn < -20 W m-2 and H < 0    flux downward, sign -1

Within a regime the fit is through-origin on the UNSIGNED ramp flux against |H|.

Screening. A through-origin fit weights by F^2, so a day on which the sensor is mis-scaled
carries enormous leverage and can drag the pooled coefficient towards zero. Days whose own
alpha is a scale outlier are dropped per channel and regime, and recorded in the output.
Screen before tuning anything else: an unscreened window can make a workable detector
configuration look impossible.

Hyperparameters. The Van Atta lag and the flux-direction convention are both selected on
this window, never on the demo day, and scored by NSE rather than agreement rate -- a
convention can be right most of the time and still have negative skill if its errors land on
the high-|H| blocks.

Writes: ola_2023-05_calibration.json
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/nbambach/srflux/src")
from srflux import (HaarDetector, VanAttaDetector, calibrate_alpha, prepare_block,  # noqa: E402
                    ramp_flux, scores)
from srflux.preprocess import increment_skewness  # noqa: E402

VOL = "/Volumes/JorgensenBambachLabData"
IRT_H5 = os.environ.get("OLA_IRT_H5", f"{VOL}/ec_cache/ola_001_irt.h5")
SONIC_H5 = f"{VOL}/ec_cache/ola_001.h5"
FLUX_CSV = f"{VOL}/ec_pipeline_outputs_2026/ola_001/ola_001_fluxes_2021-2026.csv"
HERE = os.path.dirname(os.path.abspath(__file__))

WIN_START, WIN_END = pd.Timestamp("2023-05-01"), pd.Timestamp("2023-05-15")
EXCLUDE_DAY = pd.Timestamp("2023-05-11")
FS, BLOCK_S = 1.0, 1800.0
Z = {"FWT": 10.5, "IRT": 5.5}
A_MAX_K = 3.0
VA_LAG_S, VA_MODE = "chen", "unit"
SIGN_LAGS_S = (1, 2, 3, 4, 6, 8, 10, 15, 20, 30)
SRC = {"FW": "FWT", "T_CANOPY": "IRT"}


def window_blocks() -> pd.DataFrame:
    with h5py.File(SONIC_H5, "r", locking=False) as fm:
        ts = pd.to_datetime(fm["meta"]["timestamp"][:])
    idx = pd.Series(np.arange(len(ts)), index=ts)
    sel = idx[(idx.index >= WIN_START) & (idx.index < WIN_END)]

    wl = HaarDetector(fs=FS)
    va = VanAttaDetector(fs=FS, lag=VA_LAG_S, period_mode=VA_MODE)
    rows = []
    with h5py.File(IRT_H5, "r", locking=False) as fi:
        for t, i in sel.items():
            key, row = str(int(i)), {"ts": t}
            for src, name in SRC.items():
                if key not in fi[src]:
                    continue
                qc = prepare_block(np.asarray(fi[src][key][:], float), fs=FS)
                if not qc.ok:
                    continue
                for lag in SIGN_LAGS_S:
                    row[f"sk{lag}_{name}"] = increment_skewness(
                        qc.values, fs=FS, lag=max(1, int(round(lag * FS))))
                rw, rv = wl.detect(qc.values), va.detect(qc.values)
                if rw.valid and rw.amplitude <= A_MAX_K:
                    row[f"WL_{name}"] = ramp_flux(rw.amplitude, count=rw.count, z=Z[name],
                                                  block_s=BLOCK_S)
                if rv.valid:
                    row[f"VA_{name}"] = ramp_flux(rv.amplitude, period=1.0, z=Z[name])
            rows.append(row)
    sr = pd.DataFrame(rows).set_index("ts")

    fx = pd.read_csv(FLUX_CSV, usecols=["timestamp", "H_Wm2", "NETRAD_v3_Wm2"])
    fx["ts"] = pd.to_datetime(fx.timestamp)
    return sr.join(fx.set_index("ts")).dropna(subset=["H_Wm2", "NETRAD_v3_Wm2"])


def daily_alpha_screen(s: pd.DataFrame, key: str, k: float = 3.0, min_blocks: int = 8):
    """Drop days whose OWN alpha is a scale outlier, and report which.

    Keeps days whose per-day alpha sits within `k` robust MADs of the median on a log scale:
    mild and scale-based, so it removes days the instrument got wrong rather than days the
    weather made noisy. It uses the reference flux, available by definition on a calibration
    set, but must never be applied to the day being predicted.
    """
    per = {}
    for day, g in s.groupby(s.index.floor("D")):
        if len(g) < min_blocks:
            continue
        a = calibrate_alpha(g[key], g.H_Wm2.abs())
        if np.isfinite(a) and a > 0:
            per[day] = a
    if len(per) < 4:
        return list(per), []
    la = np.log(np.fromiter(per.values(), float))
    med = float(np.median(la))
    mad = float(1.4826 * np.median(np.abs(la - med)))
    keep = [d for d, a in per.items() if abs(np.log(a) - med) <= k * max(mad, 1e-6)]
    dropped = [str(d.date()) for d in sorted(set(per) - set(keep))]
    return keep, dropped


def main() -> None:
    d = window_blocks()
    d = d[d.index.floor("D") != EXCLUDE_DAY]
    regimes = {"unstable_day": (d.NETRAD_v3_Wm2 > 50) & (d.H_Wm2 > 0),
               "stable_night": (d.NETRAD_v3_Wm2 < -20) & (d.H_Wm2 < 0)}
    out = {
        "site": "OLA_001 (almond orchard)",
        "fit_period": f"{WIN_START.date()} to {(WIN_END - pd.Timedelta(days=1)).date()}",
        "excluded_day": str(EXCLUDE_DAY.date()),
        "block_minutes": 30,
        "heights_m": Z,
        "va_lag_s": VA_LAG_S,
        "va_period_mode": VA_MODE,
        "regimes": {"unstable_day": "NETRAD > 50 W m-2 and H > 0, sign +1",
                    "stable_night": "NETRAD < -20 W m-2 and H < 0, sign -1"},
        "note": ("alpha fitted through the origin on unsigned ramp flux against |H|; "
                 "apply the regime sign separately"),
        "qc": (f"ramp amplitudes above {A_MAX_K} K rejected; days whose own alpha is a scale "
               "outlier (>3 robust MADs from the median, log scale) dropped per channel and "
               "regime -- see days_dropped"),
        "coefficients": {},
    }
    print(f"window blocks after excluding {EXCLUDE_DAY.date()}: {len(d)}")
    for rname, mask in regimes.items():
        sub = d[mask]
        out["coefficients"][rname] = {}
        for meth in ("WL", "VA"):
            for name in ("FWT", "IRT"):
                key = f"{meth}_{name}"
                s = sub.dropna(subset=[key])
                if len(s) < 25:
                    continue
                keep, dropped = daily_alpha_screen(s, key)
                s = s[s.index.floor("D").isin(keep)] if keep else s
                if len(s) < 25:
                    continue
                F, H = s[key].to_numpy(), s.H_Wm2.abs().to_numpy()
                a = calibrate_alpha(F, H)
                r = float(np.corrcoef(F, H)[0, 1])
                rmse = float(np.sqrt(np.mean((a * F - H) ** 2)))
                out["coefficients"][rname][key] = {
                    "alpha": round(a, 5), "n": int(len(s)), "r": round(r, 3),
                    "rmse_Wm2": round(rmse, 1),
                    "n_days": int(s.index.floor("D").nunique()),
                    "days_dropped": dropped}
                print(f"  {rname:13s} {key:7s} alpha {a:8.4f}  r {r:+.2f}  "
                      f"RMSE {rmse:5.1f}  n {len(s):4d}"
                      + (f"  dropped {dropped}" if dropped else ""))

    out["sign"] = {}
    d = d.assign(regime=np.where(d.NETRAD_v3_Wm2 > 50, "unstable_day",
                        np.where(d.NETRAD_v3_Wm2 < -20, "stable_night",
                        np.where(d.NETRAD_v3_Wm2 > 0, "unstable_day", "stable_night"))))
    for name in ("FWT", "IRT"):
        akey = f"WL_{name}"
        alpha = np.array([out["coefficients"][r].get(akey, {}).get("alpha", np.nan)
                          for r in d.regime])
        best = None
        for lag in SIGN_LAGS_S:
            sk = d.get(f"sk{lag}_{name}")
            if sk is None:
                continue
            fin = np.isfinite(sk) & np.isfinite(alpha) & np.isfinite(d[akey])
            if fin.sum() < 100:
                continue
            lo, hi = np.nanpercentile(sk[fin], [2, 98])
            for tau in np.linspace(lo, hi, 81):
                sgn = np.where(sk < tau, 1.0, -1.0)
                est = (sgn * alpha * d[akey])[fin]
                sc = scores(est, d.H_Wm2[fin], 1.0)
                acc = float(np.mean(sgn[fin] == np.sign(d.H_Wm2[fin])))
                if best is None or sc["nse"] > best["window_nse"]:
                    best = {"source": "skewness", "channel": name, "lag_s": float(lag),
                            "tau": round(float(tau), 4),
                            "window_nse": round(sc["nse"], 3),
                            "window_agreement": round(acc, 3)}
        if best:
            out["sign"][name] = best
            print(f"  sign {name}: skewness lag {best['lag_s']:.0f} s tau {best['tau']:+.3f} "
                  f"-> window NSE {best['window_nse']:+.2f}, agree {100*best['window_agreement']:.0f}%")

    p = os.path.join(HERE, "ola_2023-05_calibration.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
