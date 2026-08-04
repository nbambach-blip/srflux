#!/usr/bin/env python
"""Fit the surface-renewal calibration on a MONTH of data, for the notebook to apply to a
single day.

Why this exists. Fitting alpha on the same day you then evaluate makes the reported skill a
goodness-of-fit, not a prediction, and one day is far too short to pin alpha down: a
held-out test across several orchards and vineyards needed 10-21 days spread over a record
before the daily-ET error settled. So the calibration is fitted here on a whole month and
the demo day is EXCLUDED from the fit, which makes the notebook's numbers genuinely
out-of-sample.

Only the resulting COEFFICIENTS are published (a few dozen numbers in a JSON file). The
month of raw data stays where it is; the repository ships one day.

Regimes. Alpha is not a single number across the diurnal cycle -- the ramp-to-flux
relationship differs between convective daytime and stable nighttime, and a coefficient
fitted on daytime convection over-estimates the weak nocturnal flux. Two regimes are fitted
separately, each with a known flux direction, so the sign never enters the calibration:

    unstable_day    Rn > 50 W m-2 and H > 0     flux upward,   sign +1
    stable_night    Rn < -20 W m-2 and H < 0    flux downward, sign -1

Within a regime the fit is through-origin on the UNSIGNED ramp flux against |H|, so alpha is
positive and the direction is applied separately at prediction time.

Source (not public):
  cache/wes_002_irt.parquet                     1 Hz IRT / fine-wire cache
  output/wes_002/wes_002_fluxes_2023-2026.csv   30-min processed eddy-covariance fluxes

Writes: wes_2023-07_calibration.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/nbambach/srflux/src")
from srflux import (HaarDetector, VanAttaDetector, calibrate_alpha, prepare_block,  # noqa: E402
                    ramp_flux)

REPO = "/Users/nbambach/Library/CloudStorage/Box-Box/ec_pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
MONTH_START, MONTH_END = pd.Timestamp("2023-07-01"), pd.Timestamp("2023-08-01")
EXCLUDE_DAY = pd.Timestamp("2023-07-06")          # the day the notebook publishes
FS, BLOCK_S = 1.0, 1800.0
Z = {"FWT": 7.5, "FWT2": 7.5, "IRT": 3.5}         # measurement height / canopy height
A_MAX_K = {"FWT": 3.0, "FWT2": 3.0, "IRT": 3.0}   # reject physically impossible ramp amplitudes
VA_LAG_S = 4.0                                    # fixed lag for the unit-period variant


def month_blocks() -> pd.DataFrame:
    """Per-block ramp statistics and reference fluxes for the whole month."""
    pf = pq.ParquetFile(os.path.join(REPO, "cache", "wes_002_irt.parquet"))
    md = pf.metadata
    rgs = [i for i in range(md.num_row_groups)
           if md.row_group(i).column(0).statistics.max >= MONTH_START
           and md.row_group(i).column(0).statistics.min <= MONTH_END]
    hi = pd.concat([pf.read_row_group(i, columns=["TIMESTAMP", "FW", "FW2", "T_CANOPY"]).to_pandas()
                    for i in rgs], ignore_index=True)
    hi["TIMESTAMP"] = pd.to_datetime(hi.TIMESTAMP)
    hi = (hi[(hi.TIMESTAMP >= MONTH_START) & (hi.TIMESTAMP < MONTH_END)]
            .rename(columns={"FW": "FWT", "FW2": "FWT2", "T_CANOPY": "IRT"})
            .set_index("TIMESTAMP"))

    wl = HaarDetector(fs=FS)
    va = VanAttaDetector(fs=FS, lag=VA_LAG_S, period_mode="unit")
    rows = []
    for t, g in hi.groupby(pd.Grouper(freq="30min")):
        row = {"ts": t}
        for col in ("FWT", "FWT2", "IRT"):
            qc = prepare_block(g[col].to_numpy(), fs=FS)
            if not qc.ok:
                continue
            rw, rv = wl.detect(qc.values), va.detect(qc.values)
            # A ramp amplitude of several kelvin is not physical at 30-min scale; it is a
            # spiking sensor. WES fine wire FW does exactly this from 2026-05-24 onward.
            if rw.valid and rw.amplitude > A_MAX_K[col]:
                continue
            if rw.valid:
                row[f"WL_{col}"] = ramp_flux(rw.amplitude, count=rw.count, z=Z[col],
                                             block_s=BLOCK_S)
            if rv.valid:
                row[f"VA_{col}"] = ramp_flux(rv.amplitude, period=1.0, z=Z[col])
        rows.append(row)
    sr = pd.DataFrame(rows).set_index("ts")

    fx = pd.read_csv(os.path.join(REPO, "output/wes_002/wes_002_fluxes_2023-2026.csv"),
                     usecols=["timestamp", "H_Wm2", "NETRAD_v3_Wm2"])
    fx["ts"] = pd.to_datetime(fx.timestamp)
    fx = fx.set_index("ts")[["H_Wm2", "NETRAD_v3_Wm2"]]
    return sr.join(fx).dropna(subset=["H_Wm2", "NETRAD_v3_Wm2"])


def main() -> None:
    d = month_blocks()
    d = d[d.index.floor("D") != EXCLUDE_DAY]                 # keep the demo day out
    regimes = {
        "unstable_day": (d.NETRAD_v3_Wm2 > 50) & (d.H_Wm2 > 0),
        "stable_night": (d.NETRAD_v3_Wm2 < -20) & (d.H_Wm2 < 0),
    }
    out = {
        "site": "WES_002 (almond orchard)",
        "fit_period": f"{MONTH_START.date()} to {(MONTH_END - pd.Timedelta(days=1)).date()}",
        "excluded_day": str(EXCLUDE_DAY.date()),
        "block_minutes": 30,
        "heights_m": Z,
        "va_lag_s": VA_LAG_S,
        "va_period_mode": "unit",
        "regimes": {"unstable_day": "NETRAD > 50 W m-2 and H > 0, sign +1",
                    "stable_night": "NETRAD < -20 W m-2 and H < 0, sign -1"},
        "note": ("alpha fitted through the origin on unsigned ramp flux against |H|; "
                 "apply the regime sign separately"),
        "qc": (f"blocks with Haar amplitude above {A_MAX_K} K rejected as sensor spikes; "
               "both fine wires are healthy in this month, unlike 2025-2026 where FWT fails"),
        "coefficients": {},
    }
    print(f"month blocks after excluding {EXCLUDE_DAY.date()}: {len(d)}")
    for rname, mask in regimes.items():
        sub = d[mask]
        out["coefficients"][rname] = {}
        for meth in ("WL", "VA"):
            for col in ("FWT", "FWT2", "IRT"):
                key = f"{meth}_{col}"
                s = sub.dropna(subset=[key])
                if len(s) < 30:
                    continue
                F, H = s[key].to_numpy(), s.H_Wm2.abs().to_numpy()
                a = calibrate_alpha(F, H)
                r = float(np.corrcoef(F, H)[0, 1])
                rmse = float(np.sqrt(np.mean((a * F - H) ** 2)))
                out["coefficients"][rname][key] = {
                    "alpha": round(a, 4), "n": int(len(s)), "r": round(r, 3),
                    "rmse_Wm2": round(rmse, 1),
                    "n_days": int(s.index.floor("D").nunique())}
                print(f"  {rname:13s} {key:7s} alpha {a:7.3f}  r {r:+.2f}  "
                      f"RMSE {rmse:5.1f}  n {len(s):4d} ({out['coefficients'][rname][key]['n_days']} days)")

    p = os.path.join(HERE, "wes_2023-07_calibration.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
