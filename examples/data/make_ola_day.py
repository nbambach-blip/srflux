#!/usr/bin/env python
"""Extract the one-day OLA sample that ships with the example notebook.

Kept in the repo so the sample is reproducible, but it only runs where the source archive
is mounted -- it is not needed to use the notebook.

The day (2023-05-11) is a traditional, non-advective case at an almond orchard: sensible
heat upward by day (median +105 W m-2), weakly downward at night (median -34) with the
nocturnal boundary layer still COUPLED (u* 0.21 m s-1), Bowen 0.32, energy-balance closure
1.03, and all 48 blocks complete. Coupled nights matter: the calm nights that make a day
look "traditional" at other sites are exactly where eddy covariance is least reliable.

Instruments are in good order in this window, which is why it was chosen. The fine wire
gives a stable calibration across 03-12 May (alpha 0.62-0.72) and the IRT housing
thermistor sits within 0.01 K of it, so the housing is a usable air proxy here.

Source (not public):
  ec_cache/ola_001_irt.h5      1 Hz FW / T_CANOPY / T_SI111_body, keyed by block index
  ec_cache/ola_001.h5          sonic cache; its meta/timestamp maps block index -> time
  ec_pipeline_outputs_2026/ola_001/ola_001_fluxes_2021-2026.csv   30-min fluxes

NB the IRT cache is keyed by BLOCK INDEX, not by timestamp: dataset "1234" of each channel
group is the block whose start time is meta/timestamp[1234] in the sonic cache. Reading it
straight off the network volume is slow (thousands of small random reads); copy it to local
disk first if you are doing more than a day.

Writes:
  ola_2023-05-11_1hz.csv.gz    TIMESTAMP, FWT, IRT, SBT      (86,400 rows)
  ola_2023-05-11_flux30.csv    TIMESTAMP, NETRAD, G, H, LE   (48 rows)
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np
import pandas as pd

VOL = "/Volumes/JorgensenBambachLabData"
IRT_H5 = os.environ.get("OLA_IRT_H5", f"{VOL}/ec_cache/ola_001_irt.h5")
SONIC_H5 = f"{VOL}/ec_cache/ola_001.h5"
FLUX_CSV = f"{VOL}/ec_pipeline_outputs_2026/ola_001/ola_001_fluxes_2021-2026.csv"
DAY = pd.Timestamp("2023-05-11")
HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = {"FW": "FWT", "T_CANOPY": "IRT", "T_SI111_body": "SBT"}


def main() -> None:
    with h5py.File(SONIC_H5, "r", locking=False) as fm:
        ts = pd.to_datetime(fm["meta"]["timestamp"][:])
    idx = pd.Series(np.arange(len(ts)), index=ts)
    sel = idx[(idx.index >= DAY) & (idx.index < DAY + pd.Timedelta(days=1))]

    frames = []
    with h5py.File(IRT_H5, "r", locking=False) as fi:
        for t, i in sel.items():
            key = str(int(i))
            block = {}
            for src, name in CHANNELS.items():
                if key in fi[src]:
                    block[name] = np.asarray(fi[src][key][:], dtype=float)
            if not block:
                continue
            n = max(len(v) for v in block.values())
            frames.append(pd.DataFrame(
                {name: np.pad(v, (0, n - len(v)), constant_values=np.nan)
                 for name, v in block.items()},
                index=pd.date_range(t, periods=n, freq="1s")))
    hi = pd.concat(frames).round(3)
    hi.index.name = "TIMESTAMP"
    hi.to_csv(os.path.join(HERE, f"ola_{DAY.date()}_1hz.csv.gz"), compression="gzip")

    fl = pd.read_csv(FLUX_CSV, usecols=["timestamp", "NETRAD_v3_Wm2", "G_v2_Wm2",
                                        "H_Wm2", "LE_Wm2"])
    fl["TIMESTAMP"] = pd.to_datetime(fl.timestamp)
    fl = fl[(fl.TIMESTAMP >= DAY) & (fl.TIMESTAMP < DAY + pd.Timedelta(days=1))]
    fl = (fl.rename(columns={"NETRAD_v3_Wm2": "NETRAD", "G_v2_Wm2": "G",
                             "H_Wm2": "H", "LE_Wm2": "LE"})
            [["TIMESTAMP", "NETRAD", "G", "H", "LE"]]
            .sort_values("TIMESTAMP").reset_index(drop=True).round(2))
    fl.to_csv(os.path.join(HERE, f"ola_{DAY.date()}_flux30.csv"), index=False)

    print(f"1 Hz  {len(hi):,} rows  {hi.index.min()} .. {hi.index.max()}")
    print(f"30min {len(fl):,} rows  H {fl.H.min():.0f}..{fl.H.max():.0f} W/m2")


if __name__ == "__main__":
    main()
