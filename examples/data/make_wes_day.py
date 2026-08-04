#!/usr/bin/env python
"""Extract the one-day WES sample that ships with the example notebook.

Kept in the repo so the sample is reproducible, but it only runs where the source archive
is mounted -- it is not needed to use the notebook.

The day was chosen for a COMPLETE eddy-covariance record across all 48 blocks, day and
night -- only six days in 2025-2026 qualify, four of them consecutive in May 2026. Its
nights are strongly advective (H about -115 W m-2 downward with LE still positive), which
sits in the bottom 5 % of May 2026 nights: a demanding case for surface renewal, not a
typical one.

Both fine wires are exported. FW and FW2 agree on this day, but FW degrades from
2026-05-24 onward (ramp amplitude rising from ~0.4 K to 8.8 K while FW2 stays at ~0.3 K),
so the notebook uses FW2 as its air channel and the month calibration excludes the bad FW
blocks.

Source (not public):
  cache/wes_002_irt.parquet                  1 Hz IRT/fine-wire cache
  output/wes_002/wes_002_fluxes_2023-2026.csv  30-min processed eddy-covariance fluxes

Writes:
  wes_2026-05-17_1hz.csv.gz    TIMESTAMP, FWT, FWT2, IRT, SBT  (86,400 rows)
  wes_2026-05-17_flux30.csv    TIMESTAMP, NETRAD, G, H, LE   (48 rows)
"""
from __future__ import annotations

import os

import pandas as pd
import pyarrow.parquet as pq

REPO = "/Users/nbambach/Library/CloudStorage/Box-Box/ec_pipeline"
DAY = pd.Timestamp("2026-05-17")
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # ---- 1 Hz: fine-wire thermocouple (air), IRT target (canopy skin) and the IRT's own
    # housing thermistor, which equilibrates with the air and supplies the sonic-free
    # flux-direction diagnostic dT = Ts - Ta
    pf = pq.ParquetFile(os.path.join(REPO, "cache", "wes_002_irt.parquet"))
    md = pf.metadata
    rgs = [i for i in range(md.num_row_groups)
           if md.row_group(i).column(0).statistics.max >= DAY
           and md.row_group(i).column(0).statistics.min <= DAY + pd.Timedelta(days=1)]
    hi = pd.concat([pf.read_row_group(i, columns=["TIMESTAMP", "FW", "FW2", "T_CANOPY", "T_SI111_body"]).to_pandas()
                    for i in rgs], ignore_index=True)
    hi["TIMESTAMP"] = pd.to_datetime(hi.TIMESTAMP)
    hi = hi[(hi.TIMESTAMP >= DAY) & (hi.TIMESTAMP < DAY + pd.Timedelta(days=1))]
    hi = (hi.rename(columns={"FW": "FWT", "FW2": "FWT2", "T_CANOPY": "IRT",
                             "T_SI111_body": "SBT"})
            .sort_values("TIMESTAMP").reset_index(drop=True).round(3))
    hi.to_csv(os.path.join(HERE, f"wes_{DAY.date()}_1hz.csv.gz"), index=False,
              compression="gzip")

    # ---- 30-min fluxes: only the four terms of the surface energy balance
    fl = pd.read_csv(os.path.join(REPO, "output/wes_002/wes_002_fluxes_2023-2026.csv"),
                     usecols=["timestamp", "NETRAD_v3_Wm2", "G_v2_Wm2", "H_Wm2", "LE_Wm2"])
    fl["TIMESTAMP"] = pd.to_datetime(fl.timestamp)
    fl = fl[(fl.TIMESTAMP >= DAY) & (fl.TIMESTAMP < DAY + pd.Timedelta(days=1))]
    fl = (fl.rename(columns={"NETRAD_v3_Wm2": "NETRAD", "G_v2_Wm2": "G",
                             "H_Wm2": "H", "LE_Wm2": "LE"})
            [["TIMESTAMP", "NETRAD", "G", "H", "LE"]]
            .sort_values("TIMESTAMP").reset_index(drop=True).round(2))
    fl.to_csv(os.path.join(HERE, f"wes_{DAY.date()}_flux30.csv"), index=False)

    print(f"1 Hz  {len(hi):,} rows  {hi.TIMESTAMP.min()} .. {hi.TIMESTAMP.max()}")
    print(f"30min {len(fl):,} rows  H {fl.H.min():.0f}..{fl.H.max():.0f} W/m2")


if __name__ == "__main__":
    main()
