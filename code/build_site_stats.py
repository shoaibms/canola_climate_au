#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build site_rainfall_stats.csv from real SILO climatology
=========================================================
Run this after fetch_silo_data.py to rebuild stats from real data.

Usage:
    python build_site_stats.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
CLIM_FILE  = DATA_DIR / "site_rainfall_climatology.csv"
SITES_FILE = DATA_DIR / "sites_clean.csv"
OUT_FILE   = DATA_DIR / "site_rainfall_stats.csv"

# Load
clim  = pd.read_csv(CLIM_FILE)
sites = pd.read_csv(SITES_FILE, keep_default_na=False)

# Annual median per site
annual = (clim.groupby("site_code")["median_mm"]
              .sum()
              .rename("annual_median_mm")
              .reset_index())

# Apr-Oct median per site (months 4-10)
season = (clim[clim["month_num"].between(4, 10)]
              .groupby("site_code")["median_mm"]
              .sum()
              .rename("apr_oct_mm")
              .reset_index())

# Mean CV — only on months where median >= 5mm to avoid division by near-zero
# (Mediterranean sites like Dongara have summer medians near 0 which blow up CV)
cv = (clim[clim["median_mm"] >= 5]
          .groupby("site_code")
          .apply(lambda x: (x["std_mm"] / x["median_mm"]).mean(), include_groups=False)
          .rename("mean_cv")
          .reset_index())

# Merge all
stats = annual.merge(season, on="site_code").merge(cv, on="site_code")
stats = stats.merge(
    sites[["site_code","site_name","state","lat","lon","environment_group"]],
    on="site_code", how="left"
)

stats.to_csv(OUT_FILE, index=False)

print(f"Saved {len(stats)} sites to:")
print(f"  {OUT_FILE}")
print()
print(stats[["site_code","site_name","state",
             "annual_median_mm","apr_oct_mm","mean_cv"]].to_string(index=False))
