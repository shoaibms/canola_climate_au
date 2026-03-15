#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SILO Climate Data Fetcher for Canola Representative Sites
==========================================================
Fetches daily rainfall + max/min temperature from SILO DataDrill API.
No registration required - just a valid email address.

Variables fetched:
    R  = daily rainfall
    X  = max temperature
    N  = min temperature

Usage:
    python fetch_silo_data.py
"""

import requests
import pandas as pd
import time
from io import StringIO
from pathlib import Path

# -- Your email ---------------------------------------------------------------
# Replace with your own email address (required by SILO DataDrill API)
SILO_EMAIL = "your.email@example.com"

# -- Paths (relative to this script) -----------------------------------------
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE  = DATA_DIR / "sites_clean.csv"
OUTPUT_FILE = DATA_DIR / "site_rainfall_climatology.csv"

# -- Load sites ---------------------------------------------------------------
sites = pd.read_csv(INPUT_FILE, keep_default_na=False)
sites = sites[sites["lat"].notna() & (sites["lat"] != "")]
sites["lat"] = pd.to_numeric(sites["lat"])
sites["lon"] = pd.to_numeric(sites["lon"])
sites = sites[sites["lat"].notna()]
print(f"Loaded {len(sites)} sites\n")

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

all_rows = []

for _, s in sites.iterrows():
    # comment=RXN fetches rainfall + Tmax + Tmin in one call
    url = (
        f"https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
        f"?lat={s['lat']}&lon={s['lon']}"
        f"&start=19900101&finish=20231231"
        f"&format=csv&comment=RXN"
        f"&username={SILO_EMAIL}"
    )

    print(f"  Fetching {s['site_code']} ({s['site_name']}, {s['state']})...")

    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            print(f"    FAILED  HTTP {r.status_code}")
            continue

        df = pd.read_csv(StringIO(r.text), comment="#")
        df.columns = df.columns.str.strip()

        # Identify columns
        # Print columns on first site to help debug
        if not all_rows:
            print(f"    Columns: {df.columns.tolist()}")

        date_col = next((c for c in df.columns
                         if c.lower() in ("date","yyyy-mm-dd")
                         or "date" in c.lower()), None)
        rain_col = next((c for c in df.columns
                         if c.lower() in ("daily_rain","rain","rainfall","r")), None)
        tmax_col = next((c for c in df.columns
                         if c.lower() in ("max_temp","tmax","maximum_temperature","x")), None)
        tmin_col = next((c for c in df.columns
                         if c.lower() in ("min_temp","tmin","minimum_temperature","n")), None)

        if not date_col or not rain_col:
            print(f"    ! Cannot identify columns: {df.columns.tolist()}")
            continue

        df["year"]  = pd.to_datetime(df[date_col]).dt.year
        df["month"] = pd.to_datetime(df[date_col]).dt.month

        for col in [rain_col, tmax_col, tmin_col]:
            if col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Monthly aggregation
        monthly_rain = (df.groupby(["year","month"])[rain_col]
                          .sum().reset_index()
                          .rename(columns={rain_col: "monthly_rain"}))

        clim = monthly_rain.groupby("month")["monthly_rain"].agg(
            median_mm="median",
            std_mm="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75)
        ).reset_index()

        # Temperature: monthly median of daily values
        if tmax_col:
            tmax_monthly = (df.groupby("month")[tmax_col]
                              .median().reset_index()
                              .rename(columns={tmax_col: "tmax_median"}))
            clim = clim.merge(tmax_monthly, on="month", how="left")
        else:
            clim["tmax_median"] = None

        if tmin_col:
            tmin_monthly = (df.groupby("month")[tmin_col]
                              .median().reset_index()
                              .rename(columns={tmin_col: "tmin_median"}))
            clim = clim.merge(tmin_monthly, on="month", how="left")
        else:
            clim["tmin_median"] = None

        clim["month_num"] = clim["month"]
        clim["month"]     = clim["month_num"].apply(lambda m: MONTH_NAMES[m-1])
        clim["site_code"] = s["site_code"]
        clim["site_name"] = s["site_name"]
        clim["state"]     = s["state"]
        clim["lat"]       = s["lat"]
        clim["lon"]       = s["lon"]

        all_rows.append(clim)
        tmax_ok = "tmax OK" if tmax_col else "tmax missing"
        tmin_ok = "tmin OK" if tmin_col else "tmin missing"
        print(f"    OK  Rain: {clim['median_mm'].sum():.0f} mm  |  {tmax_ok}  |  {tmin_ok}")

    except Exception as e:
        print(f"    ERROR: {e}")

    time.sleep(1.5)

print()
if all_rows:
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result)} rows to:\n  {OUTPUT_FILE}")
    print("\nColumns:", result.columns.tolist())
    print("\nSummary (annual rainfall):")
    summary = result.groupby("site_code")["median_mm"].sum().reset_index()
    summary.columns = ["site_code", "annual_median_mm"]
    print(summary.to_string(index=False))
else:
    print("No data fetched. Check internet connection and email address.")
