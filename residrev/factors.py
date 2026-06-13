"""Factor data: FF5+UMD loading, French-12 sector mapping, factor covariance."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import pandas_market_calendars as mcal
import requests

from residrev.config import Config

logger = logging.getLogger(__name__)

_FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
_UMD_DATASET = "F-F_Momentum_Factor_daily"
_UMD_RAW_COL = "Mom"
_EXPECTED_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]

_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_HEADERS = {"User-Agent": "sumit thakkar thakkarsumit972@gmail.com"}

# French-12 SIC ranges: (label, list-of-(lo, hi) inclusive pairs)
_FRENCH12_RANGES: list[tuple[str, list[tuple[int, int]]]] = [
    ("Consumer NonDurables", [
        (100, 999), (2000, 2399), (2700, 2749), (2770, 2799),
        (3100, 3149), (3940, 3989),
    ]),
    ("Consumer Durables", [
        (2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711),
        (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792),
        (3900, 3939), (3990, 3999),
    ]),
    ("Manufacturing", [
        (2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099),
        (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713),
        (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
        (3830, 3839), (3860, 3899),
    ]),
    ("Energy", [(1300, 1399), (2900, 2999)]),
    ("Chemicals", [(2800, 2829), (2840, 2899)]),
    ("Business Equipment", [
        (3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379),
    ]),
    ("Telecom", [(4800, 4899)]),
    ("Utilities", [(4900, 4949)]),
    ("Wholesale & Retail", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Healthcare", [(2830, 2836), (3840, 3859), (8000, 8099)]),
    ("Finance", [(6000, 6999)]),
]

_MIN_OBS = 60


def _sic_to_french12(sic: int) -> str:
    """Map a SIC code to a French-12 industry label."""
    for label, ranges in _FRENCH12_RANGES:
        for lo, hi in ranges:
            if lo <= sic <= hi:
                return label
    return "Other"


def get_ff_factors(config: Config) -> pd.DataFrame:
    """Load and cache Ken French FF5+UMD daily factors.

    Validates that a cached parquet file covers the full config date range
    before reusing it; re-fetches otherwise.

    Returns DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, UMD, RF],
    index is tz-naive dates aligned to NYSE trading calendar.
    """
    cache_path = os.path.join(config.data_dir, "factors_daily.parquet")
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if cached.index.min() <= start and cached.index.max() >= end:
            logger.info("Loading FF factors from cache: %s", cache_path)
            return cached
        logger.info("Cache does not cover full range — re-fetching FF factors")

    logger.info("Fetching FF5 factors from pandas_datareader")
    ff5_raw = pdr.get_data_famafrench(_FF5_DATASET, start=config.start_date, end=config.end_date)[0]

    logger.info("Fetching UMD (momentum) from pandas_datareader")
    umd_raw = pdr.get_data_famafrench(_UMD_DATASET, start=config.start_date, end=config.end_date)[0]
    umd = umd_raw[[_UMD_RAW_COL]].rename(columns={_UMD_RAW_COL: "UMD"})

    df = ff5_raw.join(umd, how="inner")
    df = df / 100.0

    # Align to NYSE calendar
    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.valid_days(start_date=config.start_date, end_date=config.end_date)
    sessions = sessions.tz_localize(None)  # make tz-naive

    df.index = pd.to_datetime(df.index)
    df = df.reindex(sessions).ffill(limit=1).dropna(how="all")

    # Enforce column order
    df = df[[c for c in _EXPECTED_COLS if c in df.columns]]

    os.makedirs(config.data_dir, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Saved FF factors to %s", cache_path)

    return df


def get_sector_map(tickers: list[str], config: Config) -> dict[str, str]:
    """Map tickers to French-12 industry labels via SEC EDGAR SIC codes.

    Results are cached to {config.data_dir}/sector_map.json.
    """
    cache_path = os.path.join(config.data_dir, "sector_map.json")

    if os.path.exists(cache_path):
        logger.info("Loading sector map from cache: %s", cache_path)
        with open(cache_path, "r") as f:
            return json.load(f)

    session = requests.Session()
    session.headers.update(_EDGAR_HEADERS)

    logger.info("Fetching SEC master ticker index")
    try:
        resp = session.get(_EDGAR_TICKERS_URL, timeout=30)
        resp.raise_for_status()
        master = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch SEC ticker index: %s", exc)
        return {t: "Other" for t in tickers}

    # Build uppercase ticker → zero-padded CIK
    ticker_to_cik: dict[str, str] = {}
    for entry in master.values():
        t = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if t:
            ticker_to_cik[t] = cik

    sector_map: dict[str, str] = {}
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 100 == 0:
            logger.info("Sector map progress: %d/%d tickers processed", i, len(tickers))

        cik = ticker_to_cik.get(ticker.upper())
        if cik is None:
            logger.debug("Ticker %s not found in SEC index — assigning Other", ticker)
            sector_map[ticker] = "Other"
            continue

        url = _EDGAR_SUBMISSIONS_URL.format(cik=cik)
        try:
            time.sleep(0.1)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            sic = int(data.get("sic", 0))
            sector_map[ticker] = _sic_to_french12(sic)
        except Exception as exc:
            logger.warning("Failed to fetch SIC for %s (CIK %s): %s", ticker, cik, exc)
            sector_map[ticker] = "Other"

    os.makedirs(config.data_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(sector_map, f, indent=2)
    logger.info("Saved sector map to %s", cache_path)

    return sector_map


def get_factor_cov(
    factors: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 252,
) -> Optional[np.ndarray]:
    """Trailing factor covariance matrix, strictly past-only.

    Excludes as_of date itself (no look-ahead).
    Returns None if fewer than 60 observations are available.
    Shape: (K, K) where K = number of columns in factors.
    """
    past = factors[factors.index < as_of]
    window_data = past.iloc[-window:]

    if len(window_data) < _MIN_OBS:
        logger.warning(
            "Only %d observations before %s (need %d) — returning None",
            len(window_data), as_of, _MIN_OBS,
        )
        return None

    # Exclude RF (risk-free rate) — it's not a risk factor and would make the
    # covariance matrix (K+1)×(K+1) while B is N×K, causing quad_form to fail.
    risk_cols = [c for c in window_data.columns if c != "RF"]
    return window_data[risk_cols].cov().values
