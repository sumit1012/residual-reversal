"""Resilient daily OHLCV price pipeline: yfinance primary, Stooq fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from residrev.config import Config

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 75
INTER_CHUNK_SLEEP: float = 1.5
MAX_RETRIES: int = 4

_EXPECTED_COLS = ["Open", "High", "Low", "Close", "Volume"]
_LARGE_MOVE_THRESHOLD: float = 0.50


def load_cached(ticker: str, cache_dir: str) -> pd.DataFrame | None:
    """Read {cache_dir}/{ticker}.parquet; return None if missing or unreadable."""
    path = Path(cache_dir) / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.warning("Failed to read cache for %s", ticker)
        return None


def save_cached(ticker: str, df: pd.DataFrame, cache_dir: str) -> None:
    """Write {cache_dir}/{ticker}.parquet; create directory if needed."""
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{ticker}.parquet")


def _extract_ticker(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extract and validate a single-ticker DataFrame from yfinance output.

    Handles both MultiIndex (multi-ticker download) and flat columns (single ticker).
    """
    if raw_df is None or raw_df.empty:
        return None

    if isinstance(raw_df.columns, pd.MultiIndex):
        lvl1 = raw_df.columns.get_level_values(1)
        lvl0 = raw_df.columns.get_level_values(0)
        if ticker in lvl1:
            df = raw_df.xs(ticker, level=1, axis=1).copy()
        elif ticker in lvl0:
            df = raw_df[ticker].copy()
        else:
            return None
    else:
        df = raw_df.copy()

    if df.empty:
        return None

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.warning("Ticker %s missing expected columns: %s", ticker, missing)
        return None

    df = df[_EXPECTED_COLS]

    large = df["Close"].pct_change().abs() > _LARGE_MOVE_THRESHOLD
    if large.any():
        logger.warning(
            "Ticker %s: large moves (>50%%) on %s", ticker, df.index[large].tolist()
        )

    return df


def _clean_stooq(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalize column casing and apply standard cleaning to Stooq data."""
    if df.empty:
        return None
    df = df.copy()
    df.columns = [c.title() for c in df.columns]
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.warning("Stooq %s missing columns: %s", ticker, missing)
        return None
    return df[_EXPECTED_COLS]


def pull_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch from Stooq: pandas_datareader first, direct CSV fallback."""
    stooq_sym = ticker.lower().replace(".", "").replace("-", "") + ".us"

    try:
        import pandas_datareader as pdr

        df = pdr.DataReader(stooq_sym, "stooq", start=start, end=end)
        df = df.sort_index()
        result = _clean_stooq(df, ticker)
        if result is not None and not result.empty:
            return result
    except Exception as exc:
        logger.warning("pandas_datareader Stooq failed for %s: %s", ticker, exc)

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    try:
        df = pd.read_csv(url, index_col="Date", parse_dates=True).sort_index()
        result = _clean_stooq(df, ticker)
        if result is not None and not result.empty:
            return result
    except Exception as exc:
        logger.warning("Stooq CSV failed for %s: %s", ticker, exc)

    return None


def _is_fresh(df: pd.DataFrame, start: str, end: str) -> bool:
    """Return True if df's date range fully covers [start, end]."""
    if df is None or df.empty:
        return False
    return (
        df.index.min() <= pd.Timestamp(start)
        and df.index.max() >= pd.Timestamp(end)
    )


def _fetch_with_retry(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame | None:
    """Call yf.download with exponential backoff; return None after MAX_RETRIES."""
    for attempt in range(MAX_RETRIES):
        try:
            return yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                actions=True,
                threads=False,
                progress=False,
            )
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                "yfinance attempt %d/%d failed (%s). Retrying in %ds.",
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    return None


def pull_prices(
    tickers: list[str],
    start: str,
    end: str,
    config: Config,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for tickers. Returns {ticker: DataFrame}."""
    out: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        cached = load_cached(ticker, config.cache_dir)
        if cached is not None and _is_fresh(cached, start, end):
            out[ticker] = cached
        else:
            to_fetch.append(ticker)

    for i in range(0, len(to_fetch), CHUNK_SIZE):
        chunk = to_fetch[i : i + CHUNK_SIZE]
        raw = _fetch_with_retry(chunk, start, end)

        for ticker in chunk:
            df = _extract_ticker(raw, ticker) if raw is not None else None
            if df is not None and not df.empty:
                save_cached(ticker, df, config.cache_dir)
                out[ticker] = df
            else:
                logger.warning("yfinance empty for %s; trying Stooq fallback", ticker)
                fallback = pull_stooq(ticker, start, end)
                if fallback is not None:
                    save_cached(ticker, fallback, config.cache_dir)
                    out[ticker] = fallback
                else:
                    logger.warning("All sources failed for %s", ticker)

        if i + CHUNK_SIZE < len(to_fetch):
            time.sleep(INTER_CHUNK_SLEEP)

    return out
