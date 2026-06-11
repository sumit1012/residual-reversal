"""10 tests for residrev.data price pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from residrev.config import Config
from residrev.data import _extract_ticker, load_cached, pull_prices, save_cached


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATES = ["2020-01-02", "2020-01-03", "2020-01-06"]


def _ohlcv(dates, tz=None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, tz=tz)
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 102.0,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def _multi_raw(tickers: list[str], dates) -> pd.DataFrame:
    """Build a MultiIndex DataFrame mimicking yfinance multi-ticker output.

    Columns are (field, ticker) matching yfinance's default group_by='column'.
    """
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = {
        (f, t): [100.0 if f != "Volume" else 1_000_000.0] * len(dates)
        for f in fields
        for t in tickers
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    df.columns = pd.MultiIndex.from_tuples(df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# _extract_ticker
# ---------------------------------------------------------------------------


def test_extract_ticker_multi_returns_single_ticker_df():
    """_extract_ticker pulls correct single-ticker data from a multi-ticker raw."""
    raw = _multi_raw(["AAPL", "MSFT"], DATES)
    df = _extract_ticker(raw, "AAPL")
    assert df is not None
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == len(DATES)


def test_extract_ticker_empty_returns_none():
    """_extract_ticker returns None for an empty DataFrame."""
    assert _extract_ticker(pd.DataFrame(), "AAPL") is None


def test_extract_ticker_removes_duplicate_index():
    """_extract_ticker drops duplicate index rows, keeping the last occurrence."""
    dates = ["2020-01-02", "2020-01-02", "2020-01-03"]
    df_dup = _ohlcv(dates)
    # Make the first duplicate distinct so we can verify keep='last'
    df_dup.iloc[0, df_dup.columns.get_loc("Close")] = 50.0
    result = _extract_ticker(df_dup, "X")
    assert result is not None
    assert not result.index.duplicated().any()
    assert len(result) == 2
    # keep='last' → the second 2020-01-02 row (Close=102.0) survives
    assert result.iloc[0]["Close"] == 102.0


def test_extract_ticker_strips_timezone():
    """_extract_ticker converts a tz-aware DatetimeIndex to tz-naive."""
    df_tz = _ohlcv(DATES, tz="UTC")
    result = _extract_ticker(df_tz, "X")
    assert result is not None
    assert result.index.tz is None


# ---------------------------------------------------------------------------
# load_cached / save_cached
# ---------------------------------------------------------------------------


def test_load_cached_returns_none_when_file_missing(tmp_path):
    """load_cached returns None when the parquet file does not exist."""
    assert load_cached("AAPL", str(tmp_path)) is None


def test_load_cached_returns_df_when_file_exists(tmp_path):
    """load_cached returns a DataFrame after a parquet file is written."""
    df = _ohlcv(DATES)
    df.to_parquet(tmp_path / "AAPL.parquet")
    result = load_cached("AAPL", str(tmp_path))
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(DATES)


def test_save_cached_creates_directory(tmp_path):
    """save_cached creates nested cache directories that do not yet exist."""
    new_dir = tmp_path / "nested" / "cache"
    save_cached("AAPL", _ohlcv(DATES), str(new_dir))
    assert (new_dir / "AAPL.parquet").exists()


def test_save_load_round_trip(tmp_path):
    """save_cached followed by load_cached reproduces the original DataFrame."""
    original = _ohlcv(DATES)
    save_cached("MSFT", original, str(tmp_path))
    loaded = load_cached("MSFT", str(tmp_path))
    assert loaded is not None
    pd.testing.assert_frame_equal(original, loaded)


# ---------------------------------------------------------------------------
# pull_prices integration
# ---------------------------------------------------------------------------


def test_pull_prices_uses_stooq_when_yfinance_empty(tmp_path):
    """pull_prices routes to Stooq when yfinance returns an empty DataFrame."""
    cfg = Config(cache_dir=str(tmp_path))
    stooq_df = _ohlcv(DATES)

    with (
        patch("residrev.data._fetch_with_retry", return_value=pd.DataFrame()),
        patch("residrev.data.pull_stooq", return_value=stooq_df) as mock_stooq,
    ):
        result = pull_prices(["AAPL"], "2020-01-02", "2020-01-06", cfg)

    mock_stooq.assert_called_once_with("AAPL", "2020-01-02", "2020-01-06")
    assert "AAPL" in result


def test_pull_prices_uses_cache_and_skips_yfinance(tmp_path):
    """pull_prices returns cached data without calling yfinance when cache is fresh."""
    cfg = Config(cache_dir=str(tmp_path))
    # Cache spans beyond the requested range so _is_fresh passes
    cached_df = _ohlcv(["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-07"])
    save_cached("AAPL", cached_df, str(tmp_path))

    with patch("residrev.data._fetch_with_retry") as mock_yf:
        result = pull_prices(["AAPL"], "2020-01-02", "2020-01-06", cfg)

    mock_yf.assert_not_called()
    assert "AAPL" in result
