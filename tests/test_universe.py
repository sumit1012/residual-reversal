"""10 tests for residrev.universe point-in-time universe construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residrev.universe import compute_adv, get_liquid_universe, get_universe_size_over_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(tickers: list[str], n_dates: int, close: float = 100.0, volume: float = 1_000.0) -> dict[str, pd.DataFrame]:
    """Build a uniform OHLCV prices dict for testing."""
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    out = {}
    for t in tickers:
        out[t] = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume},
            index=dates,
        )
    return out


def _make_adv_panel(n_dates: int, tickers: list[str], values: dict[str, float] | None = None) -> pd.DataFrame:
    """Build a synthetic ADV panel with constant per-ticker values."""
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    data = {}
    for t in tickers:
        v = values[t] if values else 1.0
        data[t] = [v] * n_dates
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_adv
# ---------------------------------------------------------------------------

def test_compute_adv_shape():
    """compute_adv returns a DataFrame with shape (n_dates, n_tickers)."""
    tickers = ["A", "B", "C"]
    prices = _make_prices(tickers, n_dates=50)
    adv = compute_adv(prices)
    assert adv.shape == (50, 3)
    assert set(adv.columns) == set(tickers)


def test_compute_adv_empty_ticker_gives_nan_column():
    """Ticker with an empty DataFrame produces an all-NaN column in the ADV panel."""
    prices = {
        "GOOD": _make_prices(["GOOD"], 50)["GOOD"],
        "EMPTY": pd.DataFrame(),
    }
    adv = compute_adv(prices)
    assert "EMPTY" in adv.columns
    assert adv["EMPTY"].isna().all()


def test_compute_adv_respects_min_periods():
    """No ADV values appear before 40 data points have accumulated."""
    prices = _make_prices(["X"], n_dates=50)
    adv = compute_adv(prices)
    col = adv["X"]
    # Positions 0-38 (39 rows) must be NaN; position 39+ must be valid
    assert col.iloc[:39].isna().all(), "Expected NaN for first 39 dates"
    assert not col.iloc[39:].isna().any(), "Expected valid values from date 40 onward"


def test_compute_adv_rolling_mean_spot_check():
    """Rolling mean equals the constant dollar volume for a constant-value series."""
    close, volume = 50.0, 2_000.0
    expected_dv = close * volume  # 100_000
    prices = _make_prices(["X"], n_dates=70, close=close, volume=volume)
    adv = compute_adv(prices)
    # At position 69 (day 70) the full window is filled — value must equal expected_dv
    assert adv["X"].iloc[69] == pytest.approx(expected_dv)


# ---------------------------------------------------------------------------
# get_liquid_universe
# ---------------------------------------------------------------------------

def test_get_liquid_universe_returns_bool_dtype():
    """get_liquid_universe output dtype is bool."""
    adv = _make_adv_panel(10, ["A", "B", "C", "D", "E"])
    result = get_liquid_universe(adv, universe_size=3, buffer=1)
    assert result.dtypes.eq(bool).all()


def test_get_liquid_universe_first_date_top_n():
    """On the first date, exactly the top-N tickers by ADV are selected."""
    # 8 tickers with distinct ADV values; top-3 should be T1, T2, T3
    tickers = [f"T{i}" for i in range(1, 9)]
    values = {f"T{i}": float(9 - i) for i in range(1, 9)}  # T1=8, T2=7, ..., T8=1
    adv = _make_adv_panel(5, tickers, values)
    result = get_liquid_universe(adv, universe_size=3, buffer=1)
    first_day = result.iloc[0]
    assert first_day["T1"] and first_day["T2"] and first_day["T3"]
    assert not first_day["T4"]


def test_get_liquid_universe_hysteresis_rank_within_buffer_stays():
    """A member whose rank falls to 4 stays in a universe_size=3, buffer=2 universe (threshold=5)."""
    # 8 tickers; date 0: T1>T2>T3>T4>...>T8 → top-3 = {T1,T2,T3}
    # date 1: T3 drops to rank 4 (still ≤ 3+2=5) → T3 should stay
    tickers = [f"T{i}" for i in range(1, 9)]
    dates = pd.date_range("2020-01-01", periods=2, freq="B")

    # Day 0: T1=8,T2=7,T3=6,T4=5,...,T8=1
    # Day 1: T4 overtakes T3 so T3 drops to rank 4
    adv_data = {
        "T1": [8.0, 8.0],
        "T2": [7.0, 7.0],
        "T3": [6.0, 5.5],  # T3 drops below T4 → rank 4
        "T4": [5.0, 6.0],  # T4 rises above T3
        "T5": [4.0, 4.0],
        "T6": [3.0, 3.0],
        "T7": [2.0, 2.0],
        "T8": [1.0, 1.0],
    }
    adv = pd.DataFrame(adv_data, index=dates)
    result = get_liquid_universe(adv, universe_size=3, buffer=2)
    assert result.loc[dates[1], "T3"], "T3 (rank 4) should stay within buffer=2"


def test_get_liquid_universe_hysteresis_rank_beyond_buffer_exits():
    """A member whose rank falls to 6 exits a universe_size=3, buffer=2 universe (threshold=5)."""
    tickers = [f"T{i}" for i in range(1, 9)]
    dates = pd.date_range("2020-01-01", periods=2, freq="B")

    adv_data = {
        "T1": [8.0, 8.0],
        "T2": [7.0, 7.0],
        "T3": [6.0, 2.5],  # T3 plummets to rank 6 (below T5,T6)
        "T4": [5.0, 5.0],
        "T5": [4.0, 4.0],
        "T6": [3.0, 3.0],
        "T7": [2.0, 2.0],
        "T8": [1.0, 1.0],
    }
    adv = pd.DataFrame(adv_data, index=dates)
    result = get_liquid_universe(adv, universe_size=3, buffer=2)
    assert not result.loc[dates[1], "T3"], "T3 (rank 6) should exit beyond buffer=2"


def test_get_liquid_universe_no_lookahead():
    """Membership at date t is unaffected by changing ADV at date t+1."""
    tickers = ["A", "B", "C", "D", "E"]
    adv = _make_adv_panel(10, tickers, values={"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})

    membership_original = get_liquid_universe(adv, universe_size=3, buffer=1)

    # Mutate future dates (indices 5-9) with completely different ADV values
    adv_modified = adv.copy()
    adv_modified.iloc[5:] = adv_modified.iloc[5:] * 100.0  # drastic change to future data

    membership_modified = get_liquid_universe(adv_modified, universe_size=3, buffer=1)

    # Past membership (dates 0-4) must be identical
    pd.testing.assert_frame_equal(
        membership_original.iloc[:5],
        membership_modified.iloc[:5],
    )


# ---------------------------------------------------------------------------
# get_universe_size_over_time
# ---------------------------------------------------------------------------

def test_get_universe_size_over_time_counts():
    """get_universe_size_over_time matches row-wise sum of the membership panel."""
    tickers = [f"T{i}" for i in range(1, 9)]
    values = {f"T{i}": float(9 - i) for i in range(1, 9)}
    adv = _make_adv_panel(10, tickers, values)
    membership = get_liquid_universe(adv, universe_size=4, buffer=2)
    size_series = get_universe_size_over_time(membership)
    expected = membership.sum(axis=1)
    pd.testing.assert_series_equal(size_series, expected)
