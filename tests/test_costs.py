"""Tests for residrev/costs.py — all synthetic data, no external dependencies."""
import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.costs import (
    build_cost_panel,
    compute_rebalance_cost,
    compute_realized_vol,
    corwin_schultz_spread,
)


@pytest.fixture
def cfg() -> Config:
    return Config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc(
    n_days: int = 50, n_stocks: int = 3, seed: int = 42
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days)
    prices: dict[str, pd.DataFrame] = {}
    for i in range(n_stocks):
        close = 100.0 * np.exp(rng.normal(0, 0.01, n_days).cumsum())
        high = close * (1.0 + rng.uniform(0.001, 0.01, n_days))
        low = close * (1.0 - rng.uniform(0.001, 0.01, n_days))
        prices[f"S{i}"] = pd.DataFrame(
            {"High": high, "Low": low, "Close": close}, index=dates
        )
    return prices


def _make_returns(n_days: int = 60, n_stocks: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_stocks)),
        index=dates,
        columns=[f"S{i}" for i in range(n_stocks)],
    )


# ---------------------------------------------------------------------------
# corwin_schultz_spread
# ---------------------------------------------------------------------------


def test_cs_spread_output_shape(cfg):
    prices = _make_ohlc(50, 3)
    spread = corwin_schultz_spread(prices, window=21)
    assert spread.shape == (50, 3)


def test_cs_spread_non_negative(cfg):
    prices = _make_ohlc(50, 3)
    spread = corwin_schultz_spread(prices, window=21)
    assert (spread.dropna() >= 0.0).all().all()


def test_cs_spread_first_date_nan(cfg):
    """First row must be NaN — formula needs 2 days and rolling needs min_periods=5."""
    prices = _make_ohlc(50, 3)
    spread = corwin_schultz_spread(prices, window=21)
    assert spread.iloc[0].isna().all()


def test_cs_spread_zero_range_near_zero():
    """High == Low on every day → raw spread is 0 → smoothed spread ≈ 0."""
    dates = pd.date_range("2020-01-01", periods=30)
    prices = {
        "FLAT": pd.DataFrame(
            {"High": 100.0, "Low": 100.0, "Close": 100.0}, index=dates
        )
    }
    spread = corwin_schultz_spread(prices, window=5)
    valid = spread["FLAT"].dropna()
    assert not valid.empty
    assert (valid.abs() < 1e-8).all()


def test_cs_spread_missing_high_low_is_nan():
    """Tickers without High or Low columns should produce an all-NaN column."""
    dates = pd.date_range("2020-01-01", periods=30)
    prices = {"NO_HL": pd.DataFrame({"Close": np.ones(30)}, index=dates)}
    spread = corwin_schultz_spread(prices, window=5)
    assert spread["NO_HL"].isna().all()


def test_cs_spread_smoothing_window_effect():
    """Shorter window → fewer leading NaNs (starts producing values sooner)."""
    prices = _make_ohlc(60, 2)
    spread_w5 = corwin_schultz_spread(prices, window=5)
    spread_w21 = corwin_schultz_spread(prices, window=21)
    n_valid_w5 = spread_w5.notna().all(axis=1).sum()
    n_valid_w21 = spread_w21.notna().all(axis=1).sum()
    assert n_valid_w5 >= n_valid_w21


# ---------------------------------------------------------------------------
# compute_realized_vol
# ---------------------------------------------------------------------------


def test_realized_vol_non_negative(cfg):
    returns = _make_returns(60, 3)
    vol = compute_realized_vol(returns, window=21)
    assert (vol.dropna() >= 0.0).all().all()


def test_realized_vol_past_only():
    """Perturbing return at row 30 must not change vol output at row 29."""
    returns = _make_returns(60, 1)
    vol_before = compute_realized_vol(returns.copy(), window=21)

    returns_perturbed = returns.copy()
    returns_perturbed.iloc[30] = 999.0
    vol_after = compute_realized_vol(returns_perturbed, window=21)

    pd.testing.assert_frame_equal(vol_before.iloc[:30], vol_after.iloc[:30])


def test_realized_vol_annualized():
    """Vol should be annualised: daily std * sqrt(252)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=60)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 1)), index=dates, columns=["A"])
    vol = compute_realized_vol(rets, window=21)
    # At row 21 (first valid), manual check
    manual = rets["A"].iloc[0:20].std() * np.sqrt(252)  # shift(1) means window ends at t-1
    computed = vol["A"].iloc[21]
    assert abs(computed - manual) < 0.01  # within 1 vol point


# ---------------------------------------------------------------------------
# compute_rebalance_cost
# ---------------------------------------------------------------------------


def _cost_args(cfg):
    tickers = ["A", "B", "C"]
    w_prev = pd.Series([0.10, 0.20, 0.30], index=tickers)
    w_new = pd.Series([0.15, 0.25, 0.25], index=tickers)
    spread = pd.Series([0.001, 0.002, 0.0015], index=tickers)
    adv = pd.Series([5e6, 3e6, 4e6], index=tickers)
    vol = pd.Series([0.20, 0.25, 0.18], index=tickers)
    return w_prev, w_new, spread, adv, vol


def test_rebalance_cost_returns_float(cfg):
    w_prev, w_new, spread, adv, vol = _cost_args(cfg)
    cost = compute_rebalance_cost(w_prev, w_new, spread, adv, vol, cfg)
    assert isinstance(cost, float)


def test_rebalance_cost_zero_turnover(cfg):
    w_prev, _, spread, adv, vol = _cost_args(cfg)
    cost = compute_rebalance_cost(w_prev, w_prev, spread, adv, vol, cfg)
    assert cost == pytest.approx(0.0, abs=1e-10)


def test_rebalance_cost_higher_spread_higher_cost(cfg):
    tickers = ["A"]
    w_prev = pd.Series([0.10], index=tickers)
    w_new = pd.Series([0.20], index=tickers)
    adv = pd.Series([5e6], index=tickers)
    vol = pd.Series([0.20], index=tickers)

    cost_low = compute_rebalance_cost(
        w_prev, w_new, pd.Series([0.001], index=tickers), adv, vol, cfg
    )
    cost_high = compute_rebalance_cost(
        w_prev, w_new, pd.Series([0.010], index=tickers), adv, vol, cfg
    )
    assert cost_high > cost_low


def test_rebalance_cost_positive_for_nonzero_trade(cfg):
    tickers = ["A"]
    cost = compute_rebalance_cost(
        pd.Series([0.0], index=tickers),
        pd.Series([0.1], index=tickers),
        pd.Series([0.001], index=tickers),
        pd.Series([5e6], index=tickers),
        pd.Series([0.20], index=tickers),
        cfg,
    )
    assert cost > 0.0


def test_rebalance_cost_in_bps_reasonable_range(cfg):
    """Result should be a positive number in a plausible bps range."""
    w_prev, w_new, spread, adv, vol = _cost_args(cfg)
    cost = compute_rebalance_cost(w_prev, w_new, spread, adv, vol, cfg)
    assert 0.0 < cost < 10_000.0


def test_rebalance_cost_participation_uses_adv(cfg):
    """Higher ADV (more liquid) → lower participation → lower impact cost."""
    tickers = ["A"]
    w_prev = pd.Series([0.0], index=tickers)
    w_new = pd.Series([0.1], index=tickers)
    spread = pd.Series([0.0], index=tickers)   # zero spread, isolate impact
    vol = pd.Series([0.25], index=tickers)

    cost_illiquid = compute_rebalance_cost(
        w_prev, w_new, spread, pd.Series([1e5], index=tickers), vol, cfg
    )
    cost_liquid = compute_rebalance_cost(
        w_prev, w_new, spread, pd.Series([1e8], index=tickers), vol, cfg
    )
    assert cost_liquid < cost_illiquid


# ---------------------------------------------------------------------------
# build_cost_panel
# ---------------------------------------------------------------------------


def test_build_cost_panel_length(cfg):
    dates = pd.date_range("2020-01-01", periods=20)
    tickers = ["A", "B"]
    rng = np.random.default_rng(0)
    weights = pd.DataFrame(rng.uniform(0, 0.1, (20, 2)), index=dates, columns=tickers)
    spread = pd.DataFrame(0.001, index=dates, columns=tickers)
    adv = pd.DataFrame(5e6, index=dates, columns=tickers)
    vol = pd.DataFrame(0.20, index=dates, columns=tickers)

    panel = build_cost_panel(weights, spread, adv, vol, cfg)
    assert len(panel) == len(dates)


def test_build_cost_panel_zero_weight_change(cfg):
    """Constant weights after date 0 → cost ≈ 0 on dates 1+."""
    dates = pd.date_range("2020-01-01", periods=5)
    tickers = ["A", "B"]
    weights = pd.DataFrame(0.1, index=dates, columns=tickers)
    spread = pd.DataFrame(0.001, index=dates, columns=tickers)
    adv = pd.DataFrame(5e6, index=dates, columns=tickers)
    vol = pd.DataFrame(0.20, index=dates, columns=tickers)

    panel = build_cost_panel(weights, spread, adv, vol, cfg)
    assert panel.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-10)
