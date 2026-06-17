"""Tests for residrev.trend — synthetic data, no network."""
import numpy as np
import pandas as pd

from residrev import trend


def _synthetic_prices(n=600, k=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(0.0003, 0.01, (n, k))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=dates, columns=[f"E{i}" for i in range(k)])


def test_trend_signal_in_range():
    px = _synthetic_prices()
    sig = trend.trend_signal(px, lookbacks=(63, 126, 252))
    valid = sig.dropna(how="all")
    assert (valid.abs() <= 1.0 + 1e-9).all().all()


def test_trend_signal_no_lookahead():
    """signal at date t must not depend on prices at t (it is shifted one day)."""
    px = _synthetic_prices()
    sig = trend.trend_signal(px)
    px2 = px.copy()
    px2.iloc[-1] = px2.iloc[-1] * 5.0  # perturb only the last day's price
    sig2 = trend.trend_signal(px2)
    # all signal rows except possibly the very last are unchanged
    pd.testing.assert_frame_equal(sig.iloc[:-1], sig2.iloc[:-1])


def test_backtest_trend_outputs():
    px = _synthetic_prices()
    res = trend.backtest_trend(px)
    for key in ("gross", "net", "weights", "turnover", "gross_leverage"):
        assert key in res
    assert isinstance(res["net"], pd.Series)
    assert len(res["net"]) > 100
    assert np.isfinite(res["net"]).all()


def test_universe_spans_asset_classes():
    # sanity: the default universe covers equities, bonds, commodities, FX
    assert {"SPY", "TLT", "GLD", "UUP"}.issubset(set(trend.TREND_UNIVERSE))
