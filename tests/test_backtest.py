"""Tests for residrev.backtest — daily simulation loop."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from residrev.backtest import BacktestResult, _spearman_ic, run_backtest
from residrev.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


def _make_tickers(n: int = 10) -> list[str]:
    return [f"T{i:02d}" for i in range(n)]


def _make_factors_df(dates: pd.DatetimeIndex, factor_names: tuple) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        rng.randn(len(dates), len(factor_names)) * 0.01,
        index=dates,
        columns=list(factor_names),
    )


def _make_test_config() -> Config:
    return Config(
        factors=("F1", "F2"),
        gamma=1.0,
        lam_to=0.0,
        max_w=0.5,
        gross_cap=2.0,
        beta_tol=1e-2,
        sector_tol=1e-2,
        sigma_f_window=20,
        eta_impact=0.5,
        adv_participation_cap=0.10,
    )


def _make_full_inputs(
    n_dates: int = 30, n_tickers: int = 60, factor_names: tuple = ("F1", "F2"),
    signal_value: float | None = None,
):
    """Build synthetic panels for run_backtest.

    If signal_value is None, uses random signal; otherwise fills with constant.
    Factor history is extended 80 business days before the signal start so that
    get_factor_cov has enough lookback (needs >= 60 observations).
    """
    rng = np.random.RandomState(42)
    dates = _make_dates(n_dates)
    tickers = _make_tickers(n_tickers)

    if signal_value is not None:
        sig = pd.DataFrame(signal_value, index=dates, columns=tickers)
    else:
        sig = pd.DataFrame(
            rng.randn(n_dates, n_tickers) * 0.01, index=dates, columns=tickers
        )

    returns = pd.DataFrame(
        rng.randn(n_dates, n_tickers) * 0.005, index=dates, columns=tickers
    )

    betas = {
        f: pd.DataFrame(
            rng.randn(n_dates, n_tickers) * 0.3, index=dates, columns=tickers
        )
        for f in factor_names
    }

    idio_var = pd.DataFrame(
        rng.rand(n_dates, n_tickers) * 0.01 + 0.001,
        index=dates,
        columns=tickers,
    )

    pre_dates = pd.bdate_range(end=dates[0] - pd.Timedelta(days=1), periods=80, freq="B")
    all_factor_dates = pre_dates.append(dates)
    factors = _make_factors_df(all_factor_dates, factor_names)

    sector_map = {}
    sectors = ["Tech", "Health", "Energy", "Finance", "Consumer"]
    for i, t in enumerate(tickers):
        sector_map[t] = sectors[i % len(sectors)]

    spread = pd.DataFrame(
        rng.rand(n_dates, n_tickers) * 0.001 + 0.0001,
        index=dates,
        columns=tickers,
    )

    adv = pd.DataFrame(
        rng.rand(n_dates, n_tickers) * 1e7 + 1e6,
        index=dates,
        columns=tickers,
    )

    vol = pd.DataFrame(
        rng.rand(n_dates, n_tickers) * 0.2 + 0.1,
        index=dates,
        columns=tickers,
    )

    config = _make_test_config()

    return {
        "tradeable_signal": sig,
        "returns": returns,
        "betas": betas,
        "idio_var": idio_var,
        "factors": factors,
        "sector_map": sector_map,
        "spread": spread,
        "adv": adv,
        "vol": vol,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def inputs():
    return _make_full_inputs()


@pytest.fixture()
def result(inputs):
    return run_backtest(**inputs)


# ---------------------------------------------------------------------------
# BacktestResult dataclass tests
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_instantiation_types(self, result):
        assert isinstance(result.pnl, pd.Series)
        assert isinstance(result.gross_pnl, pd.Series)
        assert isinstance(result.positions, pd.DataFrame)
        assert isinstance(result.turnover, pd.Series)
        assert isinstance(result.costs_bps, pd.Series)
        assert isinstance(result.factor_exposures, pd.DataFrame)
        assert isinstance(result.ic_series, pd.Series)
        assert isinstance(result.meta, dict)

    def test_picklable(self, result):
        data = pickle.dumps(result)
        restored = pickle.loads(data)
        pd.testing.assert_series_equal(result.pnl, restored.pnl)
        pd.testing.assert_frame_equal(result.positions, restored.positions)
        assert result.meta == restored.meta


# ---------------------------------------------------------------------------
# run_backtest output shape and consistency
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_pnl_length(self, inputs, result):
        """PnL length equals number of active (post-warmup) dates."""
        sig = inputs["tradeable_signal"]
        warmup_mask = sig.notna().sum(axis=1) >= 50
        expected_len = warmup_mask.sum()
        assert len(result.pnl) == expected_len

    def test_positions_dollar_neutral(self, result):
        """Positions sum to approximately 0 per date (dollar-neutral)."""
        row_sums = result.positions.sum(axis=1)
        assert row_sums.abs().max() < 0.05, (
            f"Max absolute row sum = {row_sums.abs().max():.4f}"
        )

    def test_turnover_consistency(self, result):
        """Turnover[t] matches |positions[t] - positions[t-1]|.sum()."""
        pos = result.positions
        for i in range(1, len(pos)):
            expected = (pos.iloc[i] - pos.iloc[i - 1]).abs().sum()
            actual = result.turnover.iloc[i]
            assert actual == pytest.approx(expected, abs=1e-10)

    def test_gross_net_relationship(self, result):
        """net_pnl = gross_pnl - costs_bps / 10000."""
        expected_net = result.gross_pnl - result.costs_bps / 10_000
        pd.testing.assert_series_equal(
            result.pnl, expected_net, check_names=False, atol=1e-12,
        )

    def test_meta_keys(self, result):
        assert "config" in result.meta
        assert "start_date" in result.meta
        assert "end_date" in result.meta
        assert "n_warmup_days" in result.meta
        assert "universe_size_mean" in result.meta
        assert "total_trades" in result.meta

    def test_factor_exposures_columns(self, inputs, result):
        expected_cols = list(inputs["config"].factors)
        assert list(result.factor_exposures.columns) == expected_cols


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_signal_yields_near_zero(self):
        """Zero signal → weights ≈ 0, PnL ≈ 0."""
        inp = _make_full_inputs(signal_value=0.0)
        res = run_backtest(**inp)
        assert res.positions.abs().max().max() < 1e-6
        assert res.pnl.abs().max() < 1e-8

    def test_no_lookahead(self):
        """Changing returns[t+1] does not affect positions[t]."""
        inp1 = _make_full_inputs()
        res1 = run_backtest(**inp1)

        inp2 = _make_full_inputs()
        rng = np.random.RandomState(999)
        dates = inp2["returns"].index
        for d in dates[15:]:
            inp2["returns"].loc[d] = rng.randn(len(inp2["returns"].columns)) * 0.1
        res2 = run_backtest(**inp2)

        check_dates = res1.positions.index[:15]
        pd.testing.assert_frame_equal(
            res1.positions.loc[check_dates],
            res2.positions.loc[check_dates],
        )

    def test_warmup_exclusion(self):
        """Dates where signal is all-NaN are excluded from output."""
        inp = _make_full_inputs(n_dates=40)
        inp["tradeable_signal"].iloc[:10] = np.nan
        res = run_backtest(**inp)
        warmup_mask = inp["tradeable_signal"].notna().sum(axis=1) >= 50
        expected_dates = inp["tradeable_signal"].index[warmup_mask]
        assert len(res.pnl) <= len(expected_dates)
        for d in res.pnl.index:
            assert d in expected_dates


# ---------------------------------------------------------------------------
# _spearman_ic tests
# ---------------------------------------------------------------------------

class TestSpearmanIc:

    def test_nan_below_30_pairs(self):
        sig = pd.Series(range(20), dtype=float)
        ret = pd.Series(range(20), dtype=float)
        assert np.isnan(_spearman_ic(sig, ret))

    def test_perfect_correlation(self):
        vals = np.arange(50, dtype=float)
        sig = pd.Series(vals)
        ret = pd.Series(vals)
        assert _spearman_ic(sig, ret) == pytest.approx(1.0)

    def test_handles_nans(self):
        rng = np.random.RandomState(7)
        sig = pd.Series(rng.randn(60))
        ret = pd.Series(rng.randn(60))
        sig.iloc[:5] = np.nan
        ret.iloc[55:] = np.nan
        ic = _spearman_ic(sig, ret)
        assert not np.isnan(ic)

    def test_negative_correlation(self):
        vals = np.arange(50, dtype=float)
        sig = pd.Series(vals)
        ret = pd.Series(-vals)
        assert _spearman_ic(sig, ret) == pytest.approx(-1.0)
