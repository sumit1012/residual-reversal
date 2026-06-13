"""Tests for residrev/analysis.py — all use synthetic BacktestResult instances."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from residrev.analysis import (
    annualized_sharpe,
    capacity_curve,
    cost_sensitivity,
    factor_exposure_report,
    max_drawdown,
    per_year_sharpe,
    summarize,
)
from residrev.config import Config


# Lightweight stand-in — mirrors the real BacktestResult fields used by analysis.py
@dataclass
class BacktestResult:
    pnl: pd.Series
    gross_pnl: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs_bps: pd.Series
    factor_exposures: pd.DataFrame
    ic_series: pd.Series
    meta: dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pnl(n: int = 252, mean: float = 0.001, std: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    vals = rng.normal(mean, std, size=n)
    return pd.Series(vals, index=dates, name="pnl")


def _make_result(
    n: int = 252,
    pnl_mean: float = 0.001,
    cost_bps: float = 2.0,
    n_factors: int = 3,
    seed: int = 42,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    gross_pnl = pd.Series(rng.normal(pnl_mean, 0.01, n), index=dates, name="gross_pnl")
    costs = pd.Series(np.full(n, cost_bps), index=dates, name="costs_bps")
    net_pnl = gross_pnl - costs / 10_000

    factors = [f"f{i}" for i in range(n_factors)]
    fe = pd.DataFrame(rng.normal(0, 1e-4, (n, n_factors)), index=dates, columns=factors)
    ic = pd.Series(rng.normal(0.03, 0.05, n), index=dates, name="ic")
    turnover = pd.Series(np.full(n, 0.1), index=dates, name="turnover")
    positions = pd.DataFrame(np.zeros((n, 10)), index=dates)

    return BacktestResult(
        pnl=net_pnl,
        gross_pnl=gross_pnl,
        positions=positions,
        turnover=turnover,
        costs_bps=costs,
        factor_exposures=fe,
        ic_series=ic,
        meta={
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
        },
    )


# ---------------------------------------------------------------------------
# annualized_sharpe
# ---------------------------------------------------------------------------

def test_annualized_sharpe_known_value():
    # spec: daily mean=0.001, std=0.01, 252 days → Sharpe ≈ 0.001/0.01 * sqrt(252)
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=252)
    vals = rng.normal(0.001, 0.01, 252)
    pnl = pd.Series(vals, index=dates)
    expected = vals.mean() / vals.std(ddof=1) * math.sqrt(252)
    assert abs(annualized_sharpe(pnl) - expected) < 1e-9


def test_annualized_sharpe_formula():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=252)
    vals = rng.normal(0.001, 0.01, 252)
    pnl = pd.Series(vals, index=dates)
    expected = vals.mean() / vals.std(ddof=1) * math.sqrt(252)
    assert abs(annualized_sharpe(pnl) - expected) < 1e-10


def test_annualized_sharpe_returns_nan_when_std_zero():
    dates = pd.bdate_range("2020-01-02", periods=252)
    pnl = pd.Series([0.0] * 252, index=dates)
    assert math.isnan(annualized_sharpe(pnl))


def test_annualized_sharpe_returns_nan_too_few_obs():
    dates = pd.bdate_range("2020-01-02", periods=10)
    pnl = pd.Series([0.001] * 10, index=dates)
    assert math.isnan(annualized_sharpe(pnl))


def test_annualized_sharpe_drops_nan():
    dates = pd.bdate_range("2020-01-02", periods=252)
    vals = [0.001] * 252
    vals[0] = float("nan")
    pnl = pd.Series(vals, index=dates)
    result = annualized_sharpe(pnl)
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_non_positive():
    pnl = _make_pnl(252)
    assert max_drawdown(pnl) <= 0.0


def test_max_drawdown_monotone_increasing_is_zero():
    dates = pd.bdate_range("2020-01-02", periods=100)
    pnl = pd.Series([0.01] * 100, index=dates)
    assert max_drawdown(pnl) == pytest.approx(0.0)


def test_max_drawdown_known_case():
    dates = pd.bdate_range("2020-01-02", periods=6)
    # cumsum: 1, 2, 3, 2, 1, 2  → drawdown from peak 3: -2
    pnl = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0, 1.0], index=dates)
    assert max_drawdown(pnl) == pytest.approx(-2.0)


def test_max_drawdown_empty():
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# per_year_sharpe
# ---------------------------------------------------------------------------

def test_per_year_sharpe_one_key_per_year():
    dates = pd.bdate_range("2020-01-02", periods=504)  # ~2 years
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(0.001, 0.01, 504), index=dates)
    result = per_year_sharpe(pnl)
    years_in_index = set(dates.year)
    assert set(result.keys()) == years_in_index


def test_per_year_sharpe_short_year_is_nan():
    # 2020 has <50 days, 2021 has plenty
    dates_20 = pd.bdate_range("2020-12-01", periods=23)  # only 23 days
    dates_21 = pd.bdate_range("2021-01-04", periods=252)
    dates = dates_20.append(dates_21)
    rng = np.random.default_rng(2)
    pnl = pd.Series(rng.normal(0.001, 0.01, len(dates)), index=dates)
    result = per_year_sharpe(pnl)
    assert math.isnan(result[2020])
    assert math.isfinite(result[2021])


# ---------------------------------------------------------------------------
# factor_exposure_report
# ---------------------------------------------------------------------------

def test_factor_exposure_report_shape():
    result = _make_result(n_factors=4)
    df = factor_exposure_report(result)
    assert df.shape == (4, 4)
    assert list(df.columns) == ["mean", "std", "max_abs", "t_stat"]


def test_factor_exposure_report_t_stat_sign():
    # Factor with positive mean should have positive t_stat
    n = 252
    dates = pd.bdate_range("2020-01-02", periods=n)
    fe = pd.DataFrame({"pos": np.full(n, 0.1), "neg": np.full(n, -0.1)}, index=dates)
    result = _make_result()
    result = BacktestResult(
        pnl=result.pnl,
        gross_pnl=result.gross_pnl,
        positions=result.positions,
        turnover=result.turnover,
        costs_bps=result.costs_bps,
        factor_exposures=fe,
        ic_series=result.ic_series,
        meta=result.meta,
    )
    df = factor_exposure_report(result)
    # std of constant series is 0 → t_stat will be nan (can't divide by 0)
    # test just that sign is consistent when not degenerate
    rng = np.random.default_rng(99)
    fe2 = pd.DataFrame(
        {"pos": rng.normal(0.5, 0.01, n), "neg": rng.normal(-0.5, 0.01, n)},
        index=dates,
    )
    result2 = BacktestResult(
        pnl=result.pnl,
        gross_pnl=result.gross_pnl,
        positions=result.positions,
        turnover=result.turnover,
        costs_bps=result.costs_bps,
        factor_exposures=fe2,
        ic_series=result.ic_series,
        meta=result.meta,
    )
    df2 = factor_exposure_report(result2)
    assert df2.loc["pos", "t_stat"] > 0
    assert df2.loc["neg", "t_stat"] < 0


# ---------------------------------------------------------------------------
# cost_sensitivity
# ---------------------------------------------------------------------------

def test_cost_sensitivity_multiplier_zero_equals_gross():
    result = _make_result()
    df = cost_sensitivity(result, multipliers=[0.0])
    # net_pnl at m=0 = gross_pnl
    expected = annualized_sharpe(result.gross_pnl)
    assert df.loc[0.0, "net_sharpe"] == pytest.approx(expected, rel=1e-6)


def test_cost_sensitivity_net_sharpe_decreases_with_multiplier():
    result = _make_result()
    mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    df = cost_sensitivity(result, multipliers=mults)
    sharpes = [df.loc[m, "net_sharpe"] for m in mults]
    # each step should be non-increasing (with positive costs)
    for i in range(len(sharpes) - 1):
        assert sharpes[i] >= sharpes[i + 1] - 1e-10


def test_cost_sensitivity_has_breakeven_row():
    result = _make_result()
    df = cost_sensitivity(result)
    assert "breakeven_multiplier" in df.index


def test_cost_sensitivity_default_multipliers():
    result = _make_result()
    df = cost_sensitivity(result)
    # 5 multiplier rows + 1 breakeven
    assert len(df) == 6


# ---------------------------------------------------------------------------
# capacity_curve
# ---------------------------------------------------------------------------

def test_capacity_curve_net_sharpe_decreases_with_aum():
    result = _make_result()
    config = Config()
    aum_values = [1e7, 1e8, 1e9]
    df = capacity_curve(result, config, aum_values=aum_values)
    sharpes = [df.loc[aum, "net_sharpe"] for aum in aum_values]
    for i in range(len(sharpes) - 1):
        assert sharpes[i] >= sharpes[i + 1] - 1e-10


def test_capacity_curve_shape():
    result = _make_result()
    config = Config()
    aum_values = [1e7, 1e8, 1e9]
    df = capacity_curve(result, config, aum_values=aum_values)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["net_sharpe", "gross_sharpe", "total_cost_bps_per_year"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_json_serializable():
    result = _make_result()
    s = summarize(result)
    json.dumps(s)  # must not raise


def test_summarize_net_annual_return_lt_gross():
    result = _make_result(pnl_mean=0.002, cost_bps=2.0)
    s = summarize(result)
    assert s["net_annual_return"] < s["gross_annual_return"]


def test_summarize_keys_present():
    result = _make_result()
    s = summarize(result)
    expected_keys = [
        "net_sharpe", "gross_sharpe", "net_annual_return", "gross_annual_return",
        "max_drawdown", "annual_turnover", "mean_daily_ic", "ic_tstat",
        "mean_cost_bps", "total_cost_bps_pa", "per_year_sharpe",
        "factor_exposures", "n_trading_days", "start_date", "end_date",
    ]
    for k in expected_keys:
        assert k in s, f"Missing key: {k}"


def test_summarize_no_numpy_scalars():
    result = _make_result()
    s = summarize(result)
    # json.dumps raises on numpy scalars; this is the definitive check
    serialized = json.dumps(s)
    assert isinstance(serialized, str)


def test_summarize_n_trading_days():
    result = _make_result(n=300)
    s = summarize(result)
    assert s["n_trading_days"] == 300


def test_summarize_per_year_sharpe_is_dict():
    result = _make_result()
    s = summarize(result)
    assert isinstance(s["per_year_sharpe"], dict)
