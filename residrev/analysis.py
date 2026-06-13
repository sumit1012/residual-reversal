"""Performance metrics computed from a BacktestResult — no backtest re-runs."""

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from residrev.backtest import BacktestResult
    from residrev.config import Config

logger = logging.getLogger(__name__)

_AUM_BASE = 1e8  # baseline AUM used in compute_rebalance_cost


def annualized_sharpe(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a daily PnL series."""
    clean = pnl.dropna()
    if len(clean) < 20:
        return float("nan")
    std = clean.std()
    if std == 0:
        return float("nan")
    return float(clean.mean() / std * sqrt(periods_per_year))


def max_drawdown(pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of cumulative PnL (returns a negative number)."""
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def per_year_sharpe(pnl: pd.Series) -> dict[int, float]:
    """Annualized Sharpe ratio computed separately for each calendar year."""
    result: dict[int, float] = {}
    for year, group in pnl.groupby(pnl.index.year):
        if len(group.dropna()) < 50:
            result[int(year)] = float("nan")
        else:
            result[int(year)] = annualized_sharpe(group)
    return result


def factor_exposure_report(result: BacktestResult) -> pd.DataFrame:
    """Summary statistics on factor exposure time series from a BacktestResult.

    A well-built factor-neutral portfolio should have |mean| < 1e-3 and
    |t_stat| < 2 for all factors.
    """
    fe = result.factor_exposures
    records = {}
    for col in fe.columns:
        s = fe[col].dropna()
        n = len(s)
        mean = float(s.mean()) if n > 0 else float("nan")
        std = float(s.std()) if n > 1 else float("nan")
        max_abs = float(s.abs().max()) if n > 0 else float("nan")
        t_stat = float(mean / (std / sqrt(n))) if (n > 1 and std > 0) else float("nan")
        records[col] = {"mean": mean, "std": std, "max_abs": max_abs, "t_stat": t_stat}
        if abs(t_stat) > 2:
            logger.warning(
                "Factor '%s' has |t_stat| = %.2f > 2 — possible residual exposure", col, t_stat
            )
    return pd.DataFrame.from_dict(records, orient="index", columns=["mean", "std", "max_abs", "t_stat"])


def cost_sensitivity(
    result: BacktestResult,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Net performance across a range of cost-scaling assumptions.

    Does not re-run the backtest — scales result.costs_bps analytically.
    Appends a 'breakeven_multiplier' row interpolated where net Sharpe = 0.
    """
    if multipliers is None:
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]

    rows: dict[float | str, dict] = {}
    for m in multipliers:
        net_pnl_m = result.gross_pnl - result.costs_bps * m / 10_000
        rows[m] = {
            "net_sharpe": annualized_sharpe(net_pnl_m),
            "annualized_return": float(net_pnl_m.mean() * 252),
            "max_drawdown": max_drawdown(net_pnl_m),
        }

    # Linear interpolation for the breakeven multiplier (net Sharpe = 0)
    sharpes = [(m, rows[m]["net_sharpe"]) for m in multipliers]
    bm = float("nan")
    for i in range(len(sharpes) - 1):
        m0, s0 = sharpes[i]
        m1, s1 = sharpes[i + 1]
        if not (np.isnan(s0) or np.isnan(s1)) and s0 * s1 <= 0:
            # s0 and s1 bracket zero
            bm = float(m0 + (0 - s0) * (m1 - m0) / (s1 - s0))
            break

    rows["breakeven_multiplier"] = {
        "net_sharpe": bm,
        "annualized_return": float("nan"),
        "max_drawdown": float("nan"),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["net_sharpe", "annualized_return", "max_drawdown"])


def capacity_curve(
    result: BacktestResult,
    config: Config,
    aum_values: list[float] | None = None,
) -> pd.DataFrame:
    """Estimate how net Sharpe degrades as AUM grows using sqrt-impact scaling.

    Assumes a 40/60 spread/impact split of total costs at baseline AUM, since
    the combined BacktestResult cost cannot be cleanly decomposed post-hoc.
    As AUM scales by x = aum / aum_base, the spread component is constant and
    the market-impact component scales as sqrt(x).
    """
    if aum_values is None:
        aum_values = [1e6, 5e6, 1e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9]

    total_cost_annual_bps = float(result.costs_bps.mean() * 252)
    spread_cost_annual = 0.40 * total_cost_annual_bps
    impact_cost_annual = 0.60 * total_cost_annual_bps

    gross_sharpe = annualized_sharpe(result.gross_pnl)

    rows: dict[float, dict] = {}
    for aum in aum_values:
        x = aum / _AUM_BASE
        cost_at_aum = spread_cost_annual + impact_cost_annual * sqrt(x)
        net_pnl = result.gross_pnl - cost_at_aum / 10_000 / 252
        rows[aum] = {
            "net_sharpe": annualized_sharpe(net_pnl),
            "gross_sharpe": gross_sharpe,
            "total_cost_bps_per_year": cost_at_aum,
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["net_sharpe", "gross_sharpe", "total_cost_bps_per_year"])


def summarize(result: BacktestResult) -> dict:
    """Compute all key metrics in one call. Returns a JSON-serializable dict."""
    net_pnl = result.pnl
    gross_pnl = result.gross_pnl
    ic = result.ic_series.dropna()

    ic_mean = float(ic.mean()) if len(ic) > 0 else float("nan")
    ic_std = float(ic.std()) if len(ic) > 1 else float("nan")
    n_ic = len(ic)
    ic_tstat = float(ic_mean / (ic_std / sqrt(n_ic))) if (n_ic > 1 and ic_std > 0) else float("nan")

    fe = result.factor_exposures
    factor_mean_exposures = {col: float(fe[col].mean()) for col in fe.columns}

    return {
        "net_sharpe": annualized_sharpe(net_pnl),
        "gross_sharpe": annualized_sharpe(gross_pnl),
        "net_annual_return": float(net_pnl.mean() * 252),
        "gross_annual_return": float(gross_pnl.mean() * 252),
        "max_drawdown": max_drawdown(net_pnl),
        "annual_turnover": float(result.turnover.mean() * 252),
        "mean_daily_ic": ic_mean,
        "ic_tstat": ic_tstat,
        "mean_cost_bps": float(result.costs_bps.mean()),
        "total_cost_bps_pa": float(result.costs_bps.mean() * 252),
        "per_year_sharpe": per_year_sharpe(net_pnl),
        "factor_exposures": factor_mean_exposures,
        "n_trading_days": int(len(net_pnl)),
        "start_date": str(result.meta.get("start_date", "")),
        "end_date": str(result.meta.get("end_date", "")),
    }
