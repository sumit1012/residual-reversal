"""Daily simulation loop wiring signal, optimizer, and cost modules."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from residrev.config import Config
from residrev.costs import compute_rebalance_cost
from residrev.factors import get_factor_cov
from residrev.portfolio import optimize_book

logger = logging.getLogger(__name__)

_MIN_IC_PAIRS = 30


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


def _spearman_ic(signal: pd.Series, returns: pd.Series) -> float:
    """Cross-sectional Spearman correlation; NaN if < 30 valid pairs."""
    mask = signal.notna() & returns.notna()
    common = mask.sum()
    if common < _MIN_IC_PAIRS:
        return np.nan
    return spearmanr(signal[mask], returns[mask]).statistic


def _slice_inputs(
    date: pd.Timestamp,
    betas: dict[str, pd.DataFrame],
    idio_var: pd.DataFrame,
    spread: pd.DataFrame,
    adv: pd.DataFrame,
    vol: pd.DataFrame,
) -> dict:
    """Slice all panel inputs at a single date and align on ticker index."""
    betas_t = {k: v.loc[date].dropna() for k, v in betas.items()}
    common = betas_t[next(iter(betas_t))].index
    for s in betas_t.values():
        common = common.intersection(s.index)

    idio_t = idio_var.loc[date].dropna()
    spread_t = spread.loc[date].dropna()
    adv_t = adv.loc[date].dropna()
    vol_t = vol.loc[date].dropna()

    common = common.intersection(idio_t.index)
    common = common.intersection(spread_t.index)
    common = common.intersection(adv_t.index)
    common = common.intersection(vol_t.index)

    return {
        "betas": {k: v.reindex(common) for k, v in betas_t.items()},
        "idio_var": idio_t.reindex(common),
        "spread": spread_t.reindex(common),
        "adv": adv_t.reindex(common),
        "vol": vol_t.reindex(common),
        "tickers": common,
    }


def run_backtest(
    tradeable_signal: pd.DataFrame,
    returns: pd.DataFrame,
    betas: dict[str, pd.DataFrame],
    idio_var: pd.DataFrame,
    factors: pd.DataFrame,
    sector_map: dict[str, str],
    spread: pd.DataFrame,
    adv: pd.DataFrame,
    vol: pd.DataFrame,
    config: Config,
) -> BacktestResult:
    """Run daily simulation loop over pre-computed panels.

    tradeable_signal[t] is already shifted in signal.py to use only data
    through t-1.  This module never shifts anything.
    """
    panels = [tradeable_signal, returns, idio_var, spread, adv, vol]
    for b in betas.values():
        panels.append(b)

    common_dates = panels[0].index
    for p in panels[1:]:
        common_dates = common_dates.intersection(p.index)
    common_dates = common_dates.sort_values()

    n_lost = len(tradeable_signal.index) - len(common_dates)
    if n_lost > 0:
        logger.info("Dropped %d dates during inner-join alignment", n_lost)

    warmup_mask = tradeable_signal.loc[common_dates].notna().sum(axis=1) >= 50
    active_dates = common_dates[warmup_mask]
    n_warmup = len(common_dates) - len(active_dates)
    logger.info(
        "Active dates: %d (skipped %d warmup)", len(active_dates), n_warmup
    )

    all_tickers = tradeable_signal.columns
    w_prev = pd.Series(0.0, index=all_tickers)

    pnl_records: list[tuple[pd.Timestamp, float]] = []
    gross_records: list[tuple[pd.Timestamp, float]] = []
    pos_records: list[tuple[pd.Timestamp, pd.Series]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    cost_records: list[tuple[pd.Timestamp, float]] = []
    exposure_records: list[tuple[pd.Timestamp, dict]] = []
    ic_records: list[tuple[pd.Timestamp, float]] = []
    total_trades = 0

    for i, date in enumerate(active_dates):
        sliced = _slice_inputs(date, betas, idio_var, spread, adv, vol)
        tickers_t = sliced["tickers"]

        if len(tickers_t) == 0:
            w_t = pd.Series(0.0, index=all_tickers)
        else:
            factor_cov_t = get_factor_cov(factors, date, config.sigma_f_window)
            if factor_cov_t is None:
                pos_records.append((date, w_prev.copy()))
                pnl_records.append((date, 0.0))
                gross_records.append((date, 0.0))
                turnover_records.append((date, 0.0))
                cost_records.append((date, 0.0))
                exposure_records.append(
                    (date, {k: 0.0 for k in config.factors})
                )
                ic_records.append((date, np.nan))
                continue

            alpha_t = tradeable_signal.loc[date].reindex(tickers_t).fillna(0.0)
            w_prev_t = w_prev.reindex(tickers_t).fillna(0.0)

            w_t_raw = optimize_book(
                alpha=alpha_t,
                betas=sliced["betas"],
                idio_var=sliced["idio_var"],
                factor_cov=factor_cov_t,
                sector_labels=sector_map,
                config=config,
                w_prev=w_prev_t,
            )
            w_t = pd.Series(0.0, index=all_tickers)
            w_t.update(w_t_raw)

        ret_t = returns.loc[date].reindex(all_tickers).fillna(0.0)
        gross_pnl_t = (w_t * ret_t).sum()

        if len(tickers_t) > 0:
            cost_spread_t = sliced["spread"]
            cost_adv_t = sliced["adv"]
            cost_vol_t = sliced["vol"]
            w_prev_cost = w_prev.reindex(tickers_t).fillna(0.0)
            w_new_cost = w_t.reindex(tickers_t).fillna(0.0)
            costs_bps_t = compute_rebalance_cost(
                w_prev_cost, w_new_cost,
                cost_spread_t, cost_adv_t, cost_vol_t, config,
                aum=config.aum,
            )
        else:
            costs_bps_t = 0.0

        net_pnl_t = gross_pnl_t - costs_bps_t / 10_000
        turnover_t = (w_t - w_prev).abs().sum()
        total_trades += (w_t - w_prev).ne(0).sum()

        factor_exp_t = {}
        for k in config.factors:
            if k in betas and date in betas[k].index:
                beta_k = betas[k].loc[date].reindex(all_tickers).fillna(0.0)
                factor_exp_t[k] = (beta_k * w_t).sum()
            else:
                factor_exp_t[k] = 0.0

        ic_t = _spearman_ic(
            tradeable_signal.loc[date], returns.loc[date]
        )

        pnl_records.append((date, net_pnl_t))
        gross_records.append((date, gross_pnl_t))
        pos_records.append((date, w_t.copy()))
        turnover_records.append((date, turnover_t))
        cost_records.append((date, costs_bps_t))
        exposure_records.append((date, factor_exp_t))
        ic_records.append((date, ic_t))

        w_prev = w_t

        if (i + 1) % 50 == 0 or i == len(active_dates) - 1:
            logger.info("Backtest progress: %d / %d dates", i + 1, len(active_dates))

    dates_out = [r[0] for r in pnl_records]
    pnl = pd.Series([r[1] for r in pnl_records], index=dates_out, name="pnl")
    gross_pnl = pd.Series(
        [r[1] for r in gross_records], index=dates_out, name="gross_pnl"
    )
    positions = pd.DataFrame(
        [r[1] for r in pos_records], index=dates_out
    )
    turnover = pd.Series(
        [r[1] for r in turnover_records], index=dates_out, name="turnover"
    )
    costs_bps = pd.Series(
        [r[1] for r in cost_records], index=dates_out, name="costs_bps"
    )
    factor_exposures = pd.DataFrame(
        [r[1] for r in exposure_records], index=dates_out
    )
    ic_series = pd.Series(
        [r[1] for r in ic_records], index=dates_out, name="ic"
    )

    universe_sizes = tradeable_signal.loc[active_dates].notna().sum(axis=1)

    meta = {
        "config": config.to_dict(),
        "start_date": str(dates_out[0].date()) if dates_out else None,
        "end_date": str(dates_out[-1].date()) if dates_out else None,
        "n_warmup_days": n_warmup,
        "universe_size_mean": float(universe_sizes.mean()) if len(universe_sizes) > 0 else 0.0,
        "total_trades": int(total_trades),
    }

    return BacktestResult(
        pnl=pnl,
        gross_pnl=gross_pnl,
        positions=positions,
        turnover=turnover,
        costs_bps=costs_bps,
        factor_exposures=factor_exposures,
        ic_series=ic_series,
        meta=meta,
    )
