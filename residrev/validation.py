"""Pre-trust validation checklist: CPCV splits, deflated Sharpe, stress tests."""

from __future__ import annotations

import json
import logging
from itertools import combinations
from math import sqrt
from typing import TYPE_CHECKING, Generator

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from residrev.analysis import annualized_sharpe, max_drawdown

if TYPE_CHECKING:
    from residrev.backtest import BacktestResult
    from residrev.config import Config

logger = logging.getLogger(__name__)


def cpcv_splits(
    dates: pd.DatetimeIndex,
    n_groups: int = 6,
    k_test: int = 2,
    purge: int = 5,
    embargo: float = 0.01,
) -> Generator[tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
    """Combinatorial Purged Cross-Validation splits (Lopez de Prado 2018).

    Yields (train_dates, test_dates) for each C(n_groups, k_test) combination.
    """
    T = len(dates)
    group_size = T // n_groups
    boundaries = [
        (i * group_size, min((i + 1) * group_size, T))
        for i in range(n_groups)
    ]
    # Last group absorbs remainder
    boundaries[-1] = (boundaries[-1][0], T)

    embargo_days = max(1, int(T * embargo))

    for test_group_ids in combinations(range(n_groups), k_test):
        test_idx = sorted(set(
            idx
            for g in test_group_ids
            for idx in range(*boundaries[g])
        ))

        excluded = set(test_idx)
        for g in test_group_ids:
            g_start, g_end = boundaries[g]
            for i in range(max(0, g_start - purge), g_start):
                excluded.add(i)
            for i in range(g_end, min(T, g_end + embargo_days)):
                excluded.add(i)

        train_idx = [i for i in range(T) if i not in excluded]
        yield dates[train_idx], dates[test_idx]


def skip_day_test(result: BacktestResult) -> dict:
    """Check whether PnL is concentrated on specific weekdays."""
    pnl = result.pnl.dropna()
    overall = annualized_sharpe(pnl)

    by_weekday: dict[int, float] = {}
    for wd in range(5):
        mask = pnl.index.weekday == wd
        subset = pnl[mask]
        by_weekday[wd] = annualized_sharpe(subset) if len(subset.dropna()) >= 20 else float("nan")

    max_diff = 0.0
    for wd, s in by_weekday.items():
        if not np.isnan(s) and not np.isnan(overall):
            max_diff = max(max_diff, abs(s - overall))

    if max_diff > 2.0:
        status = "fail"
    elif max_diff > 1.0:
        status = "warn"
    else:
        status = "pass"

    logger.info("Skip-day test: %s (max weekday deviation=%.2f)", status.upper(), max_diff)
    return {
        "status": status,
        "by_weekday": by_weekday,
        "overall_sharpe": overall,
        "note": f"Max weekday Sharpe deviation from overall: {max_diff:.2f}",
    }


def cost_stress_test(result: BacktestResult) -> dict:
    """Check that the strategy survives a 2x cost assumption."""
    sharpe_1x = annualized_sharpe(result.pnl)
    net_pnl_2x = result.gross_pnl - result.costs_bps * 2.0 / 10_000
    sharpe_2x = annualized_sharpe(net_pnl_2x)

    if np.isnan(sharpe_2x) or sharpe_2x < 0:
        status = "fail"
    elif sharpe_2x < 0.3:
        status = "warn"
    else:
        status = "pass"

    logger.info("Cost stress test: %s (1x=%.2f, 2x=%.2f)", status.upper(), sharpe_1x, sharpe_2x)
    return {
        "status": status,
        "sharpe_1x": sharpe_1x,
        "sharpe_2x": sharpe_2x,
        "note": f"Sharpe at 2x costs: {sharpe_2x:.2f}",
    }


def factor_crash_stress(result: BacktestResult) -> dict:
    """Check strategy performance during COVID crash and rate-hike regime."""
    pnl = result.pnl.dropna()
    date_range = (pnl.index.min(), pnl.index.max())

    periods = {
        "covid": (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-31")),
        "rate_hike": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    }

    results_by_period: dict[str, dict] = {}
    n_below = 0
    n_checked = 0

    for name, (start, end) in periods.items():
        if start > date_range[1] or end < date_range[0]:
            results_by_period[name] = {"sharpe": float("nan"), "max_dd": float("nan"), "note": "outside backtest range"}
            continue

        mask = (pnl.index >= start) & (pnl.index <= end)
        subset = pnl[mask]
        if len(subset.dropna()) < 5:
            results_by_period[name] = {"sharpe": float("nan"), "max_dd": float("nan"), "note": "insufficient data"}
            continue

        n_checked += 1
        s = annualized_sharpe(subset)
        dd = max_drawdown(subset)
        results_by_period[name] = {"sharpe": s, "max_dd": dd}

        if not np.isnan(s) and s < -0.5:
            n_below += 1

    if n_checked == 0:
        status = "pass"
    elif n_below >= 2:
        status = "fail"
    elif n_below == 1:
        status = "warn"
    else:
        status = "pass"

    logger.info("Factor crash stress: %s (%d/%d periods below -0.5)", status.upper(), n_below, n_checked)
    return {"status": status, **results_by_period}


def deflated_sharpe(trials_log_path: str, result: BacktestResult) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
    try:
        with open(trials_log_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.info("Deflated Sharpe: skip (trials log not found)")
        return {"status": "skip", "note": "trials log not found"}

    trials = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            trials.append(entry["net_sharpe"])
        except (json.JSONDecodeError, KeyError):
            continue

    if len(trials) < 2:
        logger.info("Deflated Sharpe: skip (insufficient trials: %d)", len(trials))
        return {"status": "skip", "note": "insufficient trials"}

    T_trials = len(trials)
    sr_star = max(trials)

    clean_pnl = result.pnl.dropna()
    n = len(clean_pnl)
    if n < 20:
        return {"status": "skip", "note": "insufficient return observations"}

    gamma3 = float(skew(clean_pnl.values))
    gamma4 = float(kurtosis(clean_pnl.values, fisher=True))

    v_sr = (1 - gamma3 * sr_star + (gamma4 - 1) / 4 * sr_star**2) / (n - 1)

    e_max_sr = float(norm.ppf(1 - 1 / T_trials))

    if v_sr > 0:
        dsr = float(norm.cdf((sr_star - e_max_sr) / sqrt(v_sr)))
    else:
        dsr = 0.0

    if dsr >= 0.95:
        status = "pass"
    elif dsr >= 0.80:
        status = "warn"
    else:
        status = "fail"

    logger.info("Deflated Sharpe: %s (DSR=%.3f, %d trials)", status.upper(), dsr, T_trials)
    return {
        "status": status,
        "dsr": dsr,
        "n_trials": T_trials,
        "best_sharpe": sr_star,
        "e_max_sharpe": e_max_sr,
    }


def run_pre_trust_checklist(result: BacktestResult, config: Config) -> dict:
    """Run all validation checks and print a formatted report."""
    checks: dict[str, dict] = {}
    checks["skip_day_test"] = skip_day_test(result)
    checks["cost_stress_test"] = cost_stress_test(result)
    checks["factor_crash_stress"] = factor_crash_stress(result)
    checks["deflated_sharpe"] = deflated_sharpe(config.trials_log, result)

    n_pass = sum(1 for c in checks.values() if c["status"] == "pass")
    n_warn = sum(1 for c in checks.values() if c["status"] == "warn")
    n_fail = sum(1 for c in checks.values() if c["status"] == "fail")
    n_skip = sum(1 for c in checks.values() if c["status"] == "skip")

    if n_fail > 0:
        overall = "fail"
    elif n_warn > 0:
        overall = "warn"
    else:
        overall = "pass"

    _print_report(checks, n_pass, n_warn, n_fail, n_skip, overall, result)

    return {
        "checks": checks,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "overall": overall,
    }


def _print_report(
    checks: dict[str, dict],
    n_pass: int,
    n_warn: int,
    n_fail: int,
    n_skip: int,
    overall: str,
    result: BacktestResult,
) -> None:
    """Print formatted pre-trust checklist report."""
    overall_sharpe = annualized_sharpe(result.pnl)
    lines = [
        "=" * 40,
        "PRE-TRUST VALIDATION CHECKLIST",
        "=" * 40,
    ]

    sd = checks["skip_day_test"]
    lines.append(f"[{sd['status'].upper()}] Skip-a-day test: {sd['note']}")

    cs = checks["cost_stress_test"]
    lines.append(f"[{cs['status'].upper()}] Cost stress (2x): Sharpe = {cs['sharpe_2x']:.2f}")

    fc = checks["factor_crash_stress"]
    for period in ("covid", "rate_hike"):
        if period in fc:
            info = fc[period]
            label = "COVID" if period == "covid" else "Rate hike"
            if "note" in info:
                lines.append(f"[{fc['status'].upper()}] Factor crash - {label}: {info['note']}")
            else:
                lines.append(
                    f"[{fc['status'].upper()}] Factor crash - {label}: "
                    f"Sharpe = {info['sharpe']:.2f}, max_dd = {info['max_dd'] * 100:.1f}%"
                )

    ds = checks["deflated_sharpe"]
    if ds["status"] == "skip":
        lines.append(f"[SKIP] Deflated Sharpe Ratio: {ds['note']}")
    else:
        lines.append(
            f"[{ds['status'].upper()}] Deflated Sharpe Ratio: "
            f"DSR = {ds['dsr']:.2f} ({ds['n_trials']} trials)"
        )

    lines.append("=" * 40)
    lines.append(f"Result: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  {n_skip} SKIP")
    print("\n".join(lines))
