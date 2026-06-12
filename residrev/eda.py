"""Exploratory analysis functions that produce exhibit DataFrames."""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_THIN_THRESHOLD = 50


def compute_return_stats(
    prices: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional daily log-return distribution, restricted to universe members.

    Returns DataFrame indexed by date with columns:
    mean, std, skew, kurt, p01, p05, p25, p50, p75, p95, p99, thin_cross_section.
    """
    # Build log-returns panel (dates × tickers)
    ret_series: dict[str, pd.Series] = {}
    for ticker, df in prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        ret_series[ticker] = df["Close"].apply(float).pipe(
            lambda s: s.pct_change().pipe(lambda r: (1 + r).apply(math.log))
        )

    if not ret_series:
        return pd.DataFrame()

    returns = pd.DataFrame(ret_series)

    # Align to membership index
    shared_dates = membership.index.intersection(returns.index)
    returns = returns.reindex(shared_dates)
    mem = membership.reindex(shared_dates, columns=returns.columns).fillna(False)

    rows = []
    for date in shared_dates:
        mask = mem.loc[date].astype(bool)
        vals = returns.loc[date][mask].dropna()
        n = len(vals)
        thin = n < _THIN_THRESHOLD
        if thin:
            logger.warning("Thin cross-section (%d members) on %s", n, date)
        if n == 0:
            rows.append({"date": date, "thin_cross_section": True})
            continue
        rows.append(
            {
                "date": date,
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "skew": vals.skew(),
                "kurt": vals.kurt(),
                "p01": vals.quantile(0.01),
                "p05": vals.quantile(0.05),
                "p25": vals.quantile(0.25),
                "p50": vals.quantile(0.50),
                "p75": vals.quantile(0.75),
                "p95": vals.quantile(0.95),
                "p99": vals.quantile(0.99),
                "thin_cross_section": thin,
            }
        )

    df = pd.DataFrame(rows).set_index("date")
    df["thin_cross_section"] = df["thin_cross_section"].fillna(False).astype(bool)
    return df


def compute_adv_stats(
    adv: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """ADV distribution for universe members over time.

    Returns DataFrame indexed by date with columns:
    mean, std, p25, p50, p75, min, max.
    """
    shared_dates = membership.index.intersection(adv.index)
    adv_aligned = adv.reindex(shared_dates, columns=membership.columns)
    mem_aligned = membership.reindex(shared_dates, columns=adv.columns).fillna(False)

    rows = []
    for date in shared_dates:
        mask = mem_aligned.loc[date].astype(bool)
        vals = adv_aligned.loc[date][mask].dropna()
        if vals.empty:
            rows.append({"date": date})
            continue
        rows.append(
            {
                "date": date,
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "p25": vals.quantile(0.25),
                "p50": vals.quantile(0.50),
                "p75": vals.quantile(0.75),
                "min": vals.min(),
                "max": vals.max(),
            }
        )

    return pd.DataFrame(rows).set_index("date")


def compute_ic_at_k(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    k_values: Optional[list[int]] = None,
) -> pd.DataFrame:
    """IC profile at multiple horizons.

    For each k, computes cross-sectional Spearman IC between signal.iloc[i]
    and fwd_returns.iloc[i+k] for each valid date i.

    Returns DataFrame indexed by k with columns: mean_ic, std_ic, t_stat, n_obs.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 21]

    # Align columns
    common_cols = signal.columns.intersection(fwd_returns.columns)
    sig = signal[common_cols]
    fwd = fwd_returns[common_cols]

    n_dates = min(len(sig), len(fwd))

    rows = []
    for k in k_values:
        ics = []
        for i in range(n_dates - k):
            s_row = sig.iloc[i].dropna()
            f_row = fwd.iloc[i + k].reindex(s_row.index).dropna()
            shared = s_row.index.intersection(f_row.index)
            if len(shared) < 5:
                continue
            corr, _ = spearmanr(s_row[shared], f_row[shared])
            if not math.isnan(corr):
                ics.append(corr)

        if len(ics) == 0:
            rows.append({"k": k, "mean_ic": float("nan"), "std_ic": float("nan"), "t_stat": float("nan"), "n_obs": 0})
            continue

        ic_series = pd.Series(ics)
        mean_ic = ic_series.mean()
        std_ic = ic_series.std(ddof=1)
        n_obs = len(ics)
        t_stat = mean_ic / (std_ic / math.sqrt(n_obs)) if std_ic > 0 else float("nan")
        rows.append({"k": k, "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat, "n_obs": n_obs})

    return pd.DataFrame(rows).set_index("k")


def compute_universe_turnover(membership: pd.DataFrame) -> pd.Series:
    """Daily turnover rate: (entries + exits) / universe size.

    Returns Series indexed by date; first date is NaN (no prior state).
    """
    mem = membership.astype(bool)
    prev = pd.DataFrame(False, index=mem.index, columns=mem.columns, dtype=bool)
    prev.iloc[1:] = mem.iloc[:-1].values
    entries = (~prev) & mem
    exits = prev & (~mem)

    size = membership.sum(axis=1).replace(0, float("nan"))
    turnover = (entries.sum(axis=1) + exits.sum(axis=1)) / size
    # First row has no prior state
    turnover.iloc[0] = float("nan")
    return turnover


def run_eda(
    prices: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    adv: pd.DataFrame,
    signal: Optional[pd.DataFrame] = None,
    fwd_returns: Optional[pd.DataFrame] = None,
    output_dir: str = "data/results/eda/",
) -> dict[str, Optional[pd.DataFrame]]:
    """Run all EDA exhibits and save to CSVs.

    Returns dict with keys: return_stats, adv_stats, universe_turnover, ic_profile.
    ic_profile is None when signal or fwd_returns is not provided.
    """
    os.makedirs(output_dir, exist_ok=True)

    return_stats = compute_return_stats(prices, membership)
    adv_stats = compute_adv_stats(adv, membership)
    universe_turnover = compute_universe_turnover(membership)

    ic_profile: Optional[pd.DataFrame] = None
    if signal is not None and fwd_returns is not None:
        ic_profile = compute_ic_at_k(signal, fwd_returns)
    else:
        logger.info("Skipping IC profile: signal or fwd_returns not provided")

    return_stats.to_csv(os.path.join(output_dir, "return_stats.csv"))
    adv_stats.to_csv(os.path.join(output_dir, "adv_stats.csv"))
    universe_turnover.to_csv(os.path.join(output_dir, "universe_turnover.csv"), header=True)
    if ic_profile is not None:
        ic_profile.to_csv(os.path.join(output_dir, "ic_profile.csv"))

    return {
        "return_stats": return_stats,
        "adv_stats": adv_stats,
        "universe_turnover": universe_turnover,
        "ic_profile": ic_profile,
    }
