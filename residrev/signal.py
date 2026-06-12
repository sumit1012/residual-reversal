"""Reversal signal construction and IC diagnostics with HAC t-statistics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.stats.mstats import winsorize as _winsorize
import statsmodels.api as sm

from residrev.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def build_signal(
    resid: pd.DataFrame,
    sector_map: dict[str, str],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the raw and tradeable reversal signal.

    Returns (raw_signal, tradeable_signal) both shaped (T, N).
    """
    k = config.signal_k
    pct = config.winsorize_pct

    # Step 1 — cumulate and negate
    cum = resid.rolling(k, min_periods=k).sum()
    raw = -cum

    # Step 2 — cross-sectional winsorization per date
    winsorized = raw.copy()
    for i in range(len(winsorized)):
        row = winsorized.iloc[i]
        valid = row.dropna()
        if len(valid) < 50:
            continue
        lo = valid.quantile(pct)
        hi = valid.quantile(1 - pct)
        winsorized.iloc[i] = row.clip(lower=lo, upper=hi)

    # Step 3 — sector-demean
    sector_labels = pd.Series(
        {t: sector_map.get(t, "Other") for t in winsorized.columns}
    )
    demeaned = winsorized.copy()
    for i in range(len(demeaned)):
        row = demeaned.iloc[i]
        for sector in sector_labels.unique():
            members = sector_labels[sector_labels == sector].index
            vals = row[members].dropna()
            if len(vals) <= 1:
                continue
            sector_mean = vals.mean()
            demeaned.iloc[i, demeaned.columns.get_indexer(members)] -= sector_mean

    # Step 4 — cross-sectional z-score
    zscored = demeaned.copy()
    for i in range(len(zscored)):
        row = zscored.iloc[i]
        valid = row.dropna()
        if len(valid) == 0:
            continue
        mu = valid.mean()
        sigma = valid.std()
        if sigma == 0:
            zscored.iloc[i] = row.where(row.isna(), 0.0)
        else:
            zscored.iloc[i] = (row - mu) / sigma

    raw_signal = zscored

    # structural lag: signal[t] is built from data through t-1
    tradeable_signal = raw_signal.shift(1)

    logger.info(
        "build_signal: mean raw signal %.4f, std %.4f",
        raw_signal.stack().mean(),
        raw_signal.stack().std(),
    )

    return raw_signal, tradeable_signal


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------


def compute_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional IC per date.

    Dates with fewer than 30 valid pairs are set to NaN.
    """
    common_dates = signal.index.intersection(fwd_returns.index)
    ic_values = pd.Series(np.nan, index=common_dates)

    corr_fn = spearmanr if method == "spearman" else pearsonr

    for dt in common_dates:
        s = signal.loc[dt]
        f = fwd_returns.loc[dt]
        mask = s.notna() & f.notna()
        n_valid = mask.sum()
        if n_valid < 30:
            continue
        corr, _ = corr_fn(s[mask].values, f[mask].values)
        ic_values[dt] = corr

    return ic_values


def compute_ic_tstat(ic: pd.Series, bandwidth: int | None = None) -> float:
    """Newey-West HAC t-statistic for the mean IC."""
    clean = ic.dropna().values
    if len(clean) == 0:
        return 0.0
    if np.std(clean) == 0:
        return 0.0
    if bandwidth is None:
        bandwidth = int(len(clean) ** 0.25)
    model = sm.OLS(clean, np.ones(len(clean)))
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": bandwidth})
    return float(result.tvalues[0])


def compute_ic_decay(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    max_lag: int = 21,
) -> pd.DataFrame:
    """IC profile at multiple forward horizons.

    Returns DataFrame with index = lag (1..max_lag),
    columns = [mean_ic, std_ic, t_stat_hac, n_obs].
    """
    rows = []
    for h in range(1, max_lag + 1):
        fwd_h = returns.shift(-h).rolling(h).sum()
        ic_series = compute_ic(signal, fwd_h)
        clean = ic_series.dropna()
        mean_ic = clean.mean() if len(clean) > 0 else np.nan
        std_ic = clean.std() if len(clean) > 0 else np.nan
        t_stat = compute_ic_tstat(ic_series) if len(clean) > 0 else np.nan
        rows.append(
            {
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "t_stat_hac": t_stat,
                "n_obs": len(clean),
            }
        )

    return pd.DataFrame(rows, index=range(1, max_lag + 1))
