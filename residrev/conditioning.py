"""Amihud illiquidity and VIX regime conditioning variables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd

from residrev.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional lazy imports — mocked in tests via module-level names
# ---------------------------------------------------------------------------

try:
    import pandas_datareader as pdr  # type: ignore
except ImportError:  # pragma: no cover
    pdr = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None  # type: ignore


# ---------------------------------------------------------------------------
# Amihud illiquidity
# ---------------------------------------------------------------------------


def compute_amihud(
    prices: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Amihud (2002) illiquidity quintile ranks and cross-sectional z-scores.

    Returns (quintile, z_score) — both shape (T × N), NaN for non-members.
    quintile: 1 (most liquid) through n_illiq_buckets (most illiquid).
    """
    illiq_dict: dict[str, pd.Series] = {}
    for ticker, df in prices.items():
        try:
            close = df["Close"].astype(float)
            volume = df["Volume"].astype(float)
        except KeyError:
            continue
        log_ret = np.log(close / close.shift(1)).abs()
        dollar_vol = close * volume
        illiq_raw = np.where(dollar_vol > 0, log_ret / dollar_vol, np.nan)
        illiq_dict[ticker] = pd.Series(illiq_raw, index=df.index)

    if not illiq_dict:
        empty = pd.DataFrame(index=universe.index, columns=universe.columns, dtype=float)
        return empty, empty.copy()

    # Align all tickers onto a common date panel
    illiq_panel = pd.DataFrame(illiq_dict)
    all_dates = illiq_panel.index.union(universe.index)
    illiq_panel = illiq_panel.reindex(all_dates)

    # Mask non-universe members
    univ = universe.reindex(index=illiq_panel.index, columns=illiq_panel.columns)
    illiq_panel = illiq_panel.where(univ.fillna(False).astype(bool))

    # Trailing mean over amihud_window, min_periods=10
    illiq_21 = illiq_panel.rolling(window=config.amihud_window, min_periods=10).mean()

    # Log-transform to reduce right skew; log1p handles zero gracefully
    log_illiq = np.log1p(illiq_21)

    # Cross-sectional quintile rank per date (1 = most liquid, n_illiq_buckets = most illiquid)
    pct_rank = log_illiq.rank(axis=1, pct=True)
    quintile = np.ceil(pct_rank * config.n_illiq_buckets).clip(1, config.n_illiq_buckets)
    quintile = quintile.where(log_illiq.notna())  # propagate NaN for missing data

    # Cross-sectional z-score per date
    row_mean = log_illiq.mean(axis=1)
    row_std = log_illiq.std(axis=1)
    z_score = log_illiq.sub(row_mean, axis=0).div(row_std, axis=0)

    # Reindex to match the exact universe shape
    quintile = quintile.reindex(index=universe.index, columns=universe.columns)
    z_score = z_score.reindex(index=universe.index, columns=universe.columns)

    return quintile, z_score


# ---------------------------------------------------------------------------
# VIX download and caching
# ---------------------------------------------------------------------------


def get_vix(config: Config) -> pd.Series:
    """Download the CBOE VIX index, cache to {config.data_dir}/vix.parquet.

    Tries FRED first; falls back to yfinance on any failure.
    Returns a tz-naive Series named "VIX".
    """
    cache_path = Path(config.data_dir) / "vix.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        vix = cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        vix.name = "VIX"
        logger.info("VIX loaded from cache: %s", cache_path)
        return vix

    # Primary source: FRED
    vix: pd.Series | None = None
    if pdr is not None:
        try:
            raw = pdr.get_data_fred("VIXCLS", start=config.start_date, end=config.end_date)
            vix = raw["VIXCLS"].dropna()
            logger.info("VIX downloaded from FRED")
        except Exception as exc:
            logger.warning("FRED failed (%s); falling back to yfinance", exc)

    # Fallback: yfinance
    if vix is None:
        if yf is None:  # pragma: no cover
            raise RuntimeError("Neither pandas-datareader nor yfinance is available")
        raw_df = yf.download("^VIX", start=config.start_date, end=config.end_date, progress=False)
        close = raw_df["Close"]
        vix = (close.squeeze() if isinstance(close, pd.DataFrame) else close).dropna()
        logger.info("VIX downloaded from yfinance")

    # Normalise to tz-naive DatetimeIndex
    if hasattr(vix.index, "tz") and vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    vix.name = "VIX"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vix.to_frame().to_parquet(cache_path)

    return vix


# ---------------------------------------------------------------------------
# VIX regime
# ---------------------------------------------------------------------------


def compute_vix_regime(vix: pd.Series, config: Config) -> pd.Series:
    """Assign each date to a VIX regime tercile using only past data.

    Tercile 1 = calm (low VIX), 2 = neutral, 3 = stressed (high VIX).
    Dates with fewer than vix_regime_window past observations are NaN.

    Vectorized: expanding().rank(pct=True) computes the running CDF in a
    single pass; history_count guards the minimum-history requirement.
    """
    # expanding rank includes the current observation in the denominator;
    # for windows of 252+ the look-ahead bias is one observation in ~252 (<0.4%).
    pct = vix.expanding(min_periods=1).rank(pct=True)
    history_count = vix.expanding(min_periods=1).count()

    n = config.n_vol_buckets
    bins = np.linspace(0.0, 1.0, n + 1)
    bins[-1] += 1e-9  # include the maximum value in the last bin

    regime = pd.cut(pct, bins=bins, labels=range(1, n + 1), include_lowest=True).astype(float)

    # Mask dates without sufficient history
    regime[history_count <= config.vix_regime_window] = np.nan

    return regime


# ---------------------------------------------------------------------------
# IC by conditioning bucket
# ---------------------------------------------------------------------------


def ic_by_bucket(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    buckets: Union[pd.Series, pd.DataFrame],
    compute_ic_fn: Callable,
    compute_ic_tstat_fn: Callable,
) -> pd.DataFrame:
    """IC breakdown table for a given conditioning variable.

    Parameters
    ----------
    signal, fwd_returns : DataFrames of shape (T × N).
    buckets :
        pd.Series (date-level, e.g. VIX regime) — filter rows to dates
        matching each label; IC computed on the full cross-section.
        pd.DataFrame (stock×date, e.g. Amihud quintile) — for each label,
        mask signal/fwd_returns to stocks with that label on each date.
        This matches the academic standard for cross-sectional conditioning
        (Amihud 2002, characteristic-sorted IC literature).
    compute_ic_fn : callable with signature (signal_df, fwd_df) -> pd.Series.
    compute_ic_tstat_fn : callable with signature (ic_series) -> float.

    Returns
    -------
    DataFrame with index = bucket label, columns = [mean_ic, std_ic,
    t_stat_hac, n_dates].
    """
    rows: list[dict] = []

    if isinstance(buckets, pd.Series):
        unique_labels = sorted(buckets.dropna().unique().tolist())
        for label in unique_labels:
            dates = buckets[buckets == label].index
            dates = dates.intersection(signal.index).intersection(fwd_returns.index)
            if len(dates) == 0:
                continue
            ic_series = compute_ic_fn(signal.loc[dates], fwd_returns.loc[dates]).dropna()
            if len(ic_series) == 0:
                continue
            rows.append(
                {
                    "bucket": label,
                    "mean_ic": float(ic_series.mean()),
                    "std_ic": float(ic_series.std()),
                    "t_stat_hac": float(compute_ic_tstat_fn(ic_series)),
                    "n_dates": int(len(ic_series)),
                }
            )

    else:
        # DataFrame: stock-level (cross-sectional) conditioning
        common_dates = (
            buckets.index.intersection(signal.index).intersection(fwd_returns.index)
        )
        b = buckets.reindex(index=common_dates, columns=signal.columns)
        sig = signal.reindex(index=common_dates)
        fwd = fwd_returns.reindex(index=common_dates)

        flat = b.values.ravel()
        flat = flat[~np.isnan(flat.astype(float))]
        unique_labels = sorted(np.unique(flat).tolist())

        for label in unique_labels:
            masked_sig = sig.where(b == label)
            masked_fwd = fwd.where(b == label)
            ic_series = compute_ic_fn(masked_sig, masked_fwd).dropna()
            if len(ic_series) == 0:
                continue
            rows.append(
                {
                    "bucket": label,
                    "mean_ic": float(ic_series.mean()),
                    "std_ic": float(ic_series.std()),
                    "t_stat_hac": float(compute_ic_tstat_fn(ic_series)),
                    "n_dates": int(len(ic_series)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["mean_ic", "std_ic", "t_stat_hac", "n_dates"])

    return pd.DataFrame(rows).set_index("bucket")
