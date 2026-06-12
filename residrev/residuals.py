"""Core residualization engine — vectorized rolling past-only 6-factor OLS."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from residrev.config import Config

logger = logging.getLogger(__name__)


def build_return_panel(
    prices: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert raw price dict to (T x N) daily log-return panel.

    Only includes (date, ticker) cells where universe membership is True;
    everything else is NaN.
    """
    series: dict[str, pd.Series] = {}
    for ticker, df in prices.items():
        close = df["Close"].sort_index()
        log_ret = np.log(close / close.shift(1))
        series[ticker] = log_ret

    panel = pd.DataFrame(series)
    panel = panel.reindex(index=panel.index.union(universe.index).sort_values())
    panel = panel.reindex(columns=universe.columns)

    mask = universe.reindex(index=panel.index, columns=panel.columns).fillna(False)
    panel = panel.where(mask)
    return panel


def rolling_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Estimate rolling past-only factor model and extract residuals.

    Returns (resid, betas, idio_var) where:
      - resid:    (T x N) residual returns
      - betas:    dict of (T x N) DataFrames, one per factor + "intercept"
      - idio_var: (T x N) idiosyncratic return variance
    """
    t_start = time.perf_counter()
    logger.info("rolling_residuals: starting estimation")

    window = config.factor_window
    min_obs = config.min_obs
    factor_names = list(config.factors)
    K = len(factor_names)

    common_dates = returns.index.intersection(factors.index)
    pct_lost = 1.0 - len(common_dates) / max(len(returns.index), 1)
    if pct_lost > 0.05:
        logger.warning(
            "rolling_residuals: %.1f%% of dates lost in alignment "
            "(%d -> %d dates)",
            pct_lost * 100,
            len(returns.index),
            len(common_dates),
        )

    returns_aligned = returns.reindex(index=common_dates)
    factors_aligned = factors.reindex(index=common_dates)

    rf = factors_aligned["RF"].values if "RF" in factors_aligned.columns else np.zeros(len(common_dates))
    factor_mat = factors_aligned[factor_names].values  # (T, K)

    ret_mat = returns_aligned.values.copy()  # (T, N)
    T, N = ret_mat.shape

    excess_mat = ret_mat - rf[:, np.newaxis]

    resid_out = np.full((T, N), np.nan)
    idio_var_out = np.full((T, N), np.nan)
    beta_out = np.full((T, N, K + 1), np.nan)  # intercept + K factors

    ones = np.ones((window, 1))

    for t in range(window, T):
        assert t >= window, "window start must be non-negative"

        X_win = factor_mat[t - window : t]  # (W, K)
        X_win_int = np.column_stack([ones, X_win])  # (W, K+1)
        y_win = excess_mat[t - window : t].copy()  # (W, N)

        valid_obs = (~np.isnan(y_win)).sum(axis=0)  # (N,)
        insufficient = valid_obs < min_obs

        y_win_clean = np.where(np.isnan(y_win), 0.0, y_win)

        beta_t, _, _, _ = np.linalg.lstsq(X_win_int, y_win_clean, rcond=None)
        # beta_t: (K+1, N)

        x_t = np.concatenate([[1.0], factor_mat[t]])  # (K+1,)
        resid_t = excess_mat[t] - x_t @ beta_t  # (N,)

        window_preds = X_win_int @ beta_t  # (W, N)
        window_resids = y_win_clean - window_preds  # (W, N)
        idio_var_t = np.var(window_resids, ddof=K + 2, axis=0)  # (N,)

        resid_t[insufficient] = np.nan
        beta_t[:, insufficient] = np.nan
        idio_var_t[insufficient] = np.nan

        resid_out[t] = resid_t
        idio_var_out[t] = idio_var_t
        beta_out[t] = beta_t.T  # (N, K+1)

        if (t - window) % 250 == 0 and t > window:
            logger.info(
                "rolling_residuals: processed %d / %d dates", t - window, T - window
            )

    betas: dict[str, pd.DataFrame] = {}
    betas["intercept"] = pd.DataFrame(
        beta_out[:, :, 0], index=returns_aligned.index, columns=returns_aligned.columns
    )
    for i, name in enumerate(factor_names):
        betas[name] = pd.DataFrame(
            beta_out[:, :, i + 1],
            index=returns_aligned.index,
            columns=returns_aligned.columns,
        )

    resid = pd.DataFrame(
        resid_out, index=returns_aligned.index, columns=returns_aligned.columns
    )
    idio_var = pd.DataFrame(
        idio_var_out, index=returns_aligned.index, columns=returns_aligned.columns
    )

    elapsed = time.perf_counter() - t_start
    logger.info("rolling_residuals: done in %.1f s", elapsed)

    return resid, betas, idio_var


def compute_idio_vol(idio_var: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Smoothed idiosyncratic volatility from trailing mean of variance."""
    return np.sqrt(idio_var.rolling(window, min_periods=1).mean())
