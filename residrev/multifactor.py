"""Daily cross-sectional multi-factor equity sleeve (the third sleeve).

A PRICE-ONLY low-risk-plus-residual-momentum book, built entirely from data the
repo already caches. Pre-registered in MULTIFACTOR_SPEC.md. The claim is
DIVERSIFICATION (drawdown reduction + a partial hedge of the reversal sleeve's
momentum-short), not standalone alpha; a near-zero net Sharpe is an accepted,
pre-registered outcome.

Why price-only: free data has no per-stock point-in-time fundamentals, so value /
quality / investment / earnings factors are not honestly backtestable and are
excluded (see MULTIFACTOR_SPEC.md section 3).

Factors (hard cap of 3):
  1. Residual 12-1 momentum  -- long firm-specific drift the reversal sleeve discards.
                                Residual (not raw) so it does not re-load the momentum
                                factor behind reversal's live loss or correlate with trend.
  2. Low risk (one of)       -- low idiosyncratic vol OR betting-against-beta.
  3. (optional) 52-week-high proximity -- only if its incremental IC clears.

Construction is intentionally low-degrees-of-freedom: 2-3 ex-ante factors, fixed
equal weights, pinned lookbacks, no per-stock parameters, and a transparent
rank-and-inverse-vol long/short (NOT the cvxpy optimizer, whose six-factor
neutralization would zero the very tilts this sleeve harvests, and whose solver
fragility hurt the reversal sleeve's reproducibility).
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from residrev.config import Config
from residrev.costs import build_cost_panel, compute_realized_vol, corwin_schultz_spread
from residrev.data import pull_prices
from residrev.factors import get_ff_factors, get_sector_map
from residrev.residuals import build_return_panel, compute_idio_vol, rolling_residuals
from residrev.universe import compute_adv, get_liquid_universe

logger = logging.getLogger(__name__)

MOM_LOOKBACK = 252   # 12 months
MOM_SKIP = 21        # skip most recent ~1 month (the reversal horizon reversal owns)
IDIO_VOL_WINDOW = 21
HIGH52_WINDOW = 252
REBAL_GAP = 21       # monthly rebalance; slow signals => low turnover


# ---------------------------------------------------------------------------
# Cross-sectional standardization (mirrors signal.build_signal steps 2-4)
# ---------------------------------------------------------------------------
def _xs_standardize(df: pd.DataFrame, sector_map: dict[str, str],
                    pct: float = 0.01, min_names: int = 50) -> pd.DataFrame:
    """Per-date winsorize -> French-12 sector-demean -> cross-sectional z-score."""
    lo = df.quantile(pct, axis=1)
    hi = df.quantile(1 - pct, axis=1)
    w = df.clip(lower=lo, upper=hi, axis=0)

    sectors = pd.Series({c: sector_map.get(c, "Other") for c in df.columns})
    out = w.copy()
    for s in sectors.unique():
        cols = sectors.index[sectors == s]
        if len(cols) == 0:
            continue
        m = w[cols].mean(axis=1)
        out[cols] = w[cols].sub(m, axis=0)

    mu = out.mean(axis=1)
    sd = out.std(axis=1).replace(0.0, np.nan)
    z = out.sub(mu, axis=0).div(sd, axis=0)
    valid = df.notna().sum(axis=1) >= min_names
    return z.where(valid, np.nan)


# ---------------------------------------------------------------------------
# Factor builders (all reuse the existing residual engine output)
# ---------------------------------------------------------------------------
def residual_momentum(resid: pd.DataFrame, lookback: int = MOM_LOOKBACK,
                      skip: int = MOM_SKIP) -> pd.DataFrame:
    """Cumulative FF5+UMD residual return over [t-lookback, t-skip] (12-1, residual form)."""
    full = resid.rolling(lookback, min_periods=lookback).sum()
    recent = resid.rolling(skip, min_periods=skip).sum()
    return full - recent


def low_risk_idiovol(idio_var: pd.DataFrame, window: int = IDIO_VOL_WINDOW) -> pd.DataFrame:
    """Low idiosyncratic volatility: long low-vol names (Ang-Hodrick-Xing-Zhang 2006)."""
    return -compute_idio_vol(idio_var, window)


def low_risk_bab(beta_mkt: pd.DataFrame) -> pd.DataFrame:
    """Betting-against-beta: long low-market-beta names (Frazzini-Pedersen 2014)."""
    return -beta_mkt


def high_52w(close: pd.DataFrame, window: int = HIGH52_WINDOW) -> pd.DataFrame:
    """52-week-high proximity: Close / trailing-252d max (George-Hwang 2004)."""
    return close / close.rolling(window, min_periods=window // 2).max()


def build_composite_alpha(factor_raws: dict[str, pd.DataFrame], sector_map: dict[str, str],
                          config: Config) -> pd.DataFrame:
    """Equal-weight composite of standardized factor z-scores, re-z-scored, then lagged 1 day.

    Slow (252-day) signals, so a single structural shift(1) enforces past-only trading;
    no skip-gap is needed (that exists only for the 5-day reversal sleeve's bid-ask bounce).
    """
    zs = [_xs_standardize(raw, sector_map, config.winsorize_pct) for raw in factor_raws.values()]
    idx, cols = zs[0].index, zs[0].columns
    stack = np.stack([z.reindex(index=idx, columns=cols).values for z in zs])  # (F, T, N)
    comp = np.nanmean(stack, axis=0)
    comp = pd.DataFrame(comp, index=idx, columns=cols)
    mu = comp.mean(axis=1)
    sd = comp.std(axis=1).replace(0.0, np.nan)
    comp = comp.sub(mu, axis=0).div(sd, axis=0)
    return comp.shift(1)


# ---------------------------------------------------------------------------
# Portfolio: transparent rank-and-inverse-vol long/short
# ---------------------------------------------------------------------------
def multifactor_weights(alpha: pd.DataFrame, idio_vol: pd.DataFrame,
                        sector_map: dict[str, str], config: Config,
                        gross: float | None = None, decile: float = 0.1) -> pd.DataFrame:
    """Decile long/short with inverse-idio-vol risk weighting WITHIN each leg.

    Long the top `decile` of the composite, short the bottom `decile`; size names
    inverse to trailing idio-vol so each contributes ~equal risk, normalize each leg
    to gross/2 (dollar-neutral), then sector-neutralize the residual tilt. This avoids
    the lopsided over-concentration of scaling the alpha by inverse vol directly.
    """
    gross = gross if gross is not None else config.gross_cap
    inv = 1.0 / idio_vol.replace(0.0, np.nan)
    rank = alpha.rank(axis=1, pct=True)
    long_w = inv.where(rank >= (1.0 - decile)).fillna(0.0)
    short_w = inv.where(rank <= decile).fillna(0.0)
    long_w = long_w.div(long_w.sum(axis=1).replace(0.0, np.nan), axis=0) * (gross / 2.0)
    short_w = short_w.div(short_w.sum(axis=1).replace(0.0, np.nan), axis=0) * (gross / 2.0)
    w = (long_w - short_w)

    # Sector-demeaning spreads small offsetting weights across all names, so a single
    # demean-then-cap leaves the book slightly net-long once the cap clips one side
    # harder. Iterate (sector-demean -> dollar-demean -> cap) to convergence so the book
    # is dollar- and sector-neutral AND respects the per-name cap simultaneously.
    sectors = pd.Series({c: sector_map.get(c, "Other") for c in w.columns})
    sector_cols = {s: sectors.index[sectors == s] for s in sectors.unique()}
    for _ in range(5):
        for cols in sector_cols.values():
            if len(cols):
                w[cols] = w[cols].sub(w[cols].mean(axis=1), axis=0)
        w = w.sub(w.mean(axis=1), axis=0)
        w = w.clip(lower=-config.max_w, upper=config.max_w)
    return w.fillna(0.0)


def backtest_multifactor(w_base: pd.DataFrame, returns: pd.DataFrame, spread: pd.DataFrame,
                         adv: pd.DataFrame, vol: pd.DataFrame, config: Config,
                         aum: float = 25e6, rebal_gap: int = REBAL_GAP) -> dict:
    """Monthly-rebalanced (held constant between) net-of-cost backtest of the sleeve.

    `w_base` derives from the already-lagged composite alpha (alpha = composite.shift(1)),
    so the held weights are used directly against same-day returns (single effective lag,
    no look-ahead): position at t uses only data through t-1.
    """
    rebal_mask = pd.Series(np.arange(len(w_base)) % rebal_gap == 0, index=w_base.index)
    w_held = w_base.where(rebal_mask).ffill().fillna(0.0)

    gross = (w_held * returns).sum(axis=1)
    cost_bps = build_cost_panel(w_held, spread, adv, vol, config, aum=aum)
    cost = (cost_bps / 1e4).reindex(gross.index).fillna(0.0)
    net = gross - cost
    turnover = (w_held - w_held.shift(1)).abs().sum(axis=1)
    return {
        "gross": gross.dropna(),
        "net": net.dropna(),
        "cost_bps": cost_bps.reindex(gross.index).fillna(0.0),
        "turnover": turnover.dropna(),
        "weights": w_held,
    }


# ---------------------------------------------------------------------------
# Data assembly (mirrors run.run's first half; no cvxpy, deterministic)
# ---------------------------------------------------------------------------
def assemble_inputs(config: Config, tickers: list[str]) -> dict:
    logger.info("multifactor: assembling inputs for %d tickers", len(tickers))
    prices = pull_prices(tickers, config.start_date, config.end_date, config)
    factors = get_ff_factors(config)
    adv = compute_adv(prices, window=config.adv_window)
    membership = get_liquid_universe(adv, config.universe_size, config.hysteresis_buffer)
    returns = build_return_panel(prices, membership)
    sector_map = get_sector_map(tickers, config)
    resid, betas, idio_var = rolling_residuals(returns, factors, config)
    spread = corwin_schultz_spread(prices, config.cs_smooth_window)
    vol = compute_realized_vol(returns)
    close = pd.DataFrame({t: prices[t]["Close"] for t in prices})
    close = close.reindex(columns=returns.columns)
    mask = membership.reindex(index=close.index, columns=close.columns).fillna(False)
    close = close.where(mask)
    return dict(returns=returns, sector_map=sector_map, resid=resid, betas=betas,
                idio_var=idio_var, spread=spread, adv=adv, vol=vol, close=close)


def multifactor_sleeve(config: Config, tickers: list[str], factor2: str = "idiovol",
                       use_52w: bool = False, aum: float = 25e6,
                       rebal_gap: int = REBAL_GAP, inputs: dict | None = None) -> dict:
    """Build the multi-factor sleeve. Returns dict with daily gross/net/cost/turnover + diagnostics.

    factor2: 'idiovol' (default) or 'bab'. use_52w: add the optional third factor.
    Pass `inputs` to reuse a previously-assembled data dict (avoids recompute).
    """
    d = inputs if inputs is not None else assemble_inputs(config, tickers)
    raws: dict[str, pd.DataFrame] = {"resid_mom": residual_momentum(d["resid"])}
    if factor2 == "idiovol":
        raws["low_risk"] = low_risk_idiovol(d["idio_var"])
    elif factor2 == "bab":
        raws["low_risk"] = low_risk_bab(d["betas"]["Mkt-RF"])
    else:
        raise ValueError(f"unknown factor2: {factor2}")
    if use_52w:
        raws["high_52w"] = high_52w(d["close"])

    alpha = build_composite_alpha(raws, d["sector_map"], config)
    idio_vol = compute_idio_vol(d["idio_var"])
    w = multifactor_weights(alpha, idio_vol, d["sector_map"], config)
    bt = backtest_multifactor(w, d["returns"], d["spread"], d["adv"], d["vol"],
                              config, aum=aum, rebal_gap=rebal_gap)
    bt["alpha"] = alpha
    bt["factor_raws"] = raws
    bt["inputs"] = d
    bt["config"] = {"factor2": factor2, "use_52w": use_52w, "rebal_gap": rebal_gap,
                    "mom_lookback": MOM_LOOKBACK, "mom_skip": MOM_SKIP}
    logger.info("multifactor: built sleeve (factor2=%s, use_52w=%s)", factor2, use_52w)
    return bt
