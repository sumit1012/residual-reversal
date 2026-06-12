"""
Transaction cost model: Corwin-Schultz bid-ask spread + Almgren sqrt market impact.

Post-hoc accounting used by backtest.py to compute net-of-cost PnL.
NOT used inside the cvxpy optimizer (that uses a turnover penalty in the objective).
"""
import logging

import numpy as np
import pandas as pd

from residrev.config import Config

logger = logging.getLogger(__name__)


def corwin_schultz_spread(
    prices: dict[str, pd.DataFrame], window: int = 21
) -> pd.DataFrame:
    """
    Estimates bid-ask half-spread from daily High/Low prices (Corwin-Schultz 2012).

    Past-only — no look-ahead. Returns smoothed half-spread as a decimal
    (e.g. 0.0010 = 10 bps), not a percentage.

    Tickers missing High or Low columns receive NaN for all dates.
    """
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    spreads: dict[str, pd.Series] = {}

    for ticker, df in prices.items():
        if "High" not in df.columns or "Low" not in df.columns:
            spreads[ticker] = pd.Series(np.nan, index=df.index)
            continue

        high = df["High"]
        low = df["Low"]

        log_hl = np.log(high / low)
        log_hl_prev = log_hl.shift(1)

        beta = log_hl_prev**2 + log_hl**2

        rolling_high = high.rolling(2).max()
        rolling_low = low.rolling(2).min()
        gamma = np.log(rolling_high / rolling_low) ** 2

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

        raw_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        raw_spread = np.clip(raw_spread, 0.0, None)

        spreads[ticker] = raw_spread.rolling(window, min_periods=5).median()

    result = pd.DataFrame(spreads)

    mean_s = result.stack().mean()
    med_s = result.stack().median()
    logger.info(
        "Corwin-Schultz spread — universe mean: %.4f (%.1f bps), median: %.4f (%.1f bps)",
        mean_s,
        mean_s * 1e4,
        med_s,
        med_s * 1e4,
    )

    return result


def compute_realized_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Trailing realized annualized volatility from daily returns. Past-only.

    vol[t, n] = std(returns[t-window : t-1, n]) * sqrt(252), min_periods=10.
    """
    return returns.shift(1).rolling(window, min_periods=10).std() * np.sqrt(252)


def compute_rebalance_cost(
    w_prev: pd.Series,
    w_new: pd.Series,
    spread: pd.Series,
    adv: pd.Series,
    vol: pd.Series,
    config: Config,
    aum: float = 1e8,
) -> float:
    """
    Total one-way transaction cost for a single rebalance, in basis points.

    Participation rate per Almgren et al. (2005):
        participation_n = (turnover_n * aum) / adv[n]
    The adv_participation_cap is a constraint checked for warnings, not the denominator.

    Parameters
    ----------
    w_prev, w_new : weights before/after rebalancing (indexed by ticker)
    spread        : Corwin-Schultz half-spread per ticker (decimal)
    adv           : average daily dollar volume per ticker
    vol           : annualized realized volatility per ticker (decimal)
    config        : Config object (uses eta_impact, adv_participation_cap)
    aum           : portfolio AUM in dollars (same units as adv)
    """
    tickers = (
        w_prev.index
        .intersection(w_new.index)
        .intersection(spread.index)
        .intersection(adv.index)
        .intersection(vol.index)
    )

    w_prev = w_prev.reindex(tickers).fillna(0.0)
    w_new = w_new.reindex(tickers).fillna(0.0)
    spread = spread.reindex(tickers).fillna(0.0)
    adv = adv.reindex(tickers).replace(0.0, np.nan).fillna(1.0)
    vol = vol.reindex(tickers).fillna(0.0)

    turnover = (w_new - w_prev).abs()

    # Spread cost: half-spread paid on each unit of turnover
    spread_cost = spread * turnover

    # Market impact: Almgren-style sqrt-impact
    # participation = dollar_traded / ADV = (weight_turnover * AUM) / ADV
    raw_participation = (turnover * aum) / adv

    over_cap = raw_participation[raw_participation > config.adv_participation_cap]
    if not over_cap.empty:
        logger.warning(
            "Participation cap (%.0f%% ADV) exceeded for %d stocks: %s",
            config.adv_participation_cap * 100,
            len(over_cap),
            over_cap.index.tolist()[:5],
        )

    participation = np.clip(raw_participation, 0.0, 1.0)
    impact_cost = config.eta_impact * vol * np.sqrt(participation) * turnover

    total_cost = float((spread_cost + impact_cost).sum())
    return total_cost * 10_000.0


def build_cost_panel(
    weights: pd.DataFrame,
    spread: pd.DataFrame,
    adv: pd.DataFrame,
    vol: pd.DataFrame,
    config: Config,
    aum: float = 1e8,
) -> pd.Series:
    """
    Applies compute_rebalance_cost across all rebalance dates.

    Returns a Series indexed by date, values in basis points.
    Turnover on the first date equals |weights[0]| (entering from flat).
    """
    costs: dict = {}
    dates = weights.index

    for i, date in enumerate(dates):
        w_new = weights.loc[date]
        w_prev = weights.iloc[i - 1] if i > 0 else pd.Series(0.0, index=weights.columns)

        def _row(panel: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
            return panel.loc[d] if d in panel.index else pd.Series(dtype=float)

        costs[date] = compute_rebalance_cost(
            w_prev=w_prev,
            w_new=w_new,
            spread=_row(spread, date),
            adv=_row(adv, date),
            vol=_row(vol, date),
            config=config,
            aum=aum,
        )

    panel = pd.Series(costs, name="cost_bps")
    logger.info(
        "Cost panel: mean=%.2f bps, median=%.2f bps over %d dates",
        panel.mean(),
        panel.median(),
        len(panel),
    )
    return panel
