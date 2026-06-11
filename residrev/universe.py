"""Point-in-time liquid-universe construction with Russell-style hysteresis."""

from __future__ import annotations

import logging

import pandas as pd

from residrev.config import Config

logger = logging.getLogger(__name__)

_MIN_UNIVERSE_WARN = 800


def compute_adv(prices: dict[str, pd.DataFrame], window: int = 63) -> pd.DataFrame:
    """Compute trailing average dollar volume for each ticker.

    Returns a wide DataFrame (index=dates, columns=tickers). Tickers with no
    data produce all-NaN columns. Date index is the union of all tickers' dates.
    """
    series: dict[str, pd.Series] = {}
    for ticker, df in prices.items():
        if df is None or df.empty:
            series[ticker] = pd.Series(dtype=float)
        else:
            dv = df["Close"] * df["Volume"]
            series[ticker] = dv.rolling(window, min_periods=40).mean()

    if not series:
        return pd.DataFrame()

    return pd.DataFrame(series)


def get_liquid_universe(
    adv: pd.DataFrame,
    universe_size: int = 1000,
    buffer: int = 200,
) -> pd.DataFrame:
    """Build a boolean membership panel with Russell-style hysteresis.

    Entry threshold: rank <= universe_size.
    Exit threshold:  rank >  universe_size + buffer.
    Tickers with NaN ADV are never members.
    """
    threshold = universe_size + buffer
    membership = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    prev = pd.Series(False, index=adv.columns)

    for i, date in enumerate(adv.index):
        row = adv.loc[date]
        rank = row.rank(ascending=False, na_option="keep")

        if i == 0:
            current = (rank <= universe_size).fillna(False)
        else:
            enters = (~prev) & (rank <= universe_size).fillna(False)
            stays = prev & (rank <= threshold).fillna(False)
            current = enters | stays

        membership.loc[date] = current
        prev = current

        size = int(current.sum())
        if size < _MIN_UNIVERSE_WARN:
            logger.warning("Universe size %d below %d on %s", size, _MIN_UNIVERSE_WARN, date)

    return membership.astype(bool)


def get_universe_size_over_time(membership: pd.DataFrame) -> pd.Series:
    """Return the count of universe members per date."""
    return membership.sum(axis=1)
