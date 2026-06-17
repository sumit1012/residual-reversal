"""Cross-asset time-series trend-following sleeve.

A diversified, vol-targeted trend follower across liquid ETFs spanning equities,
bonds, commodities, and the US dollar. Trend (time-series momentum) is one of the
most durable, longest-documented systematic premia (Moskowitz-Ooi-Pedersen 2012;
Hurst-Ooi-Pedersen "A Century of Evidence on Trend-Following", 2017). It is the
economic complement to short-horizon reversal: reversal fades moves, trend rides
them, so the two diversify across regimes.

Mechanism / who is on the other side: trend harvests the slow diffusion of
information and the demand of hedgers/forced rebalancers who trade against the
trend (risk transfer). It earns most in persistent, trending regimes and bleeds in
choppy, mean-reverting ones, which is exactly when the reversal sleeve earns.

All signals are strictly past-only (lagged), so there is no look-ahead.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Liquid, long-history ETFs spanning four asset classes (free via yfinance).
TREND_UNIVERSE: dict[str, str] = {
    "SPY": "US large-cap equity",
    "QQQ": "US tech equity",
    "IWM": "US small-cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "TLT": "US 20y+ Treasury",
    "IEF": "US 7-10y Treasury",
    "LQD": "US investment-grade credit",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "USO": "Crude oil",
    "UUP": "US dollar index",
}

TREND_LOOKBACKS = (63, 126, 252)  # ~3, 6, 12 trading months; blended for robustness
VOL_WINDOW = 63                   # trailing window for inverse-vol sizing
TARGET_VOL = 0.10                 # 10% annualized portfolio vol target
REBAL_GAP = 5                     # rebalance every 5 trading days (weekly) to cut turnover
ETF_COST_BPS = 2.0                # round-trip cost per unit turnover (liquid ETFs ~1-3 bps)


def fetch_trend_prices(
    tickers: list[str] | None = None,
    start: str = "2009-01-01",
    end: str | None = None,
    cache_dir: str = "cache/trend",
) -> pd.DataFrame:
    """Adjusted daily closes for the ETF universe, cached to parquet. Past-only by date."""
    import yfinance as yf

    tickers = tickers or list(TREND_UNIVERSE)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "etf_closes.parquet")
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        have = [t for t in tickers if t in cached.columns]
        if len(have) == len(tickers) and (end is None or cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)):
            logger.info("Loaded trend prices from cache: %s", cache_path)
            return cached[tickers]

    logger.info("Downloading %d ETFs from yfinance", len(tickers))
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all").ffill(limit=2)
    raw.to_parquet(cache_path)
    return raw[[t for t in tickers if t in raw.columns]]


def trend_signal(prices: pd.DataFrame, lookbacks=TREND_LOOKBACKS) -> pd.DataFrame:
    """Blended time-series-momentum signal in [-1, 1] per instrument, lagged one day.

    For each lookback L, sign(P_t / P_{t-L} - 1); averaged across lookbacks, then
    shifted one day so the position at t uses only information through t-1.
    """
    sigs = []
    for L in lookbacks:
        sigs.append(np.sign(prices / prices.shift(L) - 1.0))
    blended = sum(sigs) / len(sigs)
    return blended.shift(1)


def backtest_trend(
    prices: pd.DataFrame,
    lookbacks=TREND_LOOKBACKS,
    vol_window: int = VOL_WINDOW,
    target_vol: float = TARGET_VOL,
    rebal_gap: int = REBAL_GAP,
    cost_bps: float = ETF_COST_BPS,
) -> dict:
    """Vol-targeted trend backtest. Returns dict with daily gross/net return series + weights.

    Weights: position_i = signal_i * (target_vol / n) / inst_vol_i (inverse-vol sizing),
    then the whole book is scaled so its ex-ante vol equals target_vol. Rebalanced every
    `rebal_gap` days; held constant in between (reduces turnover/costs).
    """
    rets = np.log(prices / prices.shift(1))
    sig = trend_signal(prices, lookbacks)
    inst_vol = rets.rolling(vol_window).std().shift(1) * np.sqrt(252)
    inst_vol = inst_vol.replace(0.0, np.nan)

    n = prices.shape[1]
    raw_w = sig * (target_vol / n) / inst_vol  # per-instrument risk-scaled position

    # scale whole book to target vol using trailing covariance of the raw-weighted book
    raw_book_ret = (raw_w.shift(1) * rets).sum(axis=1)
    book_vol = raw_book_ret.rolling(vol_window).std().shift(1) * np.sqrt(252)
    scale = (target_vol / book_vol).clip(upper=3.0).fillna(0.0)
    w = raw_w.mul(scale, axis=0)

    # rebalance every rebal_gap days: hold weights constant between rebalances
    rebal_mask = pd.Series(np.arange(len(w)) % rebal_gap == 0, index=w.index)
    w_held = w.where(rebal_mask).ffill().fillna(0.0)

    gross = (w_held.shift(1) * rets).sum(axis=1)
    turnover = (w_held - w_held.shift(1)).abs().sum(axis=1)
    cost = turnover * cost_bps / 1e4
    net = gross - cost

    return {
        "gross": gross.dropna(),
        "net": net.dropna(),
        "weights": w_held,
        "turnover": turnover.dropna(),
        "gross_leverage": w_held.abs().sum(axis=1),
    }
