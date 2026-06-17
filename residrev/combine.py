"""Multi-sleeve portfolio combination (WSQ Module 4: combining strategies).

Combines independent strategy sleeves (here: idiosyncratic residual reversal and
cross-asset trend) into one book. The thesis: two individually-fragile, low-
correlation return streams combine into a more robust book, lower drawdown and a
higher risk-adjusted return than either alone, because their regime exposures are
opposite (reversal earns in choppy/mean-reverting regimes, trend in persistent ones).

All sleeve weights use only trailing (past-only) information, so the combination
introduces no look-ahead beyond what each sleeve already respects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def _max_drawdown(r: pd.Series) -> float:
    cum = (1 + r.dropna()).cumprod()
    return float((cum / cum.cummax() - 1).min()) if len(cum) else float("nan")


def align_sleeves(sleeves: dict[str, pd.Series]) -> pd.DataFrame:
    """Align sleeve daily-return series on their common dates."""
    df = pd.DataFrame({k: v for k, v in sleeves.items()})
    return df.dropna(how="any")


def sleeve_weights(
    panel: pd.DataFrame,
    scheme: str = "risk_parity",
    vol_window: int = 63,
) -> pd.DataFrame:
    """Daily sleeve weights (sum to 1), past-only.

    scheme:
      'equal'        equal capital weight
      'risk_parity'  inverse trailing-vol (equal risk contribution)
    """
    if scheme == "equal":
        w = pd.DataFrame(1.0 / panel.shape[1], index=panel.index, columns=panel.columns)
    elif scheme == "risk_parity":
        inv_vol = 1.0 / (panel.rolling(vol_window).std().shift(1))
        w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return w.fillna(0.0)


def combine(
    sleeves: dict[str, pd.Series],
    scheme: str = "risk_parity",
    target_vol: float | None = 0.10,
    vol_window: int = 63,
) -> dict:
    """Combine sleeves into one book. Optionally scale the book to a target annual vol.

    Returns dict with the combined daily return series, the sleeve weights, and metrics.
    """
    panel = align_sleeves(sleeves)
    w = sleeve_weights(panel, scheme=scheme, vol_window=vol_window)
    combined = (w * panel).sum(axis=1)

    leverage = pd.Series(1.0, index=combined.index)
    if target_vol is not None:
        realized = combined.rolling(vol_window).std().shift(1) * np.sqrt(TRADING_DAYS)
        leverage = (target_vol / realized).clip(upper=3.0).fillna(0.0)
        combined = combined * leverage

    combined = combined.dropna()
    return {
        "combined": combined,
        "sleeve_weights": w,
        "leverage": leverage,
        "metrics": {
            "sharpe": _ann_sharpe(combined),
            "ann_return": float(combined.mean() * TRADING_DAYS),
            "ann_vol": float(combined.std() * np.sqrt(TRADING_DAYS)),
            "max_drawdown": _max_drawdown(combined),
        },
    }


def diversification_report(sleeves: dict[str, pd.Series], freeze: str | pd.Timestamp) -> pd.DataFrame:
    """Per-sleeve and combined (across weighting schemes) metrics, split in-sample vs live.

    `freeze` is the train/live boundary; rows after it are out-of-sample.
    """
    panel = align_sleeves(sleeves)
    freeze = pd.Timestamp(freeze)
    rows: list[dict] = []

    def add(name: str, r: pd.Series):
        ins = r[r.index < freeze]
        live = r[r.index >= freeze]
        rows.append({
            "book": name,
            "full_sharpe": round(_ann_sharpe(r), 2),
            "insample_sharpe": round(_ann_sharpe(ins), 2),
            "live_sharpe": round(_ann_sharpe(live), 2),
            "live_return_pct": round((float((1 + live).prod()) - 1) * 100, 1),
            "ann_vol_pct": round(float(r.std() * np.sqrt(TRADING_DAYS)) * 100, 1),
            "max_dd_pct": round(_max_drawdown(r) * 100, 1),
        })

    for c in panel.columns:
        add(c, panel[c])
    for scheme in ("equal", "risk_parity"):
        out = combine(sleeves, scheme=scheme, target_vol=None)
        add(f"combined_{scheme}", out["combined"])

    corr = panel.corr()
    df = pd.DataFrame(rows).set_index("book")
    df.attrs["correlation"] = corr
    return df
