"""Build a holdings snapshot (current book) from sleeve positions, for the dashboard.

Turns the latest position weights into a portfolio view: longs/shorts in dollars at a
notional AUM, sector / asset-class exposure, and each holding's recent return (a
yfinance-style grid). The reversal sleeve is market-neutral (longs ~= shorts), so its
"allocation" is shown as GROSS exposure by sector plus a near-zero NET exposure that
demonstrates the neutrality; the trend sleeve is directional, so its allocation by
asset class is meaningful. Hypothetical / simulated, not a real-money account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Clean asset-class buckets for the 12 trend ETFs.
TREND_ASSET_CLASS: dict[str, str] = {
    "SPY": "Equities", "QQQ": "Equities", "IWM": "Equities", "EFA": "Equities", "EEM": "Equities",
    "TLT": "Bonds", "IEF": "Bonds", "LQD": "Credit",
    "GLD": "Commodities", "DBC": "Commodities", "USO": "Commodities",
    "UUP": "US dollar",
}

_MIN_W = 1e-5


def _row_to_dict(ticker, w, group, recent, aum):
    return {
        "ticker": str(ticker),
        "group": group,
        "side": "long" if w >= 0 else "short",
        "weight_pct": round(float(w) * 100.0, 2),
        "dollar": int(round(float(w) * aum)),
        "recent_return_pct": (None if recent is None or not np.isfinite(recent)
                              else round(float(recent) * 100.0, 1)),
    }


def _agg_by_group(weights: pd.Series, groups: pd.Series, recent: pd.Series, aum: float):
    out = []
    for g in sorted(groups.dropna().unique()):
        members = groups.index[groups == g]
        w = weights.reindex(members).dropna()
        if w.empty:
            continue
        r = recent.reindex(members)
        out.append({
            "group": str(g),
            "gross_pct": round(float(w.abs().sum()) * 100.0, 1),
            "net_pct": round(float(w.sum()) * 100.0, 1),
            "gross_dollar": int(round(float(w.abs().sum()) * aum)),
            "avg_return_pct": (None if r.dropna().empty else round(float(r.dropna().mean()) * 100.0, 1)),
            "n": int((w.abs() > _MIN_W).sum()),
        })
    return sorted(out, key=lambda d: d["gross_pct"], reverse=True)


def _book_summary(weights: pd.Series, aum: float) -> dict:
    w = weights.dropna()
    gl = float(w[w > 0].sum())
    gs = float(-w[w < 0].sum())
    return {
        "gross_long_dollar": int(round(gl * aum)),
        "gross_short_dollar": int(round(gs * aum)),
        "net_dollar": int(round(float(w.sum()) * aum)),
        "gross_dollar": int(round((gl + gs) * aum)),
        "leverage": round((gl + gs), 2),
        "n_long": int((w > _MIN_W).sum()),
        "n_short": int((w < -_MIN_W).sum()),
    }


def reversal_holdings(pos_row: pd.Series, recent: pd.Series, sector_map: dict[str, str],
                      aum: float, top_n: int = 15) -> dict:
    w = pos_row.dropna()
    w = w[w.abs() > _MIN_W]
    groups = pd.Series({t: sector_map.get(t, "Other") for t in w.index})
    longs = w[w > 0].sort_values(ascending=False)
    shorts = w[w < 0].sort_values()
    top_long = [_row_to_dict(t, longs[t], groups[t], recent.get(t), aum) for t in longs.index[:top_n]]
    top_short = [_row_to_dict(t, shorts[t], groups[t], recent.get(t), aum) for t in shorts.index[:top_n]]
    return {
        "summary": _book_summary(w, aum),
        "top_long": top_long,
        "top_short": top_short,
        "sectors": _agg_by_group(w, groups, recent, aum),
    }


def trend_holdings(w_row: pd.Series, recent: pd.Series, aum: float) -> dict:
    w = w_row.dropna()
    w = w[w.abs() > _MIN_W]
    groups = pd.Series({t: TREND_ASSET_CLASS.get(t, "Other") for t in w.index})
    positions = [_row_to_dict(t, w[t], groups[t], recent.get(t), aum)
                 for t in w.reindex(w.abs().sort_values(ascending=False).index).index]
    return {
        "summary": _book_summary(w, aum),
        "positions": positions,
        "asset_class": _agg_by_group(w, groups, recent, aum),
    }


def build_holdings(rev_row, rev_recent, trend_row, trend_recent, sector_map,
                   aum: float, asof_rev: str, asof_trend: str, top_n: int = 15) -> dict:
    """Assemble one holdings snapshot for both sleeves."""
    return {
        "aum": int(aum),
        "as_of": {"reversal": asof_rev, "trend": asof_trend},
        "reversal": reversal_holdings(rev_row, rev_recent, sector_map, aum, top_n),
        "trend": trend_holdings(trend_row, trend_recent, aum),
    }
