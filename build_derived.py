"""Derived analytics for the dashboard, computed from the COMMITTED report.json.

This intentionally does NOT re-run the strategy. The reversal optimizer is
solver-sensitive near its turnover cliff, so the backtest block is frozen
(computed once, committed). Re-running it would change the in-sample numbers.
Instead we derive the richer tear-sheet analytics (drawdown, per-year and
monthly returns, rolling Sharpe, return distribution, extended risk metrics,
rolling inter-sleeve correlation, and a market-beta / factor-exposure proof)
directly from the frozen equity curves already in report.json.

Resolution note: the stored curves are downsampled (every 5th trading day in the
backtest, every 2nd in the live block), so curve-derived stats are weekly-ish,
not daily. The headline numbers (Sharpe, ann. return, ann. vol, max DD) are NOT
recomputed here; they are read straight from the frozen block. Only ADDITIONAL
analytics are added, and the downsampling is footnoted on the site.

Usage:
    python build_derived.py [path/to/report.json]   # default: site/public/data/report.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "REPORT_OUT", "site/public/data/report.json")
BOOKS = ("reversal", "trend", "combined")


def _curve_to_nav(curve):
    """[[date, nav], ...] -> pd.Series(nav, index=DatetimeIndex), de-duplicated."""
    if not curve:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in curve}).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _period_returns(nav):
    """Simple returns between consecutive stored points + their day-gaps."""
    r = nav.pct_change().dropna()
    dt = nav.index.to_series().diff().dt.days.dropna()
    dt = dt[dt > 0]
    r = r.loc[dt.index]
    return r, dt


def _ann_factor(dt):
    avg = float(dt.mean()) if len(dt) else 7.0
    return np.sqrt(365.25 / max(avg, 1.0))


def _drawdown(nav, n=90):
    dd = (nav / nav.cummax() - 1.0) * 100.0
    if len(dd) > n:
        step = len(dd) / n
        idx = sorted({int(i * step) for i in range(n)} | {len(dd) - 1})
        dd = dd.iloc[idx]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in dd.items()]


def _years(nav):
    """Calendar-year total return from year-boundary NAV."""
    yends = nav.resample("YE").last()
    out = []
    prev = None
    start_val = nav.iloc[0]
    for ts, v in yends.items():
        base = prev if prev is not None else start_val
        if base and np.isfinite(base):
            out.append([int(ts.year), round((float(v) / float(base) - 1.0) * 100.0, 1)])
        prev = v
    return out


def _monthly(nav):
    m = nav.resample("ME").last()
    mr = m.pct_change().dropna() * 100.0
    return [[ts.strftime("%Y-%m"), round(float(v), 2)] for ts, v in mr.items()]


def _monthly_returns_series(nav):
    m = nav.resample("ME").last()
    return m.pct_change().dropna()


def _rolling_sharpe(nav, win=12, n=80):
    mr = _monthly_returns_series(nav)
    if len(mr) < win + 1:
        return []
    rs = (mr.rolling(win).mean() / mr.rolling(win).std()) * np.sqrt(12.0)
    rs = rs.dropna()
    if len(rs) > n:
        step = len(rs) / n
        idx = sorted({int(i * step) for i in range(n)} | {len(rs) - 1})
        rs = rs.iloc[idx]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in rs.items()]


def _distribution(r, bins=21):
    r = r.dropna() * 100.0
    if len(r) < 5:
        return None
    lo, hi = float(r.min()), float(r.max())
    counts, edges = np.histogram(r, bins=bins, range=(lo, hi))
    return {
        "bins": [[round(float(edges[i]), 3), int(counts[i])] for i in range(len(counts))],
        "mean_pct": round(float(r.mean()), 3),
    }


def _metrics(nav, stored):
    r, dt = _period_returns(nav)
    af = _ann_factor(dt)
    mr = _monthly_returns_series(nav)
    downside = r[r < 0]
    sortino = (float(r.mean()) / float(downside.std()) * af) if len(downside) > 2 and downside.std() else None
    ann_ret = stored.get("ann_return_pct")
    if ann_ret is None and stored.get("return_pct") is not None:
        yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
        ann_ret = ((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1.0) * 100.0
    maxdd = stored.get("max_dd_pct")
    calmar = (ann_ret / abs(maxdd)) if (ann_ret is not None and maxdd) else None
    var95 = float(np.percentile(r, 5)) * 100.0 if len(r) > 10 else None
    cvar95 = float(r[r <= np.percentile(r, 5)].mean()) * 100.0 if len(r) > 10 else None
    return {
        "sharpe": stored.get("sharpe"),
        "ann_return_pct": round(ann_ret, 2) if ann_ret is not None else None,
        "ann_vol_pct": stored.get("ann_vol_pct"),
        "max_dd_pct": maxdd,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "calmar": round(calmar, 2) if calmar is not None else None,
        "hit_rate_pct": round(float((r > 0).mean()) * 100.0, 1) if len(r) else None,
        "best_month_pct": round(float(mr.max()) * 100.0, 1) if len(mr) else None,
        "worst_month_pct": round(float(mr.min()) * 100.0, 1) if len(mr) else None,
        "skew": round(float(r.skew()), 2) if len(r) > 3 else None,
        "kurtosis": round(float(r.kurtosis()), 2) if len(r) > 3 else None,
        "var95_pct": round(var95, 2) if var95 is not None else None,
        "cvar95_pct": round(cvar95, 2) if cvar95 is not None else None,
        "n_periods": int(len(r)),
    }


def _ols(y, X):
    """OLS with intercept. Returns (betas dict-less arrays, tstats, r2). X: (n,k) no intercept col."""
    n, k = X.shape
    Xi = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    resid = y - Xi @ beta
    dof = max(n - k - 1, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(Xi.T @ Xi)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return beta, tstat, r2


def _align_factor_to_dates(daily_cum, dates):
    """Period factor return between consecutive `dates` from a daily cumulative-growth series."""
    c = daily_cum.reindex(daily_cum.index.union(dates)).sort_index().ffill().reindex(dates)
    return c.pct_change()


def market_beta(report):
    """Regress each book's in-sample returns on the in-sample SPX (no network)."""
    spx = _curve_to_nav(report["backtest"].get("spx_curve", []))
    if spx.empty:
        return None
    spx_r, _ = _period_returns(spx)
    out = {}
    for b in BOOKS:
        nav = _curve_to_nav(report["backtest"]["books"][b]["curve"])
        r, _ = _period_returns(nav)
        idx = r.index.intersection(spx_r.index)
        if len(idx) < 30:
            continue
        beta, t, r2 = _ols(r.loc[idx].values, spx_r.loc[idx].values.reshape(-1, 1))
        out[b] = {"beta": round(float(beta[1]), 3), "tstat": round(float(t[1]), 2),
                  "r2": round(float(r2), 3), "n": int(len(idx))}
    return out


def factor_exposures(report):
    """FF5 + momentum OLS on each book's in-sample returns. Network; returns None on failure."""
    try:
        import pandas_datareader.data as web
        start, end = "2018-01-01", report["backtest"].get("end", "2024-12-31")
        ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0] / 100.0
        mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0] / 100.0
        ff.index = pd.DatetimeIndex(ff.index)
        mom.index = pd.DatetimeIndex(mom.index)
        mom.columns = ["Mom"]
        fac = ff.join(mom, how="inner").dropna()
        rf = fac["RF"]
        names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
        cum = {c: (1.0 + fac[c]).cumprod() for c in names}
        rf_cum = (1.0 + rf).cumprod()
    except Exception as e:  # noqa: BLE001
        print(f"[factor_exposures] skipped (no factor data): {e}", file=sys.stderr)
        return None

    out = {}
    for b in BOOKS:
        nav = _curve_to_nav(report["backtest"]["books"][b]["curve"])
        r, _ = _period_returns(nav)
        dates = nav.index
        fr = {c: _align_factor_to_dates(cum[c], dates).reindex(r.index) for c in names}
        rfp = _align_factor_to_dates(rf_cum, dates).reindex(r.index)
        df = pd.DataFrame(fr)
        df["y"] = r - rfp
        df = df.dropna()
        if len(df) < 40:
            continue
        beta, t, r2 = _ols(df["y"].values, df[names].values)
        out[b] = {
            "alpha_ann_pct": round(float(beta[0]) * 52.0 * 100.0, 2),
            "loadings": {names[i]: {"beta": round(float(beta[i + 1]), 3),
                                     "tstat": round(float(t[i + 1]), 2)} for i in range(len(names))},
            "r2": round(float(r2), 3), "n": int(len(df)),
        }
    return out


def hero_series(report, start="2018-06-01", n=120):
    """Growth of 100 stitched across the frozen backtest + live blocks, per book.
    Each book is rebased to 100 at `start`; the live block continues from the
    backtest's ending value. Used for the home hero chart."""
    out = {}
    for b in BOOKS:
        bt = _curve_to_nav(report["backtest"]["books"][b]["curve"])
        bt = bt[bt.index >= pd.Timestamp(start)]
        if bt.empty:
            continue
        bt = bt / bt.iloc[0] * 100.0
        lv = _curve_to_nav(report["live"]["books"][b]["curve"])
        if not lv.empty:
            lv = lv / lv.iloc[0] * float(bt.iloc[-1])
        full = pd.concat([bt, lv])
        full = full[~full.index.duplicated(keep="last")].sort_index()
        if len(full) > n:
            step = len(full) / n
            idx = sorted({int(i * step) for i in range(n)} | {len(full) - 1})
            full = full.iloc[idx]
        out[b] = [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in full.items()]
    return out


def rolling_correlation(report, win=12, n=80):
    rev = _monthly_returns_series(_curve_to_nav(report["backtest"]["books"]["reversal"]["curve"]))
    trd = _monthly_returns_series(_curve_to_nav(report["backtest"]["books"]["trend"]["curve"]))
    df = pd.concat([rev.rename("r"), trd.rename("t")], axis=1).dropna()
    if len(df) < win + 2:
        return []
    rc = df["r"].rolling(win).corr(df["t"]).dropna()
    if len(rc) > n:
        step = len(rc) / n
        idx = sorted({int(i * step) for i in range(n)} | {len(rc) - 1})
        rc = rc.iloc[idx]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in rc.items()]


def main():
    report = json.load(open(OUT))
    derived = {
        "backtest": {}, "live": {},
        "footnote": ("Curve-derived analytics use the stored equity curves, which are "
                     "downsampled (every 5th trading day in the backtest, every 2nd live). "
                     "Headline Sharpe, annual return, annual volatility and max drawdown are "
                     "the frozen daily figures; distribution, rolling and per-period stats are "
                     "computed at the stored (weekly-ish) resolution."),
    }
    for block in ("backtest", "live"):
        for b in BOOKS:
            book = report[block]["books"][b]
            nav = _curve_to_nav(book["curve"])
            if nav.empty:
                continue
            derived[block][b] = {
                "drawdown": _drawdown(nav),
                "years": _years(nav),
                "monthly": _monthly(nav),
                "rolling_sharpe": _rolling_sharpe(nav),
                "distribution": _distribution(_period_returns(nav)[0]),
                "metrics": _metrics(nav, book),
            }
    derived["rolling_corr"] = rolling_correlation(report)
    derived["market_beta"] = market_beta(report)
    derived["factors"] = factor_exposures(report)
    derived["hero"] = hero_series(report)

    report["derived"] = derived
    json.dump(report, open(OUT, "w"), indent=2)

    mb = derived["market_beta"] or {}
    print(json.dumps({
        "wrote": OUT,
        "blocks": {bl: list(derived[bl].keys()) for bl in ("backtest", "live")},
        "market_beta_reversal": mb.get("reversal"),
        "factors": "ok" if derived["factors"] else "skipped",
        "rolling_corr_pts": len(derived["rolling_corr"]),
    }, indent=2))


if __name__ == "__main__":
    main()
