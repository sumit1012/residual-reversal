"""Report builder for the two-sleeve book (trend-primary + reversal diversifier).

Emits site data with a FROZEN backtest block (<= 2024-12-31, computed once, committed)
and a LIVE block (>= 2025-01-01, refreshed daily by the GitHub Action via --live-only).
Freezing the backtest keeps the in-sample numbers stable run-to-run; only the live
out-of-sample track updates. Deterministic (single-thread BLAS + fixed universe).

Usage:
    python build_reports.py              # full build (recomputes backtest + live)
    python build_reports.py --live-only  # keep committed backtest block, refresh live only
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import dataclasses
import io
import json
import logging
import os.path
import sys

import numpy as np
import pandas as pd
import requests

from residrev import combine, trend
from residrev.config import Config
from residrev.run import run as run_reversal_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_reports")

FREEZE = "2025-01-01"        # train/live boundary; backtest is everything strictly before this
BACKTEST_END = "2024-12-31"  # backtest block covers up to here (frozen)
START = "2018-01-01"
END = "2026-04-30"           # Ken French factor-data limit (reversal sleeve)
TICKER_CACHE = "universe_tickers.txt"
OUT = os.environ.get("REPORT_OUT", "site/public/data/report.json")
TD = 252


def get_tickers_fixed() -> list[str]:
    if os.path.exists(TICKER_CACHE):
        with open(TICKER_CACHE) as f:
            return [t.strip() for t in f if t.strip()]
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (research-bot; educational use)"}, timeout=30,
    )
    resp.raise_for_status()
    tickers = (pd.read_html(io.StringIO(resp.text))[0]["Symbol"]
               .str.replace(".", "-", regex=False).tolist())
    tickers = sorted(set(t.upper() for t in tickers))
    with open(TICKER_CACHE, "w") as f:
        f.write("\n".join(tickers))
    return tickers


def reversal_sleeve() -> pd.Series:
    cfg = dataclasses.replace(Config(), start_date=START, end_date=END, aum=25e6)
    logger.info("Building reversal sleeve...")
    result, _ = run_reversal_pipeline(cfg, get_tickers_fixed(), eda_output_dir=None)
    r = result.pnl.dropna()
    r.index = pd.DatetimeIndex(r.index).tz_localize(None)
    return r


def trend_sleeve() -> pd.Series:
    logger.info("Building trend sleeve...")
    px = trend.fetch_trend_prices(start="2013-01-01", end=None)
    return trend.backtest_trend(px)["net"].dropna()


def spx_series() -> pd.Series:
    import yfinance as yf
    spy = yf.download("SPY", start="2015-01-01", end=None, progress=False, auto_adjust=True)["Close"]
    spy = spy.squeeze()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None)
    return spy.pct_change().dropna()


def _sharpe(r):
    r = r.dropna()
    return round(float(r.mean() / r.std() * np.sqrt(TD)), 3) if len(r) > 20 and r.std() else None


def _mdd(r):
    c = (1 + r.dropna()).cumprod()
    return round(float((c / c.cummax() - 1).min()) * 100, 1) if len(c) else None


def _curve(r, ds=5):
    c = (1 + r).cumprod()
    return [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in c.iloc[::ds].items()]


def _book_backtest(r):
    r = r.dropna()
    return {
        "sharpe": _sharpe(r),
        "ann_return_pct": round(float(r.mean() * TD) * 100, 2),
        "ann_vol_pct": round(float(r.std() * np.sqrt(TD)) * 100, 1),
        "max_dd_pct": _mdd(r),
        "curve": _curve(r),
    }


def _book_live(r):
    r = r.dropna()
    return {
        "sharpe": _sharpe(r),
        "return_pct": round((float((1 + r).prod()) - 1) * 100, 1) if len(r) else None,
        "max_dd_pct": _mdd(r),
        "curve": _curve(r, ds=2),
        "as_of": r.index.max().strftime("%Y-%m-%d") if len(r) else None,
    }


def compute():
    rev, trd = reversal_sleeve(), trend_sleeve()
    sleeves = {"reversal": rev, "trend": trd}
    comb = combine.combine(sleeves, scheme="risk_parity", target_vol=0.10)["combined"]
    panel = combine.align_sleeves(sleeves)
    spx = spx_series()

    fz = pd.Timestamp(FREEZE)
    series = {"reversal": panel["reversal"], "trend": trd, "combined": comb}
    corr = round(float(panel["reversal"].corr(panel["trend"])), 3)

    backtest = {"end": BACKTEST_END, "books": {}, "spx_curve": _curve(spx[spx.index < fz].dropna())}
    live = {"books": {}, "as_of": {}}
    for name, s in series.items():
        s = s.dropna()
        backtest["books"][name] = _book_backtest(s[s.index < fz])
        lv = _book_live(s[s.index >= fz])
        live["books"][name] = {k: lv[k] for k in ("sharpe", "return_pct", "max_dd_pct", "curve")}
        live["as_of"][name] = lv["as_of"]
    return backtest, live, corr, series


def main(live_only: bool):
    backtest, live, corr, series = compute()
    report = {
        "freeze_date": FREEZE,
        "generated_at": max(v for v in live["as_of"].values() if v),
        "correlation_reversal_trend": corr,
        "note": "Trend-primary two-sleeve book. Backtest (<=2024-12-31) is frozen; the live "
                "block (2025+) refreshes daily. Reversal is a near-uncorrelated diversifier, "
                "marginal on its own; trend is the return engine.",
        "live": live,
    }
    if live_only and os.path.exists(OUT):
        prev = json.load(open(OUT))
        report["backtest"] = prev.get("backtest", backtest)  # keep frozen backtest
        logger.info("--live-only: preserved committed backtest block")
    else:
        report["backtest"] = backtest

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    logger.info("Wrote %s", OUT)
    print(json.dumps({"freeze": FREEZE, "corr": corr,
                      "backtest_combined_sharpe": report["backtest"]["books"]["combined"]["sharpe"],
                      "live_returns": {k: report["live"]["books"][k]["return_pct"] for k in report["live"]["books"]}},
                     indent=2))


if __name__ == "__main__":
    main(live_only="--live-only" in sys.argv)
