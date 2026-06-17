"""Integrated report builder for the two-sleeve book (reversal + trend).

Produces site/public/data/report.json: the single data store the Vercel site and the
daily GitHub Action consume. Deterministic (single-thread BLAS + fixed, cached ticker
universe) so the published numbers are reproducible run-to-run.

Usage:
    python build_reports.py            # build everything, write report.json
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

import numpy as np
import pandas as pd
import requests

from residrev import combine, trend
from residrev.backtest import run_backtest
from residrev.config import Config
from residrev.run import run as run_reversal_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_reports")

FREEZE = "2025-06-01"
START = "2018-01-01"
END = "2026-04-30"          # Ken French factor-data limit (reversal sleeve)
TICKER_CACHE = "universe_tickers.txt"  # committed (repo root) so CI uses a fixed, deterministic universe
OUT = "site/public/data/report.json"


def get_tickers_fixed() -> list[str]:
    """Deterministic S&P 500 universe: fetch once, sort, cache; reuse thereafter."""
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
    os.makedirs(os.path.dirname(TICKER_CACHE), exist_ok=True)
    with open(TICKER_CACHE, "w") as f:
        f.write("\n".join(tickers))
    return tickers


def reversal_sleeve() -> pd.Series:
    cfg = dataclasses.replace(Config(), start_date=START, end_date=END, aum=25e6)
    tickers = get_tickers_fixed()
    logger.info("Building reversal sleeve (%d tickers)...", len(tickers))
    result, _ = run_reversal_pipeline(cfg, tickers, eda_output_dir=None)
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


def equity_curve(r: pd.Series, downsample: int = 5) -> list:
    """[ [iso_date, cumulative_value] ] starting at 1.0, downsampled to keep JSON small."""
    cum = (1 + r).cumprod()
    cum = cum.iloc[::downsample]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in cum.items()]


def main():
    rev = reversal_sleeve()
    trd = trend_sleeve()
    spx = spx_series()

    sleeves = {"reversal": rev, "trend": trd}
    div = combine.diversification_report(sleeves, freeze=FREEZE)
    comb = combine.combine(sleeves, scheme="risk_parity", target_vol=0.10)
    combined = comb["combined"]

    freeze_ts = pd.Timestamp(FREEZE)
    panel = combine.align_sleeves(sleeves)
    series = {
        "reversal": panel["reversal"],
        "trend": trd,                         # trend alone runs longer than the common window
        "combined": combined,
    }

    def book_block(name, s):
        s = s.dropna()
        ins = s[s.index < freeze_ts]
        live = s[s.index >= freeze_ts]
        full_curve = equity_curve(s)
        return {
            "name": name,
            "as_of": s.index.max().strftime("%Y-%m-%d"),
            "metrics": {k: (None if pd.isna(v) else float(v)) for k, v in div.loc[name].items()}
            if name in div.index else {},
            "equity_curve_full": full_curve,
            "equity_curve_live": equity_curve(live) if len(live) else [],
            "insample_start": s.index.min().strftime("%Y-%m-%d"),
            "live_start": FREEZE,
        }

    report = {
        "generated_at": rev.index.max().strftime("%Y-%m-%d"),
        "freeze_date": FREEZE,
        "note": "Two-sleeve systematic book: idiosyncratic residual equity reversal + cross-asset trend. "
                "Pre-registered freeze 2025-06-01; everything after is true out-of-sample.",
        "correlation_reversal_trend": round(float(panel["reversal"].corr(panel["trend"])), 3),
        "books": {
            "reversal": book_block("reversal", series["reversal"]),
            "trend": book_block("trend", series["trend"]),
            "combined": book_block("combined_risk_parity", combined) if "combined_risk_parity" in div.index
            else book_block("combined", combined),
        },
        "diversification_table": json.loads(div.reset_index().to_json(orient="records")),
        "spx_equity_curve": equity_curve(spx.reindex(panel.index).dropna()),
    }
    # rename combined metrics key for clarity
    report["books"]["combined"]["name"] = "combined"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", OUT)
    print(json.dumps({"correlation": report["correlation_reversal_trend"],
                      "combined_metrics": report["books"]["combined"]["metrics"],
                      "diversification_table": report["diversification_table"]}, indent=2))


if __name__ == "__main__":
    main()
