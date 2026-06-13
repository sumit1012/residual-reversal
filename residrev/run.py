"""Main entry point — wires every module in order and saves outputs."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests

from residrev.analysis import summarize
from residrev.backtest import BacktestResult, run_backtest
from residrev.conditioning import compute_amihud, compute_vix_regime, get_vix
from residrev.config import Config
from residrev.costs import corwin_schultz_spread, compute_realized_vol
from residrev.data import pull_prices
from residrev.eda import run_eda
from residrev.factors import get_ff_factors, get_sector_map
from residrev.residuals import build_return_panel, compute_idio_vol, rolling_residuals
from residrev.signal import build_signal
from residrev.universe import compute_adv, get_liquid_universe
from residrev.validation import run_pre_trust_checklist

logger = logging.getLogger(__name__)


def build_config(args: argparse.Namespace) -> Config:
    """Construct Config, overriding only fields explicitly passed on the CLI."""
    overrides: dict = {}
    mapping = {
        "start_date": "start_date",
        "end_date": "end_date",
        "universe_size": "universe_size",
        "signal_k": "signal_k",
        "gamma": "gamma",
        "lam_to": "lam_to",
        "output_dir": "data_dir",
    }
    for attr, field in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides[field] = val
    return dataclasses.replace(Config(), **overrides)


def get_tickers(args: argparse.Namespace) -> list[str]:
    """Return ticker list from file or Wikipedia S&P 500."""
    if getattr(args, "tickers_file", None):
        with open(args.tickers_file) as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        logger.info("Fetching S&P 500 constituents from Wikipedia")
        _resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (research-bot; educational use)"},
            timeout=30,
        )
        _resp.raise_for_status()
        tickers = (
            pd.read_html(io.StringIO(_resp.text))[0]["Symbol"]
            .str.replace(".", "-", regex=False)
            .tolist()
        )
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    logger.info("Loaded %d tickers", len(tickers))
    return tickers


def run(
    config: Config,
    tickers: list[str],
    eda_output_dir: str | None = None,
) -> tuple[BacktestResult, dict]:
    """Execute the full pipeline. Returns (BacktestResult, summary).

    Pass eda_output_dir to also save EDA exhibits; None skips EDA.
    """
    logger.info("Pulling price data for %d tickers", len(tickers))
    prices = pull_prices(tickers, config.start_date, config.end_date, config)

    logger.info("Fetching FF factors")
    factors = get_ff_factors(config)

    logger.info("Computing ADV")
    adv = compute_adv(prices, window=config.adv_window)

    logger.info("Building universe membership")
    membership = get_liquid_universe(adv, config.universe_size, config.hysteresis_buffer)

    logger.info("Building return panel")
    returns = build_return_panel(prices, membership)

    logger.info("Fetching sector map")
    sector_map = get_sector_map(tickers, config)

    logger.info("Computing rolling residuals")
    resid, betas, idio_var = rolling_residuals(returns, factors, config)

    logger.info("Computing idiosyncratic vol")
    compute_idio_vol(idio_var)

    logger.info("Building signal")
    raw_signal, tradeable_signal = build_signal(resid, sector_map, config)

    logger.info("Computing Amihud illiquidity")
    compute_amihud(prices, membership, config)

    logger.info("Getting VIX and regime")
    vix = get_vix(config)
    compute_vix_regime(vix, config)

    logger.info("Computing Corwin-Schultz spread")
    spread = corwin_schultz_spread(prices, config.cs_smooth_window)

    logger.info("Computing realized vol")
    vol = compute_realized_vol(returns)

    logger.info("Running backtest")
    result = run_backtest(
        tradeable_signal=tradeable_signal,
        returns=returns,
        betas=betas,
        idio_var=idio_var,
        factors=factors,
        sector_map=sector_map,
        spread=spread,
        adv=adv,
        vol=vol,
        config=config,
    )

    logger.info("Summarizing results")
    summary = summarize(result)

    if eda_output_dir is not None:
        logger.info("Running EDA")
        run_eda(
            prices=prices,
            membership=membership,
            adv=adv,
            signal=raw_signal,
            fwd_returns=returns,
            output_dir=eda_output_dir,
        )

    return result, summary


def save_outputs(
    result: BacktestResult, summary: dict, run_id: str, config: Config
) -> None:
    """Save all artifacts to {data_dir}/results/{run_id}/."""
    out_dir = os.path.join(config.data_dir, "results", run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pnl_df = pd.DataFrame(
        {
            "gross_pnl": result.gross_pnl,
            "net_pnl": result.pnl,
            "costs_bps": result.costs_bps,
        }
    )
    pnl_df.to_parquet(os.path.join(out_dir, "pnl.parquet"))
    result.positions.to_parquet(os.path.join(out_dir, "positions.parquet"))
    result.factor_exposures.to_parquet(os.path.join(out_dir, "factor_exposures.parquet"))
    result.ic_series.to_frame("ic").to_parquet(os.path.join(out_dir, "ic_series.parquet"))
    logger.info("Outputs saved to %s", out_dir)


def log_trial(summary: dict, config: Config, run_id: str) -> None:
    """Append one JSON line to the trials log."""
    log_path = config.trials_log
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "summary": summary,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Trial logged to %s", log_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual Reversal Strategy")
    parser.add_argument("--start-date", dest="start_date", default=None)
    parser.add_argument("--end-date", dest="end_date", default=None)
    parser.add_argument("--universe-size", dest="universe_size", type=int, default=None)
    parser.add_argument("--signal-k", dest="signal_k", type=int, default=None)
    parser.add_argument("--gamma", dest="gamma", type=float, default=None)
    parser.add_argument("--lam-to", dest="lam_to", type=float, default=None)
    parser.add_argument("--tickers-file", dest="tickers_file", default=None)
    parser.add_argument("--eda", action="store_true", default=False)
    parser.add_argument("--no-checklist", dest="no_checklist", action="store_true", default=False)
    parser.add_argument("--report", action="store_true", default=False,
                        help="Generate research note to data/results/<run_id>/research_note.md")
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = os.path.join("data", "results", run_id)
    os.makedirs(log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    logger.info("Run ID: %s", run_id)

    try:
        args = _parse_args()
        config = build_config(args)
        tickers = get_tickers(args)
        eda_dir = os.path.join(config.data_dir, "results", run_id, "eda") if args.eda else None
        result, summary = run(config, tickers, eda_output_dir=eda_dir)
        save_outputs(result, summary, run_id, config)
        log_trial(summary, config, run_id)
        checklist_result = None
        if not args.no_checklist:
            checklist_result = run_pre_trust_checklist(result, config)
            summary["checklist"] = checklist_result
        if args.report:
            from residrev.report import generate_report
            from residrev.html_report import generate_html_report

            report_path = os.path.join(config.data_dir, "results", run_id, "research_note.md")
            generate_report(result, summary, checklist_result, config, output_path=report_path)

            html_path = os.path.join(config.data_dir, "results", run_id, "report.html")
            generate_html_report(result, summary, checklist_result, config,
                                 output_path=html_path, run_id=run_id)
        print(json.dumps(summary, indent=2))
    except Exception:
        logger.exception("Run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
