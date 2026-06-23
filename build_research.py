"""Signal-research analytics for the dashboard (decile long-short studies).

Frozen, deterministic, backtest-window only (2018 -> 2024-12-31). Reuses the production
residualization, then studies the SIGNAL via fast vectorized decile sorts (no MVO), which
is how signal research is actually presented to a quant reviewer:

  - benchmark horse-race: residual reversal vs raw reversal vs momentum vs a market-neutral
    baseline (isolates the residualization edge),
  - parameter robustness: decile Sharpe over lookback x skip-gap,
  - regime breakdown: decile Sharpe by VIX volatility regime,
  - information coefficient and cost-sensitivity (break-even cost).

Writes site/public/data/research.json. Run: python build_research.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import dataclasses
import json
import math

import numpy as np
import pandas as pd

import build_reports as br
from residrev.config import Config
from residrev.data import pull_prices
from residrev.factors import get_ff_factors
from residrev.universe import compute_adv, get_liquid_universe
from residrev.residuals import build_return_panel, rolling_residuals
from residrev.conditioning import get_vix, compute_vix_regime

TD = 252
START, END = "2018-01-01", "2024-12-31"
OUT = os.environ.get("RESEARCH_OUT", "site/public/data/research.json")


def _xs_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Pearson correlation between two aligned T x N frames."""
    az = a.sub(a.mean(axis=1), axis=0)
    bz = b.sub(b.mean(axis=1), axis=0)
    num = (az * bz).sum(axis=1)
    den = np.sqrt((az ** 2).sum(axis=1) * (bz ** 2).sum(axis=1))
    return num / den.replace(0, np.nan)


def _signal(panel: pd.DataFrame, k: int, g: int, direction: str) -> pd.DataFrame:
    """Cumulative k-day score ending at t-g; reversal negates (buy recent losers)."""
    cum = panel.rolling(k).sum().shift(g)
    return -cum if direction == "reversal" else cum


def _decile_ls(signal: pd.DataFrame, returns: pd.DataFrame, top=0.1, cost_bps=10.0):
    """Equal-weight, dollar-neutral top-vs-bottom-decile long-short. Past-only: the signal
    at t is shifted one day so it earns t+1's return. Returns (gross, net, turnover)."""
    sig = signal.shift(1)
    valid = sig.notna() & returns.notna()
    sig = sig.where(valid)
    ranks = sig.rank(axis=1, pct=True)
    long = ranks >= (1 - top)
    short = ranks <= top
    nL = long.sum(axis=1).replace(0, np.nan)
    nS = short.sum(axis=1).replace(0, np.nan)
    w = long.div(nL, axis=0).fillna(0.0) - short.div(nS, axis=0).fillna(0.0)
    gross = (w * returns).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    ok = nL.notna() & nS.notna()
    return gross[ok].dropna(), net[ok].dropna(), turnover[ok].dropna()


def _stats(r: pd.Series) -> dict | None:
    r = r.dropna()
    if len(r) < 40 or r.std() == 0:
        return None
    cum = (1 + r).cumprod()
    ds = max(1, len(r) // 320)
    return {
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(TD)), 2),
        "ann_return_pct": round(float(r.mean() * TD) * 100, 1),
        "ann_vol_pct": round(float(r.std() * math.sqrt(TD)) * 100, 1),
        "max_dd_pct": round(float((cum / cum.cummax() - 1).min()) * 100, 1),
        "curve": [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in cum.iloc[::ds].items()],
    }


def main():
    cfg = dataclasses.replace(Config(), start_date=START, end_date=END, aum=25e6)
    tickers = br.get_tickers_fixed()
    cr, cd = "data/_research_returns.parquet", "data/_research_resid.parquet"
    if os.path.exists(cr) and os.path.exists(cd):
        print("loading cached panels...", flush=True)
        returns = pd.read_parquet(cr)
        resid = pd.read_parquet(cd)
    else:
        print("pull prices...", flush=True)
        prices = pull_prices(tickers, cfg.start_date, cfg.end_date, cfg)
        factors = get_ff_factors(cfg)
        adv = compute_adv(prices, window=cfg.adv_window)
        membership = get_liquid_universe(adv, cfg.universe_size, cfg.hysteresis_buffer)
        returns = build_return_panel(prices, membership)
        print("residualize...", flush=True)
        resid, _betas, _idio = rolling_residuals(returns, factors, cfg)
        returns.to_parquet(cr)
        resid.to_parquet(cd)

    win = (returns.index >= START) & (returns.index <= END)
    returns = returns[win]
    resid = resid.reindex(returns.index)

    K, G, COST = 5, 2, 10.0  # production-matched signal params; 10 bps round-trip on the decile sort

    # ---- benchmark horse-race (gross + net) ------------------------------------------
    print("benchmarks...", flush=True)
    defs = {
        "residual_reversal": (resid, "reversal"),
        "raw_reversal": (returns, "reversal"),
        "momentum_xs": (returns, "momentum"),       # cross-sectional momentum (raw)
        "residual_momentum": (resid, "momentum"),   # contrast: residual in the momentum direction
    }
    benchmarks = {}
    for name, (panel, direction) in defs.items():
        sig = _signal(panel, K, G, direction)
        gross, net, _to = _decile_ls(sig, returns, cost_bps=COST)
        ic = _xs_corr(sig.rank(axis=1).shift(1), returns.rank(axis=1)).dropna()
        ic_mean = float(ic.mean()) if len(ic) > 40 else None
        ic_t = float(ic.mean() / (ic.std() / math.sqrt(len(ic)))) if len(ic) > 40 and ic.std() else None
        benchmarks[name] = {
            "gross": _stats(gross),
            "net": _stats(net),
            "ic": round(ic_mean, 4) if ic_mean is not None else None,
            "ic_t": round(ic_t, 1) if ic_t is not None else None,
        }

    # ---- parameter robustness (residual reversal decile net Sharpe over k x g) --------
    print("parameter grid...", flush=True)
    lookbacks = [3, 5, 10, 21]
    gaps = [0, 1, 2]
    grid = []
    for k in lookbacks:
        row = []
        for g in gaps:
            sig = _signal(resid, k, g, "reversal")
            gross, _net, _to = _decile_ls(sig, returns, cost_bps=COST)
            st = _stats(gross)
            row.append(st["sharpe"] if st else None)
        grid.append(row)

    # ---- regime breakdown (residual reversal decile by VIX regime) -------------------
    print("regimes...", flush=True)
    regimes = None
    try:
        vix = get_vix(cfg)
        reg = compute_vix_regime(vix, cfg)
        reg = reg.reindex(returns.index).ffill()
        sig = _signal(resid, K, G, "reversal")
        gross, _net, _to = _decile_ls(sig, returns, cost_bps=COST)
        labels = {1: "Low vol", 2: "Mid vol", 3: "High vol"}
        out = []
        for code in (1, 2, 3):
            mask = reg == code
            rr = gross[gross.index.isin(reg.index[mask])]
            st = _stats(rr)
            out.append({"label": labels[code], "sharpe": st["sharpe"] if st else None,
                        "ann_return_pct": st["ann_return_pct"] if st else None,
                        "n_days": int(mask.sum())})
        regimes = out
    except Exception as e:  # VIX optional
        print("regime skip:", e, flush=True)

    # ---- cost-sensitivity (residual reversal net Sharpe over a cost grid) ------------
    print("cost curve...", flush=True)
    sig = _signal(resid, K, G, "reversal")
    gross, _net, turnover = _decile_ls(sig, returns, cost_bps=0.0)
    cost_curve = []
    for bps in [0, 5, 10, 20, 30, 50]:
        net = gross - turnover.reindex(gross.index).fillna(0) * bps / 1e4
        st = _stats(net)
        cost_curve.append({"cost_bps": bps, "sharpe": st["sharpe"] if st else None,
                           "ann_return_pct": st["ann_return_pct"] if st else None})
    ann_turnover = round(float(turnover.mean() * TD), 1)

    report = {
        "window": {"start": START, "end": END},
        "params": {"lookback": K, "skip_gap": G, "cost_bps": COST, "deciles": 10},
        "benchmarks": benchmarks,
        "parameter_grid": {"lookbacks": lookbacks, "gaps": gaps, "sharpe": grid},
        "regimes": regimes,
        "cost_curve": cost_curve,
        "decile_ann_turnover_x": ann_turnover,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print("WROTE", OUT, flush=True)
    print(json.dumps({k: (benchmarks[k]["net"]["sharpe"] if benchmarks[k]["net"] else None,
                          benchmarks[k]["ic"]) for k in benchmarks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
