# Residual Reversal + Cross-Asset Trend — a two-sleeve systematic book

Two low-correlation systematic premia combined into one book: a market-neutral,
factor-neutral **residual short-horizon reversal** sleeve in US equities, and a
vol-targeted **cross-asset trend-following** sleeve across global ETFs. Built
end-to-end as a hedge-fund-style research project, with honest costs, overfitting
controls, and a pre-registered out-of-sample live track.

![Tests](https://img.shields.io/badge/tests-266%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Data](https://img.shields.io/badge/data-free%20(yfinance%20%2B%20Ken%20French%20%2B%20FRED)-blue)

> **Live dashboard:** _(Vercel URL — add after deploy)_ · **Full writeup:** [STRATEGY_SELECTION.md](STRATEGY_SELECTION.md)

---

## The idea in one paragraph

Short-horizon residual reversal fades idiosyncratic overshoots; it earns in choppy,
mean-reverting regimes and **bleeds when moves persist**. Cross-asset trend rides
persistent moves; it earns precisely in those regimes. They are economic opposites,
so their returns are nearly uncorrelated (realized correlation **−0.05**) and they
hedge each other's worst environments. The reversal sleeve was built first and
**failed out-of-sample** in the 2025–26 momentum regime (live −18.6%); the trend
sleeve was added because it harvests exactly that regime (live +20.4%). Combining
them is the demonstration: the blended book has a **higher in-sample Sharpe than
either sleeve and roughly half the drawdown.**

---

## Results (deterministic, reproducible; common window 2018-06 → 2026-04)

| Book | Corr. | In-sample Sharpe | Live Sharpe | Live return | Max DD |
|------|-------|------------------|-------------|-------------|--------|
| Reversal only | — | 0.20 | −3.39 | −18.6% | −21.7% |
| Trend only | — | 0.21 | +1.97 | +20.4% | −16.5% |
| **Combined (risk-parity)** | **−0.05** | **0.30** | −1.19 | −6.0% | **−11.5%** |
| Combined (equal-weight) | −0.05 | 0.28 | −0.11 | −0.8% | −9.1% |

What survives any reasonable weighting: near-zero sleeve correlation, a combined
in-sample Sharpe above both sleeves, and a roughly halved drawdown. Numbers are
modest and honestly so — the point is the rigorous diversification result, not a
headline return.

**Honest limitations.** Trend's standalone in-sample Sharpe is modest (the documented
2010s trend drought); the live window is short; combined live is mildly negative
under both weightings (but far better than reversal's −18.6%); results use current
index constituents historically (survivorship bias) and free retail data. See
[STRATEGY_SELECTION.md](STRATEGY_SELECTION.md) for the full analysis and the strategy
search that led here.

---

## Architecture

```
residrev/
├── config.py        Frozen dataclass — single source of truth for all parameters
├── data.py          Resilient price pipeline: yfinance → Stooq fallback → Parquet cache
├── universe.py      Point-in-time liquid-300: 63-day ADV rank + hysteresis buffer
├── factors.py       Ken French FF5+UMD, French-12 sector map via SEC EDGAR SIC codes
├── residuals.py     Vectorized rolling past-only 6-factor OLS (NumPy lstsq, all N at once)
├── signal.py        k-day reversal: winsorize → sector-demean → z-score → smooth → shift+skip-day gap
├── trend.py         Cross-asset trend sleeve: blended 3/6/12-mo TS-momentum, inverse-vol, vol-targeted
├── portfolio.py     cvxpy MVO: dollar/beta/sector-neutral, z-scored alpha, turnover penalty
├── costs.py         Corwin-Schultz half-spread + Almgren sqrt-impact (daily vol)
├── backtest.py      Daily simulation loop → BacktestResult
├── combine.py       Risk-parity / equal-weight sleeve combination + diversification report
├── analysis.py      Sharpe, drawdown, IC, capacity curve, cost sensitivity
├── validation.py    CPCV (C(6,2)=15 OOS paths) + Deflated Sharpe Ratio
├── eda.py           Exploratory analysis and exhibit generation
├── run.py           Reversal pipeline entry point
└── report.py        Structured research note generator
build_reports.py     Builds both sleeves, combines, emits the JSON the dashboard consumes
```

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate     # (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python build_reports.py        # builds reversal + trend + combined, writes site/public/data/report.json
pytest tests/ -v               # 266 tests
```

First reversal build downloads price history (slow once, then cached). See
[HOW_TO_RUN.md](HOW_TO_RUN.md) for CLI flags.

---

## Key design decisions

- **No look-ahead, structurally enforced.** Rolling OLS uses `[t-W, t-1]` windows; the signal is shifted by `1 + signal_gap` days; trend signals are lagged.
- **Skip the bid-ask-bounce window.** The last 1–2 days of reversal are microstructure noise that costs consume; skipping them is what gives a real net edge.
- **Realistic costs.** Corwin-Schultz half-spread + Almgren √-impact using *daily* vol, with a capacity curve.
- **Honest validation.** CPCV (purged/embargoed) + Deflated Sharpe; a pre-registered 2025-06-01 freeze separates in-sample from a live out-of-sample track.
- **Combining beats either sleeve.** Two uncorrelated, individually-fragile signals risk-parity-weighted into a more robust book (WSQ Module 4).

---

## Tech stack

| Category | Tools |
|----------|-------|
| Data | yfinance, pandas-datareader (Ken French + FRED), SEC EDGAR (all free) |
| Computation | NumPy, SciPy, statsmodels |
| Optimization | cvxpy + CLARABEL |
| Testing | pytest (266 tests) |
| Storage | Apache Parquet via pyarrow; JSON report for the dashboard |

---

## References

- Jegadeesh (1990) — Evidence of Predictable Behavior of Security Returns
- Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum
- Hurst, Ooi & Pedersen (2017) — A Century of Evidence on Trend-Following Investing
- Corwin & Schultz (2012) — Estimating Bid-Ask Spreads from Daily High and Low Prices
- Almgren et al. (2005) — Direct Estimation of Equity Market Impact
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio
- López de Prado (2018) — Advances in Financial Machine Learning (CPCV)
- Fama & French (2015) — A Five-Factor Asset Pricing Model
