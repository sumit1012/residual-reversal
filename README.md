# Residual Short-Horizon Reversal Strategy

A factor-neutral, residual short-horizon reversal strategy in US equities — built end-to-end as a hedge fund-style research project.

![Tests](https://img.shields.io/badge/tests-257%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-87%25-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## What It Does

Identifies short-term idiosyncratic price dislocations in the US equity liquid-1000 universe and fades them via a market-neutral portfolio. "Idiosyncratic" means the raw returns are first stripped of six systematic risk factors (Fama-French Five + Momentum) using a rolling past-only OLS regression — so the signal is purely stock-specific.

**Strategy in one line:** fade k-day factor-residual overshoots, dollar/factor/sector-neutral, with realistic transaction costs and honest out-of-sample validation.

---

## Architecture

```
residrev/
├── config.py        Frozen dataclass — single source of truth for all parameters
├── data.py          Resilient price pipeline: yfinance → Stooq fallback → Parquet cache
├── universe.py      Point-in-time liquid-1000: 63-day ADV rank + hysteresis buffer
├── eda.py           Exploratory analysis and exhibit generation
├── factors.py       Ken French FF5+UMD, French-12 sector map via SEC EDGAR SIC codes
├── residuals.py     Vectorized rolling past-only 6-factor OLS (NumPy lstsq, all N stocks at once)
├── signal.py        k-day reversal signal: winsorize → sector-demean → z-score → pre-shift
├── conditioning.py  Amihud illiquidity quintiles + VIX trailing-percentile terciles
├── portfolio.py     cvxpy MVO: dollar/beta/sector-neutral, IC-scaled alpha, turnover penalty
├── costs.py         Corwin-Schultz half-spread + Almgren sqrt-impact (participation = Δw·AUM/ADV)
├── backtest.py      Daily simulation loop → BacktestResult dataclass
├── analysis.py      Sharpe, drawdown, IC, capacity curve, cost sensitivity
├── validation.py    CPCV (C(6,2)=15 OOS paths) + Deflated Sharpe Ratio
├── run.py           Main entry point — CLI, trial logging, output saving
└── report.py        Generates a structured research note from BacktestResult
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/residual-reversal.git
cd residual-reversal

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (fetches S&P 500 tickers automatically)
python -m residrev.run --report

# 5. Run tests
pytest tests/ -v
```

First run takes 30–90 minutes (price download + rolling OLS across 1500 dates). Subsequent runs are fast — prices are cached to `cache/prices/`.

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for all CLI flags and output details.

---

## Key Design Decisions

**No look-ahead bias, structurally enforced.** The rolling OLS uses `[t-W, t-1]` windows with an assertion in the loop. The signal is pre-shifted one day in `signal.py` — downstream code physically cannot access today's signal on today.

**Transaction costs are realistic.** Half-spread via Corwin-Schultz (2012) estimated from daily HL prices. Market impact via the Almgren square-root model with `participation = (ΔW × AUM) / ADV` — not a flat bps assumption.

**Honest validation.** Combinatorial Purged Cross-Validation with N=6 groups, k=2 held-out generates C(6,2)=15 independent OOS test paths. The Deflated Sharpe Ratio (Bailey & López de Prado 2014) corrects the best observed Sharpe for multiple testing across parameter sweeps.

**Efficient factor risk.** The cvxpy optimizer uses `quad_form(B.T @ w, Σ_f)` instead of constructing the N×N covariance matrix — keeps the quadratic form in (K×K)=(6×6) space.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | yfinance, pandas-datareader (Ken French + FRED), SEC EDGAR |
| Computation | NumPy, SciPy, statsmodels |
| Optimization | cvxpy + CLARABEL solver |
| Testing | pytest, pytest-cov (257 tests, 87% coverage) |
| Storage | Apache Parquet via pyarrow |

---

## References

- Jegadeesh (1990) — Evidence of Predictable Behavior of Security Returns
- Corwin & Schultz (2012) — A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices
- Almgren et al. (2005) — Direct Estimation of Equity Market Impact
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio
- López de Prado (2018) — Advances in Financial Machine Learning (CPCV)
- Fama & French (2015) — A Five-Factor Asset Pricing Model
