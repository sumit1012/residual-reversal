# Residual Short-Horizon Reversal Strategy

A factor-neutral, residual short-horizon reversal strategy in US equities — built end-to-end as a hedge fund-style research project.

![Tests](https://img.shields.io/badge/tests-257%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-87%25-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## What It Does

Identifies short-term idiosyncratic price dislocations in the most-liquid ~300 US equities and fades them via a market-neutral portfolio. "Idiosyncratic" means the raw returns are first stripped of six systematic risk factors (Fama-French Five + Momentum) using a rolling past-only OLS regression — so the signal is purely stock-specific. The most-recent 1–2 days of the reversal are skipped, because that component is dominated by bid-ask bounce and is uncapturable net of trading costs.

**Strategy in one line:** fade k-day factor-residual overshoots (skipping the bid-ask-bounce window), dollar/factor/sector-neutral, on the most-liquid names, with realistic transaction costs and honest out-of-sample validation.

---

## Architecture

```
residrev/
├── config.py        Frozen dataclass — single source of truth for all parameters
├── data.py          Resilient price pipeline: yfinance → Stooq fallback → Parquet cache
├── universe.py      Point-in-time liquid-300: 63-day ADV rank + hysteresis buffer
├── eda.py           Exploratory analysis and exhibit generation
├── factors.py       Ken French FF5+UMD, French-12 sector map via SEC EDGAR SIC codes
├── residuals.py     Vectorized rolling past-only 6-factor OLS (NumPy lstsq, all N stocks at once)
├── signal.py        k-day reversal: winsorize → sector-demean → z-score → smooth → shift+skip-day gap
├── conditioning.py  Amihud illiquidity quintiles + VIX terciles (diagnostics; not used in production signal)
├── portfolio.py     cvxpy MVO: dollar/beta/sector-neutral, z-scored alpha, turnover penalty
├── costs.py         Corwin-Schultz half-spread + Almgren sqrt-impact (daily vol; participation = Δw·AUM/ADV)
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

**No look-ahead bias, structurally enforced.** The rolling OLS uses `[t-W, t-1]` windows with an assertion in the loop. The signal is shifted in `signal.py` by `1 + signal_gap` days — downstream code physically cannot access today's signal on today.

**Skip the bid-ask-bounce window — this is what makes it net-profitable.** Short-horizon reversal in the last one or two days is dominated by bid-ask bounce: huge gross IC but pure microstructure noise that costs eat entirely (you'd be trading the spread). Skipping those days (`signal_gap`) and trading only the most-liquid ~300 names is the difference between a strategy that bleeds to costs and one with a real, if thin, net edge.

**Transaction costs are realistic.** Half-spread via Corwin-Schultz (2012) estimated from daily HL prices. Market impact via the Almgren square-root model using *daily* volatility over the execution horizon, with `participation = (ΔW × AUM) / ADV` — not a flat bps assumption.

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

## Sample Results

Full backtest, 2018-06 to 2024-12, most-liquid ~300 S&P names. Run single-threaded
for reproducibility; exact figures shift modestly with the data-pull date.

| Metric | Value |
|--------|-------|
| Gross Sharpe | 0.81 |
| Net Sharpe (after costs) | 0.26 |
| Net annual return | +1.2% |
| Max drawdown | −6.2% |
| Mean daily IC (t-stat) | 0.0049 (1.94) |
| Annual turnover | 669% |
| CPCV out-of-sample | 60% of 15 paths positive (mean 0.29) |
| Est. capacity (net Sharpe ≥ 0.5) | ~$10M AUM |

**Honest caveats.** The net edge is real but **thin**: it does not survive a 2×
transaction-cost stress, per-year net Sharpe ranges from −0.66 (2018) to +1.16
(2023), and the strategy is capacity-constrained (~$10M). Parameters (skip-day gap,
smoothing span, universe size) were selected in-sample; CPCV mitigates but does not
eliminate selection bias, so a true walk-forward test is required before trusting
the live edge. This is a textbook short-horizon-reversal result — strong gross
predictability that transaction costs nearly consume.

[Sample research note](docs/sample_output/research_note.md) · [Visual report (HTML)](docs/sample_output/report.html) · [Full terminal output](docs/sample_output/terminal_output.txt)

---

## References

- Jegadeesh (1990) — Evidence of Predictable Behavior of Security Returns
- Corwin & Schultz (2012) — A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices
- Almgren et al. (2005) — Direct Estimation of Equity Market Impact
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio
- López de Prado (2018) — Advances in Financial Machine Learning (CPCV)
- Fama & French (2015) — A Five-Factor Asset Pricing Model
