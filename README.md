# Cross-asset trend, with a residual-reversal diversification study

A vol-targeted **cross-asset trend-following** strategy (the return engine), plus an
honest test of whether a near-uncorrelated, market-neutral **residual-reversal** sleeve
improves it. Built end-to-end as a hedge-fund-style research project: factor-neutral
signals, realistic costs, overfitting controls, and a pre-registered out-of-sample track.

![Tests](https://img.shields.io/badge/tests-275%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Data](https://img.shields.io/badge/data-free%20(yfinance%20%2B%20Ken%20French%20%2B%20FRED)-blue)

> **Live dashboard:** [residual-reversal-private.vercel.app](https://residual-reversal-private.vercel.app) ([backtest](https://residual-reversal-private.vercel.app/backtest) · [live](https://residual-reversal-private.vercel.app/live)) · **Full writeup:** [STRATEGY_SELECTION.md](STRATEGY_SELECTION.md)

---

## The idea

Cross-asset time-series trend earns in persistent, trending regimes (Hurst-Ooi-Pedersen,
"A Century of Evidence on Trend-Following"). Short-horizon residual reversal earns in
choppy, mean-reverting regimes and bleeds when moves persist. They are economic
opposites, so their returns are nearly uncorrelated. The research question this project
answers honestly: **does adding a weak, uncorrelated sleeve improve a strong one?**

The honest answer here is **no, out-of-sample it did not.** Trend is the dependable
engine; the reversal sleeve is marginal, numerically sensitive near its optimizer's
turnover cliff, and in the 2025-26 momentum regime it lost, dragging the combined book
below trend alone. Reporting that candidly, rather than claiming a diversification win
that does not hold, is the point of the project.

## Results

**Backtest, frozen, 2018 → 2024-12-31** (the in-sample period parameters were chosen on):

| Sleeve | Sharpe | Ann. return | Ann. vol | Max DD |
|--------|--------|-------------|----------|--------|
| Trend | 0.46 | 4.7% | 10.2% | −18.9% |
| Reversal | 0.29* | 1.3% | 4.5% | −7.0% |
| Combined (risk-parity) | 0.52 | 5.3% | 10.2% | −13.3% |

**Live, out-of-sample, 2025-01-01 → present** (parameters frozen; refreshed daily). The live
figures drift day-to-day, and the reversal sleeve's optimizer is solver-sensitive run-to-run,
so this README deliberately does **not** hardcode them: the
[live dashboard](https://residual-reversal-private.vercel.app/live) is the single source of
truth for the current trend / reversal / combined returns and Sharpes. The honest summary is
stable regardless of the exact figures: trend earns through the 2025-26 momentum regime, the
reversal diversifier loses in it (it is structurally short momentum), and the combined book
ends modestly negative, below trend alone. That is the result the project reports.

Sleeve correlation: near-zero (about **−0.05** over the in-sample period; it is recomputed
full-sample daily, so the live dashboard's current value drifts toward zero as the out-of-sample
window grows). The reversal/combined backtest figures are marked `*`
because the reversal optimizer is degenerate near its turnover cliff: its in-sample
Sharpe varies run-to-run (roughly 0.0 to 0.4). The backtest block is therefore computed
once and frozen; the live block updates daily. Trend's numbers are stable.

## Architecture

```
residrev/
├── trend.py         Trend engine: blended 3/6/12-mo TS-momentum across 12 ETFs, inverse-vol, vol-targeted
├── signal.py        Reversal sleeve: k-day residual reversal, winsorize → sector-demean → z-score → smooth → skip-day gap
├── residuals.py     Rolling past-only FF5+UMD residualization (NumPy lstsq)
├── factors.py       Ken French factors + French-12 sectors via SEC EDGAR
├── portfolio.py     cvxpy MVO: dollar/beta/sector-neutral, turnover penalty
├── costs.py         Corwin-Schultz half-spread + Almgren sqrt-impact (daily vol)
├── backtest.py      Daily simulation loop
├── combine.py       Risk-parity / equal-weight sleeve combination + diversification report
├── analysis.py      Sharpe, drawdown, IC, capacity curve, cost sensitivity
├── validation.py    CPCV (C(6,2)=15 OOS paths) + Deflated Sharpe Ratio
├── universe.py · data.py · eda.py · conditioning.py · run.py · report.py
build_reports.py     Builds both sleeves, freezes the backtest block, emits the dashboard JSON
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python build_reports.py        # builds sleeves, writes site/public/data/report.json
pytest tests/ -v               # 275 tests
```

## Key design decisions

- **No look-ahead, structurally enforced.** Rolling OLS uses `[t-W, t-1]` windows; the reversal signal is shifted by `1 + signal_gap` days; trend signals are lagged.
- **Honest out-of-sample.** A pre-registered 2025-01-01 freeze; the backtest block is frozen (computed once), only the live block updates. No re-tuning against the live data.
- **Realistic costs.** Corwin-Schultz half-spread + Almgren √-impact using daily vol, with a capacity curve.
- **Reported the negative result.** The reversal diversifier did not help out-of-sample; the project says so plainly rather than curve-fitting a win.

## Tech stack

| Category | Tools |
|----------|-------|
| Data | yfinance, pandas-datareader (Ken French + FRED), SEC EDGAR (all free) |
| Computation | NumPy, SciPy, statsmodels; cvxpy + CLARABEL |
| Testing | pytest (275 tests), run in CI on every push (`.github/workflows/tests.yml`) |
| Site | Next.js + framer-motion (deployment repo), daily GitHub Action refreshes the live block, Vercel |

## References

- Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum
- Hurst, Ooi & Pedersen (2017) — A Century of Evidence on Trend-Following Investing
- Jegadeesh (1990) — Predictable Behavior of Security Returns
- Corwin & Schultz (2012); Almgren et al. (2005) — spread and market-impact models
- Bailey & López de Prado (2014); López de Prado (2018) — Deflated Sharpe, CPCV
- Fama & French (2015) — A Five-Factor Asset Pricing Model
