# PROJECT — Residual Short-Horizon Reversal in US Equities

## What are you building?
A factor-neutral residual short-horizon reversal strategy in US equities. The strategy strips out market, size, value, momentum, profitability, and investment factor returns from stock returns using rolling OLS regression, then fades the k-day idiosyncratic overshoot. It is conditioned on Amihud illiquidity and VIX regime. Portfolio is constructed via cvxpy with dollar/beta/sector neutrality and a turnover penalty. Backtested with transaction costs and validated with CPCV and deflated Sharpe.

## Tech stack
Python 3.11+, pandas, numpy, scipy, cvxpy, yfinance, pandas-datareader (Ken French factors + FRED), pytest

## What does "done" look like?
`python -m residrev.run` runs end-to-end, all 130 tests pass, pre-trust checklist runs clean.

## Primary working directory
`C:\Users\thakk\Documents\JOb search\wallstreetquants\Project_Residual_Reversal\`

## Entry point
`residrev/run.py`

## Key design decisions
- No mlfinlab (commercial); implement CPCV and deflated Sharpe from scratch (~50 lines each)
- One-day lag enforced structurally via pre-shifted signal panel
- Factor risk model: Σ = B Σ_f Bᵀ + D (factor-based, not sample covariance)
- Corwin-Schultz half-spread + sqrt-impact for costs
- VIX regime via trailing-percentile terciles (not fixed thresholds)
- SEC SIC → French 12-industry for sector-demeaning
