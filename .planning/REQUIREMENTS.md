# REQUIREMENTS — Residual Short-Horizon Reversal

## Functional Requirements

### Data
- Pull daily OHLCV for a broad US equity universe (~4,000–6,000 tickers) via yfinance
- Automatic Stooq fallback for tickers yfinance returns empty
- Cache all raw pulls to Parquet (idempotent, delta-fetch on rerun)
- Download Ken French FF5 + UMD daily factors and FRED VIXCLS
- Build SEC SIC → French 12-industry sector map

### Universe
- Reconstruct point-in-time liquid-1000 using trailing 63-day ADV
- Hysteresis: enter at rank ≤ 1000, exit at rank > 1200
- No look-ahead in universe membership

### Signal
- Rolling 90-day past-only OLS on 6 factors (Mkt-RF, SMB, HML, RMW, CMA, UMD)
- k-day residual reversal (base k=5): s_t = −Σ resid_{t-j}
- Winsorize at 1st/99th pct → sector-demean → cross-sectional z-score
- IC (Spearman) with Newey-West HAC t-stats; IC-decay curve over h=1..10

### Conditioning
- Amihud illiquidity: 21-day trailing, quintile-ranked cross-sectionally
- VIX regime: trailing 252-day percentile terciles (point-in-time)

### Portfolio Construction
- cvxpy optimizer: dollar-neutral, beta-neutral (all 6 factors), sector-neutral
- Factor risk model: Σ = B Σ_f Bᵀ + D
- IC-scaled alpha: alpha_i = IC · σ_resid_i · z_i
- Turnover penalty in objective (not hard constraint)
- Per-name cap 2%, gross leverage cap 2.0×

### Costs
- Corwin-Schultz half-spread from daily high-low
- Square-root market impact: η=0.5, participation = |Δshares| / ADV
- Same cost model wired into optimizer turnover penalty

### Backtest
- Structural one-day lag: pre-shifted signal panel, forward-return panel
- BacktestResult dataclass: pnl, positions, turnover, costs, exposures, ic, meta
- Net Sharpe target: ~0.8–1.3; flag if > 2 (likely bug)

### Validation
- Skip-a-day test: sharpe retention ≥ 50% is healthy
- Mid/VWAP fill test: signal + fills on (H+L+C)/3 proxy
- CPCV: n_groups=6, k_test=2, purge=holding horizon, embargo≈1%
- Deflated Sharpe: log all trials to trials.jsonl, raw + clustered N
- Factor-crash stress: 2020 momentum reversal, Jan-2021 squeeze

## Non-Functional Requirements
- All parameters in config.py frozen dataclass
- pytest --collect-only exits 0; target 130 tests
- python -m residrev.run runs end-to-end
- No look-ahead enforced by test assertions
- All headline Sharpe figures are net-of-cost
