# ROADMAP — Residual Short-Horizon Reversal

## Phase 1: config.py — frozen dataclass, all parameters
Frozen dataclass holding every tunable parameter (universe size, OLS window, signal k, optimizer gamma, cost η, CPCV splits, etc.). Single source of truth threaded through all modules. Serializable to JSON for trial logging.

## Phase 2: data.py — resilient cached batch puller (yfinance + Stooq fallback + Parquet)
Batch-pull daily OHLCV for the full ticker superset in chunks of 75, exponential backoff, Stooq fallback for empties, Parquet cache per ticker. Also pulls Ken French FF5+UMD daily and FRED VIXCLS. Validates: duplicate dates, extreme returns (|r|>0.5), column presence.

## Phase 3: universe.py — point-in-time liquid-1000 with trailing ADV and hysteresis
Compute trailing 63-day dollar ADV, rank per date, apply entry (≤1000) / exit (>1200) hysteresis. Output: boolean `in_universe[date, ticker]` panel. No look-ahead.

## Phase 4: factors.py — Ken French FF5+UMD loading, NYSE calendar, SEC SIC sector map, factor covariance
Align factor returns to NYSE trading calendar, build SEC SIC → French 12-industry map from company_tickers.json + Siccodes12.txt. Compute trailing factor return covariance Σ_f.

## Phase 5: residuals.py — vectorized rolling past-only 6-factor OLS, residual panel, betas, idio variances
Shared pseudo-inverse across all stocks per window (fast). Past-only: window [t-W, t-1], residual at t. Min 60 obs threshold. Output: resid (T×N), betas B (T×N×K), idiosyncratic variances D (T×N).

## Phase 6: signal.py — k-day residual reversal, winsorize/demean/z-score, IC and IC-decay with HAC t-stats
s_t = −Σ resid_{t-j} (j=1..k). Winsorize 1/99 pct → sector-demean → z-score. Cross-sectional Spearman IC with Newey-West HAC t-stats. IC-decay curve h=1..10. Pre-shifted tradeable signal panel.

## Phase 7: conditioning.py — Amihud illiquidity quintiles, VIX trailing-percentile terciles, IC-by-bucket
Amihud = |r|/dollar_vol rolling 21d, cross-sectionally quintile-ranked. VIX terciles from trailing 252d percentile (point-in-time). IC-by-liquidity-bucket and IC-by-vol-bucket tables.

## Phase 8: portfolio.py — cvxpy optimizer, dollar/beta/sector-neutral, IC-scaled alpha, turnover penalty
Objective: maximize alpha·w − γ·risk − λ_to·‖w−w_prev‖₁. Constraints: dollar-neutral, |Bᵀw|≤ε (all 6 factors), sector-neutral, |w_i|≤0.02, ‖w‖₁≤2. Solvers: CLARABEL default, OSQP/SCS fallback.

## Phase 9: costs.py — Corwin-Schultz half-spread, sqrt-impact, per-rebalance cost
Corwin-Schultz spread from consecutive day high-low pairs, floored at 0, 21d median smoothed. Impact: η·σ_i·√(participation_i), η=0.5. Per-rebalance cost = Σ|Δw_i|·(halfspread_i + impact_i).

## Phase 10: backtest.py — daily loop, structurally-enforced one-day lag, BacktestResult dataclass
Pre-shifted signal panel + forward-return panel make same-bar leakage structurally impossible. BacktestResult: pnl (net+gross), positions (T×N), turnover, costs, exposures (T×K), ic, meta/config.

## Phase 11: analysis.py — Sharpe, drawdown, IC decay, factor exposure check, capacity curve, cost sweep
Annualized net Sharpe, max drawdown, hit rate, turnover. IC-decay exhibit. Factor exposure check (realized Bᵀw ≈ 0 t-stats). Capacity curve (net Sharpe vs AUM). Cost-sweep surface ({5,10,20,30} bps × η).

## Phase 12: validation.py — skip-a-day test, VWAP fill test, CPCV from scratch, deflated Sharpe, crash stress
Skip-a-day: sharpe retention ≥50%. VWAP proxy fill: (H+L+C)/3 signal+fills survive. CPCV (n=6, k=2). Deflated Sharpe from trials.jsonl (raw + clustered N). Factor-crash P&L: 2020-09, 2021-01.

## Phase 13: run.py — main entry point, trial logging to trials.jsonl, CLI flags, full output
Wires all modules. CLI flags: --start, --end, --k, --window, --aum, --cost-bps, --skip-validation. Logs each run as one JSON line to trials.jsonl. Saves BacktestResult and analysis exhibits to data/results/.
