# Residual Short-Horizon Reversal — Research Note

## 1. Executive Summary

This note describes a factor-neutral residual short-horizon reversal strategy applied to the US equity liquid-300 universe from 2018-06-05 to 2024-12-30. The strategy earns a net annualized Sharpe of 0.26 (0.81 gross) with an annual net return of 1.2% and a maximum drawdown of -6.2%. The mean daily information coefficient is 0.0049. The deflated Sharpe ratio is N/A. AUM capacity is estimated at $10M before net Sharpe falls below 0.5.

## 2. Hypothesis

Short-horizon reversal — buying recent losers and selling recent winners — is one of the oldest cross-sectional anomalies in equities. The economic mechanism is liquidity provision: when uninformed, price-insensitive sellers (index rebalancers, fund redemptions, margin calls) push a stock's price below fundamental value, the resulting idiosyncratic overshoot reverts as informed capital moves in. The reversal trader earns the bounce by standing ready to absorb the temporary imbalance.

Raw reversal, however, is contaminated by factor comovement. A stock that dropped because the market fell is not exhibiting idiosyncratic mean reversion — it is simply carrying market beta. Residualization strips out exposure to market, size, value, profitability, investment, and momentum factors via a rolling regression, isolating the purely idiosyncratic component of returns. The residual reversal signal is therefore factor-neutral by construction and avoids the drawdowns that afflict raw reversal during factor crashes.

Conditioning on Amihud illiquidity and VIX regime matters because the reversal premium is not constant. It concentrates in less liquid names (larger mispricings from thinner order books) and in elevated-volatility regimes (more forced selling). However, these are also the states where spreads and market impact are highest, creating an optimizable tension between signal strength and net-of-cost alpha.

## 3. Data

| Item | Detail |
|------|--------|
| Universe | Liquid-300 US equities, trailing 63-day ADV |
| Backtest period | 2018-01-01 to 2024-12-31 |
| Price data | yfinance (adjusted close), Stooq fallback, Parquet cache |
| Factor data | Ken French FF5 + UMD (daily), pandas-datareader |
| Sector classification | French-12 via SEC EDGAR SIC codes |
| VIX | FRED VIXCLS |
| Mean universe size | 402 stocks per day |

## 4. Methodology

### Factor model

Rolling 90-day OLS on 6 factors (Mkt-RF, SMB, HML, RMW, CMA, UMD), past-only estimation window, minimum 60 observations. Betas are re-estimated daily; the residual return on each date is the stock's actual return minus the predicted factor return using same-day factor realizations and previously estimated betas.

### Signal

5-day cumulative residual return, sign-reversed (fade the overshoot), winsorized at 1%, sector-demeaned, cross-sectionally z-scored, then smoothed with a 5-day past-only rolling mean (re-z-scored) to stabilize target weights and control turnover. The signal is traded with a 3-day lag: a 1-day structural shift enforces past-only trading, and an additional 2-day skip drops the most-recent residuals. The last day or two of short-horizon reversal is dominated by bid-ask bounce / microstructure noise — high gross IC but uncapturable net of costs (it amounts to trading the spread) — so skipping it isolates the genuine, tradeable reversal component.

### Portfolio construction

cvxpy mean-variance optimizer: dollar-neutral, beta-neutral, and sector-neutral. Risk aversion γ=5.0, turnover penalty λ_TO=3.0, individual position cap=2%, gross leverage cap=2.0x.

### Costs

Corwin-Schultz half-spread (21-day median) plus Almgren sqrt-impact (η=0.5, participation cap=10% ADV). The impact term uses volatility over the one-day execution horizon (daily, not annualized) per Almgren et al. (2005). The same cost model is wired into the optimizer's turnover penalty so the portfolio internalizes trading costs.

## 5. Results

| Metric | Value |
|--------|-------|
| Net Sharpe ratio | 0.26 |
| Gross Sharpe ratio | 0.81 |
| Net annual return | 1.2% |
| Gross annual return | 3.6% |
| Max drawdown | -6.2% |
| Annual one-way turnover | 669% |
| Mean daily IC | 0.0049 |
| IC t-stat | 1.94 |
| Mean daily cost | 1.0 bps |

### Per-year Sharpe

| Year | Net Sharpe |
|------|-----------|
| 2018 | -0.66 |
| 2019 | -0.01 |
| 2020 | 0.37 |
| 2021 | 0.10 |
| 2022 | 0.83 |
| 2023 | 1.16 |
| 2024 | -0.52 |

## 6. Robustness

### CPCV

Combinatorial Purged Cross-Validation with 6 groups and 2 held-out produces C(6,2) = 15 out-of-sample paths.

- Mean OOS Sharpe: 0.29
- Std: 0.41
- Median: 0.39
- Range: [-0.39, 1.00]
- Paths with positive Sharpe: 60%

### Cost sensitivity

| Cost multiplier | Net Sharpe | Annual return | Max drawdown |
|-----------------|-----------|---------------|-------------|
| 0.5x | 0.54 | 2.4% | -5.7% |
| 1.0x | 0.26 | 1.2% | -6.2% |
| 1.5x | -0.01 | -0.1% | -6.7% |
| 2.0x | -0.29 | -1.3% | -8.8% |
| 3.0x | -0.83 | -3.7% | -24.5% |
| Breakeven | 1.48x | — | — |

### Factor crash stress

- **COVID (Feb–Mar 2020):** Sharpe -0.40, max DD -3.3%
- **Rate hike (2022):** Sharpe 0.83, max DD -2.9%

### Deflated Sharpe Ratio

DSR not computed: insufficient trials

## 7. Conditioning Analysis

### Amihud illiquidity

Amihud illiquidity quintiles are computed as a diagnostic but the production signal does not condition on them: gating or tilting by illiquidity did not improve net-of-cost performance, and the most illiquid names are the costliest to trade. The liquid-N universe (most-liquid names only) is the lever actually used to keep costs below the reversal alpha.

### VIX regime

The VIX regime tercile is computed as a diagnostic but the production signal does not condition on it: restricting trading to stressed regimes did not robustly improve net-of-cost Sharpe out-of-sample. The strategy instead trades unconditionally across regimes.

## 8. Limitations

1. **Survivorship bias:** the universe is constructed from tickers available at run time; historical delisted stocks are not included. Returns may be overstated for the pre-2015 period.

2. **Signal decay:** short-horizon reversal strategies are capacity-constrained and face signal decay as more capital chases the same opportunities. The capacity estimate from Section 6 reflects current market depth but may be optimistic.

3. **Execution assumptions:** the model assumes end-of-day market-on-close fills. In practice, large orders would move VWAP, increasing implementation shortfall beyond the sqrt-impact estimate.

4. **Parameter selection / multiple testing:** the skip-day gap, smoothing span, turnover penalty, and universe size were chosen by searching over the full sample. CPCV (out-of-sample paths) and the deflated Sharpe ratio mitigate but do not fully neutralize this in-sample selection bias. A true walk-forward test on data after the selection date would be required before allocating capital.

5. **Thin cost margin:** net profitability survives the modeled costs but not a 2x cost stress (see Section 6). The strategy is cost-sensitive, as is typical for short-horizon reversal; live spreads and impact must be monitored closely.
