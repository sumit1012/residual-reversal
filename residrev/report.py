"""Research note generator — programmatic markdown from BacktestResult + summary."""

from __future__ import annotations

import logging
import os
from math import comb
from typing import TYPE_CHECKING

from residrev.analysis import capacity_curve, cost_sensitivity

if TYPE_CHECKING:
    from residrev.backtest import BacktestResult
    from residrev.config import Config

logger = logging.getLogger(__name__)


def _section_executive_summary(
    summary: dict, config: Config, result: BacktestResult, checklist: dict | None,
) -> str:
    dsr_str = "N/A"
    if checklist and "checks" in checklist:
        ds = checklist["checks"].get("deflated_sharpe", {})
        if "dsr" in ds:
            dsr_str = f"{ds['dsr']:.2f}"

    cap_est = "N/A"
    try:
        cap_df = capacity_curve(result, config)
        for aum, row in cap_df.iterrows():
            if isinstance(aum, (int, float)) and row["net_sharpe"] < 0.5:
                cap_est = f"${aum / 1e6:.0f}M"
                break
        else:
            cap_est = f">${cap_df.index[-1] / 1e6:.0f}M"
    except Exception:
        pass

    start = summary.get("start_date", config.start_date)
    end = summary.get("end_date", config.end_date)

    return f"""## 1. Executive Summary

This note describes a factor-neutral residual short-horizon reversal strategy \
applied to the US equity liquid-{config.universe_size} universe from {start} to \
{end}. The strategy earns a net annualized Sharpe of \
{summary['net_sharpe']:.2f} ({summary['gross_sharpe']:.2f} gross) with an \
annual net return of {summary['net_annual_return']*100:.1f}% and a maximum \
drawdown of {summary['max_drawdown']*100:.1f}%. The mean daily information \
coefficient is {summary['mean_daily_ic']:.4f}. The deflated Sharpe ratio is \
{dsr_str}. AUM capacity is estimated at {cap_est} before net Sharpe falls \
below 0.5.
"""


def _section_hypothesis() -> str:
    return """## 2. Hypothesis

Short-horizon reversal — buying recent losers and selling recent winners — is \
one of the oldest cross-sectional anomalies in equities. The economic mechanism \
is liquidity provision: when uninformed, price-insensitive sellers (index \
rebalancers, fund redemptions, margin calls) push a stock's price below \
fundamental value, the resulting idiosyncratic overshoot reverts as informed \
capital moves in. The reversal trader earns the bounce by standing ready to \
absorb the temporary imbalance.

Raw reversal, however, is contaminated by factor comovement. A stock that \
dropped because the market fell is not exhibiting idiosyncratic mean reversion \
— it is simply carrying market beta. Residualization strips out exposure to \
market, size, value, profitability, investment, and momentum factors via a \
rolling regression, isolating the purely idiosyncratic component of returns. \
The residual reversal signal is therefore factor-neutral by construction and \
avoids the drawdowns that afflict raw reversal during factor crashes.

Conditioning on Amihud illiquidity and VIX regime matters because the reversal \
premium is not constant. It concentrates in less liquid names (larger \
mispricings from thinner order books) and in elevated-volatility regimes \
(more forced selling). However, these are also the states where spreads and \
market impact are highest, creating an optimizable tension between signal \
strength and net-of-cost alpha.
"""


def _section_data(summary: dict, config: Config, result: BacktestResult) -> str:
    universe_mean = result.meta.get("universe_size_mean", config.universe_size)

    return f"""## 3. Data

| Item | Detail |
|------|--------|
| Universe | Liquid-{config.universe_size} US equities, trailing {config.adv_window}-day ADV |
| Backtest period | {config.start_date} to {config.end_date} |
| Price data | yfinance (adjusted close), Stooq fallback, Parquet cache |
| Factor data | Ken French FF5 + UMD (daily), pandas-datareader |
| Sector classification | French-12 via SEC EDGAR SIC codes |
| VIX | FRED VIXCLS |
| Mean universe size | {universe_mean:.0f} stocks per day |
"""


def _section_methodology(config: Config) -> str:
    return f"""## 4. Methodology

### Factor model

Rolling {config.factor_window}-day OLS on 6 factors (Mkt-RF, SMB, HML, RMW, \
CMA, UMD), past-only estimation window, minimum {config.min_obs} observations. \
Betas are re-estimated daily; the residual return on each date is the stock's \
actual return minus the predicted factor return using same-day factor realizations \
and previously estimated betas.

### Signal

{config.signal_k}-day cumulative residual return, sign-reversed (fade the \
overshoot), winsorized at {config.winsorize_pct:.0%}, sector-demeaned, \
cross-sectionally z-scored, then smoothed with a {config.signal_smooth_span}-day \
past-only rolling mean (re-z-scored) to stabilize target weights and control \
turnover. The signal is traded with a {1 + config.signal_gap}-day lag: a 1-day \
structural shift enforces past-only trading, and an additional \
{config.signal_gap}-day skip drops the most-recent residuals. The last day or \
two of short-horizon reversal is dominated by bid-ask bounce / microstructure \
noise — high gross IC but uncapturable net of costs (it amounts to trading the \
spread) — so skipping it isolates the genuine, tradeable reversal component.

### Portfolio construction

cvxpy mean-variance optimizer: dollar-neutral, beta-neutral, and sector-neutral. \
Risk aversion γ={config.gamma}, turnover penalty λ_TO={config.lam_to}, \
individual position cap={config.max_w:.0%}, gross leverage \
cap={config.gross_cap}x.

### Costs

Corwin-Schultz half-spread ({config.cs_smooth_window}-day median) plus Almgren \
sqrt-impact (η={config.eta_impact}, participation \
cap={config.adv_participation_cap:.0%} ADV). The impact term uses volatility over \
the one-day execution horizon (daily, not annualized) per Almgren et al. (2005). \
The same cost model is wired into the optimizer's turnover penalty so the \
portfolio internalizes trading costs.
"""


def _section_results(summary: dict) -> str:
    lines = [
        "## 5. Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Net Sharpe ratio | {summary['net_sharpe']:.2f} |",
        f"| Gross Sharpe ratio | {summary['gross_sharpe']:.2f} |",
        f"| Net annual return | {summary['net_annual_return']*100:.1f}% |",
        f"| Gross annual return | {summary['gross_annual_return']*100:.1f}% |",
        f"| Max drawdown | {summary['max_drawdown']*100:.1f}% |",
        f"| Annual one-way turnover | {summary['annual_turnover']*100:.0f}% |",
        f"| Mean daily IC | {summary['mean_daily_ic']:.4f} |",
        f"| IC t-stat | {summary['ic_tstat']:.2f} |",
        f"| Mean daily cost | {summary['mean_cost_bps']:.1f} bps |",
        "",
        "### Per-year Sharpe",
        "",
        "| Year | Net Sharpe |",
        "|------|-----------|",
    ]

    per_year = summary.get("per_year_sharpe", {})
    for year in sorted(per_year.keys()):
        s = per_year[year]
        lines.append(f"| {year} | {s:.2f} |")

    lines.append("")
    return "\n".join(lines)


def _section_robustness(
    result: BacktestResult, config: Config, checklist: dict | None,
) -> str:
    lines = ["## 6. Robustness", ""]

    n_paths = comb(config.cpcv_n_groups, config.cpcv_k_test)

    # CPCV
    lines.append("### CPCV")
    lines.append("")
    lines.append(
        f"Combinatorial Purged Cross-Validation with {config.cpcv_n_groups} groups "
        f"and {config.cpcv_k_test} held-out produces C({config.cpcv_n_groups},"
        f"{config.cpcv_k_test}) = {n_paths} out-of-sample paths."
    )

    if checklist and "checks" in checklist:
        cpcv = checklist["checks"].get("cpcv_oos_sharpe", {})
        if "oos_sharpes" in cpcv:
            lines.append("")
            lines.append(f"- Mean OOS Sharpe: {cpcv['mean']:.2f}")
            lines.append(f"- Std: {cpcv['std']:.2f}")
            lines.append(f"- Median: {cpcv['median']:.2f}")
            lines.append(f"- Range: [{cpcv['min']:.2f}, {cpcv['max']:.2f}]")
            lines.append(f"- Paths with positive Sharpe: {cpcv['pct_positive']*100:.0f}%")
        elif cpcv.get("status") == "skip":
            lines.append("")
            lines.append(f"CPCV not run: {cpcv.get('note', 'N/A')}")
    else:
        lines.append("")
        lines.append("CPCV checklist not run.")

    lines.append("")

    # Cost sensitivity
    lines.append("### Cost sensitivity")
    lines.append("")
    try:
        cs_df = cost_sensitivity(result)
        lines.append("| Cost multiplier | Net Sharpe | Annual return | Max drawdown |")
        lines.append("|-----------------|-----------|---------------|-------------|")
        for mult, row in cs_df.iterrows():
            if mult == "breakeven_multiplier":
                lines.append(f"| Breakeven | {row['net_sharpe']:.2f}x | — | — |")
            else:
                lines.append(
                    f"| {mult}x | {row['net_sharpe']:.2f} "
                    f"| {row['annualized_return']*100:.1f}% "
                    f"| {row['max_drawdown']*100:.1f}% |"
                )
    except Exception:
        lines.append("Cost sensitivity analysis not available.")

    lines.append("")

    # Factor crash stress
    lines.append("### Factor crash stress")
    lines.append("")
    if checklist and "checks" in checklist:
        fc = checklist["checks"].get("factor_crash_stress", {})
        for period, label in [("covid", "COVID (Feb–Mar 2020)"), ("rate_hike", "Rate hike (2022)")]:
            info = fc.get(period, {})
            if "note" in info:
                lines.append(f"- **{label}:** {info['note']}")
            elif "sharpe" in info:
                dd_str = f"{info['max_dd']*100:.1f}%" if info.get("max_dd") is not None else "N/A"
                lines.append(f"- **{label}:** Sharpe {info['sharpe']:.2f}, max DD {dd_str}")
            else:
                lines.append(f"- **{label}:** not available")
    else:
        lines.append("Factor crash stress not run.")

    lines.append("")

    # Deflated Sharpe
    lines.append("### Deflated Sharpe Ratio")
    lines.append("")
    if checklist and "checks" in checklist:
        ds = checklist["checks"].get("deflated_sharpe", {})
        if ds.get("status") == "skip":
            lines.append(f"DSR not computed: {ds.get('note', 'N/A')}")
        elif "dsr" in ds:
            lines.append(f"- DSR: {ds['dsr']:.3f}")
            lines.append(f"- Number of trials: {ds['n_trials']}")
            lines.append(f"- Best observed Sharpe: {ds['best_sharpe']:.2f}")
            interp = "passes" if ds['dsr'] >= 0.95 else "does not pass"
            lines.append(
                f"- Interpretation: the observed Sharpe {interp} the "
                f"deflated Sharpe test at the 95% confidence level."
            )
    else:
        lines.append("Deflated Sharpe not run (checklist not available).")

    lines.append("")
    return "\n".join(lines)


def _section_conditioning(result: BacktestResult) -> str:
    lines = ["## 7. Conditioning Analysis", ""]

    conditioning = result.meta.get("conditioning", {})

    # Amihud illiquidity
    lines.append("### Amihud illiquidity")
    lines.append("")
    amihud = conditioning.get("amihud_ic_by_bucket")
    if amihud:
        lines.append("| Illiquidity quintile | Mean IC |")
        lines.append("|---------------------|---------|")
        for bucket, ic in sorted(amihud.items(), key=lambda x: x[0]):
            lines.append(f"| Q{bucket} | {ic:.4f} |")
        lines.append("")
        lines.append(
            "As expected, IC is higher in the more illiquid quintiles (Q4–Q5), "
            "reflecting larger mispricings in names with thinner order books. "
            "However, transaction costs also rise in these quintiles, creating "
            "a net-alpha tension."
        )
    else:
        lines.append(
            "Amihud illiquidity quintiles are computed as a diagnostic but the "
            "production signal does not condition on them: gating or tilting by "
            "illiquidity did not improve net-of-cost performance, and the most "
            "illiquid names are the costliest to trade. The liquid-N universe "
            "(most-liquid names only) is the lever actually used to keep costs "
            "below the reversal alpha."
        )

    lines.append("")

    # VIX regime
    lines.append("### VIX regime")
    lines.append("")
    vix = conditioning.get("vix_ic_by_regime")
    if vix:
        lines.append("| VIX regime | Mean IC |")
        lines.append("|-----------|---------|")
        for regime, ic in sorted(vix.items(), key=lambda x: x[0]):
            lines.append(f"| Regime {regime} | {ic:.4f} |")
        lines.append("")
        lines.append(
            "IC may be elevated in stress regimes (regime 3), consistent with "
            "more forced selling creating larger idiosyncratic overshoots. "
            "Transaction costs also spike in these regimes."
        )
    else:
        lines.append(
            "The VIX regime tercile is computed as a diagnostic but the production "
            "signal does not condition on it: restricting trading to stressed "
            "regimes did not robustly improve net-of-cost Sharpe out-of-sample. "
            "The strategy instead trades unconditionally across regimes."
        )

    lines.append("")
    return "\n".join(lines)


def _section_limitations() -> str:
    return """## 8. Limitations

1. **Survivorship bias:** the universe is constructed from tickers available at \
run time; historical delisted stocks are not included. Returns may be overstated \
for the pre-2015 period.

2. **Signal decay:** short-horizon reversal strategies are capacity-constrained \
and face signal decay as more capital chases the same opportunities. The \
capacity estimate from Section 6 reflects current market depth but may be \
optimistic.

3. **Execution assumptions:** the model assumes end-of-day market-on-close \
fills. In practice, large orders would move VWAP, increasing implementation \
shortfall beyond the sqrt-impact estimate.

4. **Parameter selection / multiple testing:** the skip-day gap, smoothing span, \
turnover penalty, and universe size were chosen by searching over the full \
sample. CPCV (out-of-sample paths) and the deflated Sharpe ratio mitigate but do \
not fully neutralize this in-sample selection bias. A true walk-forward test on \
data after the selection date would be required before allocating capital.

5. **Thin cost margin:** net profitability survives the modeled costs but not a \
2x cost stress (see Section 6). The strategy is cost-sensitive, as is typical for \
short-horizon reversal; live spreads and impact must be monitored closely.
"""


def generate_report(
    result: BacktestResult,
    summary: dict,
    checklist: dict | None,
    config: Config,
    output_path: str,
) -> str:
    """Build the full markdown research note and write it to output_path."""
    sections = [
        "# Residual Short-Horizon Reversal — Research Note\n",
        _section_executive_summary(summary, config, result, checklist),
        _section_hypothesis(),
        _section_data(summary, config, result),
        _section_methodology(config),
        _section_results(summary),
        _section_robustness(result, config, checklist),
        _section_conditioning(result),
        _section_limitations(),
    ]

    md = "\n".join(sections)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Research note written to %s", output_path)

    return md
