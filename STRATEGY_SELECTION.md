# Quant Portfolio Project — Strategy Selection & Build Blueprint

> **Update (current direction).** The project is now framed **trend-primary**: cross-asset trend is the
> return engine, and the residual-reversal sleeve is an honestly-evaluated diversifier that did **not** pay off
> out-of-sample (it lost in the 2025-26 momentum regime and is solver-sensitive near its turnover cliff, so its
> in-sample Sharpe varies run-to-run). The pre-registered freeze moved to **2025-01-01** (in-sample 2018 to
> 2024-12-31, frozen; live 2025-present, refreshed daily). The strategy-selection analysis below remains valid;
> the headline conclusion is updated to this honest, trend-led result. For current figures see the README and the
> live dashboard.

## 1. Decision

Build a **two-sleeve systematic book**:

1. **Trend sleeve (new, primary):** cross-asset time-series trend-following across 12 liquid ETFs (equities, bonds, credit, commodities, USD), vol-targeted to 10% annual.
2. **Reversal sleeve (existing):** the idiosyncratic, factor-neutral residual short-horizon equity reversal already built in this repo.
3. **Combiner:** risk-parity (inverse-vol) weighting of the two sleeves (WSQ Module 4), with an equal-weight robustness variant.

The headline result is **not** a profit number; it is a demonstrable diversification effect: two individually-fragile, near-zero-correlation return streams combine into a book with a higher in-sample Sharpe than either sleeve and roughly half the drawdown. This is the senior-level "combine weak uncorrelated signals" result, and it is what proves research competence to an interviewer.

## 2. Why this, over the alternatives

The reversal sleeve, built first, **failed out-of-sample**: in-sample Sharpe 0.40, but frozen and tracked live (2025-06 to 2026-04) it lost 16.6% (live Sharpe −2.9). Diagnosis: 2025-06 onward was a strong-momentum bull regime, and short-horizon reversal is structurally short momentum, so it bled exactly when trend strategies thrived. That failure is the design input: the natural, durable, free-data complement is **cross-asset trend**, which earns precisely in the regimes that hurt reversal.

Strategy families considered (ranked on durability of edge, fit to a rigorous quant project, free-data feasibility, and differentiation, not on backtest Sharpe):

| Family | Durable edge | Free data | Quick-testable | Verdict |
|--------|-------------|-----------|----------------|---------|
| **Cross-asset trend-following** | High (century of evidence; Hurst-Ooi-Pedersen 2017) | Yes (ETFs) | Yes | **Primary sleeve** |
| Residual short-horizon reversal | Decayed/crowded; regime-fragile | Yes | Built | **Second sleeve** (already built) |
| SEC filing-text "Lazy Prices" (NLP) | Moderate; low-capacity | Yes (EDGAR) | No (heavy parse) | Strong future 3rd signal |
| Post-earnings-announcement drift | Moderate, persistent | Partial (earnings dates) | No | Candidate |
| Overnight/intraday session | Moderate; execution-gated | Yes (OHLC) | Yes | Lighter alternative |
| Pairs / cointegration | Crowded/decayed | Yes | Yes | Generic |
| Merger arbitrage | Persistent but deal-data gated | No (deal data not free) | No | Data-blocked |
| Volatility risk premium | Persistent but tail-risk | No (options/VIX-fut not free) | No | Data-blocked |

Trend wins on durability and free-data feasibility; combining it with the existing reversal is the senior move and salvages prior work.

## 3. Verified results (real backtests, not estimates)

> **Superseded, see the live dashboard.** This section documents an earlier exploratory run with a **2025-06-01** boundary. The project was subsequently re-frozen at the pre-registered **2025-01-01** freeze (backtest <= 2024-12-31). The single source of truth for the current numbers is the [live dashboard](https://residual-reversal-private.vercel.app) (and `site/public/data/report.json`); the only stable figures worth quoting here are the **frozen backtest** Sharpes (trend **0.46** / reversal **0.29** / combined **0.52**) and the sleeve correlation (**-0.05**). The live out-of-sample figures drift day-to-day and run-to-run, so they are intentionally not pinned in this doc. The qualitative conclusion is unchanged: trend is the engine; the reversal diversifier is near-uncorrelated but regime-fragile and hurt the book out-of-sample. The exploratory table below is retained for transparency only.

All single-threaded and reproducible. Train/live boundary (freeze) = **2025-06-01**. Common evaluation window for the two-sleeve comparison is 2018-06 to 2026-04 (the reversal sleeve needs Ken French factor data, which currently ends 2026-04; the trend sleeve alone runs 2015 to 2026-06).

**Trend sleeve, standalone (daily, 12 ETFs, vol-targeted 10%, weekly rebalance):**

| Window | Net Sharpe | Ann. return | Ann. vol | Max DD |
|--------|-----------|------------|----------|--------|
| Full (2015–2026) | 0.42 | 4.5% | 10.7% | −18.9% |
| In-sample (<2025-06) | 0.27 | 2.9% | — | — |
| Live (≥2025-06) | 1.91 | +22.7% cum | — | — |

**Two-sleeve comparison (deterministic build, common window 2018-06 to 2026-04):**

| Book | Corr. | In-sample Sharpe | Live Sharpe | Live return | Ann. vol | Max DD |
|------|-------|------------------|-------------|-------------|----------|--------|
| Reversal only | — | 0.20 | −3.39 | −18.6% | 4.8% | −21.7% |
| Trend only | — | 0.21 | +1.97 | +20.4% | 10.8% | −16.5% |
| **Combined (risk-parity)** | **−0.05** | **0.30** | −1.19 | −6.0% | 4.5% | **−11.5%** |
| Combined (equal-weight) | −0.05 | 0.28 | −0.11 | −0.8% | 5.8% | −9.1% |

The four claims that survive either weighting choice:
- **Correlation −0.05** between sleeves (genuine independence).
- **Combined in-sample Sharpe (0.28–0.30) exceeds either sleeve** (0.20, 0.21).
- **Max drawdown roughly halved** (−9 to −11.5% vs −21.7% / −16.5%).
- **Live: the trend sleeve cushioned reversal's −18.6% loss to −0.8%** (equal-weight) or −6.0% (risk-parity).

These are the deterministic, reproducible numbers `build_reports.py` publishes to the site: the universe is pinned to a fixed sorted S&P 500 list because the reversal sleeve's optimizer is sensitive to ticker ordering near its turnover cliff. An earlier pickle-order run showed a higher reversal/combined in-sample Sharpe (0.40 / 0.43); pinning the universe trades that draw for reproducibility, which matters more for a published, daily-updating site.

## 4. Honest limitations

- Trend's standalone in-sample Sharpe is modest (0.27–0.36), reflecting the well-documented 2010s trend-following drought. Its credibility rests on long-run published evidence, not this sample.
- The strong live numbers cover only ~11–13 months; that is a small sample and partly regime luck, stated as such.
- Combined live is mildly negative under both weightings (−0.8% equal-weight, −6.0% risk-parity), but far better than the reversal sleeve's −18.6%: the diversification cushioned the loss rather than erasing it. The weighting is disclosed as a robustness panel, not cherry-picked.
- All results use current index constituents historically (survivorship bias) and free retail data (yfinance/Ken French); a CRSP/WRDS rebuild would harden them.
- This is a small-capital strategy; the reversal sleeve's capacity is ~$10M (its own capacity curve).

## 5. Build blueprint (mapped to the standard quant-project structure)

- **Problem/hypothesis:** can two low-correlation systematic premia (idiosyncratic reversal, cross-asset trend) be combined into a book more robust than either? Sub-hypothesis: trend hedges reversal's momentum-regime failure.
- **Data:** yfinance (equities + ETFs), Ken French factors, FRED; point-in-time universe; cached to parquet.
- **EDA:** per-sleeve return distributions, rolling correlation, regime conditioning.
- **Signals:** reversal = `residrev` (k=5 residual reversal, gap=2 skip-day, smooth=5, sector/factor-neutral); trend = blended 3/6/12-month time-series momentum, inverse-vol sized.
- **Strategy construction:** each sleeve produces a daily net-of-cost return stream; combiner risk-parity weights them to a 10% vol target.
- **Backtest framework:** reuses the existing daily simulator, Corwin-Schultz + Almgren cost model, and pre-registered 2025-06-01 freeze.
- **Performance metrics:** Sharpe, drawdown, turnover, capacity, per-year, correlation, diversification lift.
- **Risk/robustness:** CPCV + deflated Sharpe (reused), weighting-scheme robustness, sub-period stability.
- **Findings:** the diversification result above, with the honest live verdict.

### Repo layout (extending this repo)
- `residrev/` — reversal sleeve (unchanged) + `trend.py` (new trend sleeve) + `combine.py` (new combiner).
- `build_reports.py` — integrated runner: builds both sleeves, combines, writes the JSON the site consumes.
- `site/` — Vercel app: main page (project writeup), sub-page A (2015–2025 backtest), sub-page B (2025–present live).
- `.github/workflows/update.yml` — daily cron: re-run the report, commit the JSON, Vercel auto-deploys.

## 6. Live-update & deploy design (you deploy; all artifacts provided)

- `build_reports.py` recomputes the sleeves and writes `site/public/data/backtest.json` and `site/public/data/live.json` (metrics + equity-curve series + freeze date).
- A GitHub Actions cron (daily, after US close) runs it in the project venv, commits the refreshed JSON.
- Vercel, connected to the repo, auto-deploys on every push, so the live sub-page always reflects the latest data.
- The two report pages read the JSON at build/runtime; no database needed (the JSON is the data store).
- Note: the reversal sleeve's live data is capped where free Ken French factors end (currently 2026-04); the trend sleeve updates to the latest trading day. The live page labels both as-of dates honestly.

## 7. Appendix — reusable deep-research prompt (for going further on the chosen strategy)

Use this in a fresh Opus session to deepen the trend sleeve or vet a third signal:

> You are a senior quant researcher. I have a two-sleeve book (idiosyncratic residual equity reversal + cross-asset time-series trend) with sleeve correlation −0.04, combined in-sample Sharpe 0.43, drawdown halved vs either sleeve, on free data (yfinance, Ken French, FRED), pre-registered freeze 2025-06-01. Goal: strengthen the trend sleeve and identify ONE low-correlation third signal, optimizing for durable, economically-grounded edge and honest demonstrability, NOT backtest Sharpe. Constraints: free data only; 2–4 week build; every claim must name the economic mechanism and the counterparty who loses, be net of realistic costs with a capacity estimate, and be validated with purged/embargoed CPCV + deflated Sharpe; no look-ahead, no survivorship bias, no fabricated numbers (cite literature for expected ranges, run real backtests for actuals). Deliver: (a) trend-sleeve improvements (instrument set, lookback blend, vol-targeting, breakout vs momentum) with a real before/after backtest; (b) a ranked shortlist of third signals with a runnable minimal-backtest spec each; (c) the expected diversification lift of adding the best third signal to the existing two-sleeve book; (d) honest limitations and what would kill each idea.
