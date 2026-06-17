# Multi-Factor Equity Sleeve, Scoping Spec (for review)

> Status: SCOPE for review, not yet built. Produced by a 7-agent research + critique workflow grounded in this repo. Critic verdict: CONDITIONAL GO as a small price-only low-risk diversifier (not a return engine), after the listed must-fixes.

## 1 Hypothesis (falsifiable) and how it differs from the existing two sleeves

**Hypothesis (H1).** A daily, *cross-sectional*, *price-only* low-risk-plus-residual-momentum equity sleeve, built entirely from data already cached in this repo, is (a) buildable honestly on free data, (b) statistically near-orthogonal to *both* existing sleeves (|full-sample return correlation| < 0.3 to each), and (c) when added as a third risk-parity sleeve, reduces the combined book's max drawdown *without lowering combined in-sample Sharpe*, under **both** risk-parity and equal-weight combination.

**Falsification (any one kills the sleeve).** H1 is false if, on the frozen window (<= 2024-12-31): the sleeve's |corr| to trend or to reversal is >= 0.3; its OWN deflated Sharpe (deflated by the honestly-logged trial count) is < 0.80; CPCV OOS `pct_positive` < 0.75; PBO > 0.20; or the three-sleeve drawdown improvement does not hold under both weightings. The expected *return* is explicitly **not** the hypothesis, this is a diversification claim, not an alpha claim, and a near-zero standalone net Sharpe is an accepted, pre-registered outcome.

**How it differs from the two existing sleeves.**

| Axis | Reversal (existing) | Trend (existing) | **Multi-factor (proposed)** |
|---|---|---|---|
| Cross-section vs time-series | Cross-sectional | Time-series (12 ETFs) | Cross-sectional (S&P names) |
| Signal horizon | 5-day residual reversal, 2-day skip | 3/6/12-month blended | 252-day (12-1), 252-day beta/idio-vol |
| Momentum stance | Structurally *short* momentum (UMD beta −0.034, t −2.50) | *Long* time-series persistence | *Long* residual momentum + long low-risk |
| What it trades | Negated 5-day FF5+UMD residual | Sign of trailing ETF return | Raw cross-sectional characteristic ranks |

The keystone separation: the reversal sleeve **neutralizes** FF5+UMD and trades the short-horizon residual; this sleeve trades **long-horizon characteristics** the reversal sleeve discards, and on the momentum axis it sits on the *opposite side* of reversal's worst exposure, so it should partially hedge reversal's 2025 momentum bleed while staying near-uncorrelated to trend. It must not re-trade the 1-5 day reversal horizon (reversal owns it) and must not re-bet raw time-series persistence (trend owns it), which is why momentum is used in **residual** form and gated on a measured correlation to trend.

## 2 Final factor set (feasible-on-free-data, economically justified, orthogonal)

Resolved to the critic's recommended set: **two core factors, one conditional third.** Hard cap of 3. No value, quality/profitability, investment, accruals, earnings/PEAD (no point-in-time fundamentals on free data, see section 3); no raw 12-1 momentum (re-bets trend and reversal's fatal factor); no size as a return signal (only an ADV/price proxy, which confounds with the cost model); no 1-5 day reversal variant (reversal owns that horizon).

**Factor 1 (keystone), 12-1 residual momentum.**
Definition: for each stock, the cumulative FF5+UMD **residual** return over t−252 ... t−21 (skip the most recent ~21 trading days). Sourced directly from `residrev/residuals.py:rolling_residuals` output `resid` (TxN residual panel) via a rolling-sum-with-skip, no new estimation code, no new data.
Economics: under-reaction to firm-specific news; disposition-effect / slow-diffusion seller is the counterparty (Jegadeesh-Titman 1993; Blitz-Huij-Martens 2011). **Residual, not raw**, because raw 12-1 is the only factor with a real correlation problem to the trend sleeve; residualizing strips the market/style/UMD-beta component, leaving firm-specific drift.
Gate: |corr to trend net return| < 0.3 and the residual form must not reintroduce a dominant UMD loading, else kill the factor (do not fall back to raw).

**Factor 2, low-risk: pick exactly one of {BAB, low idiosyncratic vol}.**
- BAB: rank by trailing market beta = the stored `betas["Mkt-RF"]` loading; long low-beta / short high-beta (Frazzini-Pedersen 2014).
- Low-idio-vol: rank by `−sqrt(idio_var)` via `compute_idio_vol(idio_var)` (Ang-Hodrick-Xing-Zhang 2006).
These two express the *same* low-risk premium and are ~0.5-0.7 correlated. Include **only one**. Compute the incremental IC of the second over the first; add the second **only if material and logged as a separate trial**. Default: ship with one. Both are already produced by `rolling_residuals` at zero new data cost.
Specific check: low-beta is a duration/bond proxy that can co-move with the trend sleeve's TLT/IEF/LQD legs in risk-off, measure correlation to those legs explicitly, do not assume.

**Factor 3 (optional, conditional), 52-week-high proximity.**
Definition: `Close / trailing-252d-max(Close)` (George-Hwang 2004), a distinct anchoring/under-reaction channel ~0.5 correlated with momentum. Buildable from the cached close panel alone. Admit **only if** a logged incremental-IC check over residual momentum clears; otherwise ship the 2-factor sleeve. This is the slot to reach 3 honestly, not pad a zoo.

**Construction of the composite.** Standardize each factor with the existing `signal.py` pipeline (winsorize 1/99 -> French-12 sector-demean -> cross-sectional z-score), average the 2-3 z-scores with **fixed equal weight**, then re-z-score per date. Inverse-vol blending is shown **only** as a disclosed robustness panel, never the headline. No per-factor fitted coefficients, no per-stock parameters, no swept lookbacks (252 and 21 are pinned conventions).

## 3 Data plan (free sources; what is and is not feasible)

**Free sources already in the repo (no new dependency):**
- `cache/prices/`, daily OHLCV for the pinned ~300/503 S&P names (measured: 503 files, ~53 MB, 2018->2026). All three factors derive from this plus the existing residual engine.
- Ken French FF5+UMD daily **factor returns** (`get_ff_factors`), used only to *residualize/neutralize*, never to rank stocks.
- FRED rates; SEC EDGAR SIC -> French-12 sector map (`get_sector_map`).
Storage/compute for this sleeve is a non-issue: it adds zero new data and seconds of pandas compute.

**What is NOT feasible on free data (and is therefore excluded):**
- **No per-stock point-in-time fundamental panel.** `factors.py` exposes only French portfolio *returns*; `data.py` caches only OHLCV. yfinance `.info` is a single current snapshot; `quarterly_financials`/`quarterly_balance_sheet` return only ~5-7 quarters, stamped with period-*end* (not filing) dates, in the *current restated* vintage. Therefore **value (HML), quality/profitability (RMW), investment (CMA), accruals, and earnings/PEAD are not honestly backtestable as cross-sectional stock signals** and are dropped. Building them from `.info` or restated quarterly data would inject look-ahead and restatement bias, the exact dishonesty the project forbids. (The construction/portfolio research lenses proposed EDGAR-XBRL value/quality; that is a *separate* spike with its own >=80% coverage gate, explicitly **out of scope** here.)
- **No true market cap.** Free OHLCV has no shares outstanding; `get_shares_full` exists but is current-restated vintage. Hence size is excluded as a return signal, an ADV/price proxy is mechanically correlated with the Corwin-Schultz+Almgren illiquidity cost term, so a measured "size premium" would risk being a cost artifact.

**Survivorship-bias honesty (restate for this sleeve).** The pinned universe applies *current* S&P constituents historically. This **flatters the low-risk factor most** (delisted high-vol blow-ups the factor would have shorted/avoided are absent), so the in-sample low-risk edge is biased *upward* and is not a true point-in-time number. It is **unfixable without paid CRSP/Compustat point-in-time constituents + delisting returns.** Mitigation: disclose the direction; run a top-150 vs top-300 universe sensitivity and report Sharpe stability.

**What would need paid / point-in-time data.** A genuine value/quality/profitability sleeve, true float-adjusted size, and a survivorship-free backtest all require CRSP + Compustat (or WRDS) point-in-time data. Stated plainly as the boundary of the free-data claim.

## 4 Signal construction and neutralization (low degrees of freedom)

**Pipeline (reuse `signal.py` per factor, then composite):**
1. Build each raw factor from the existing `rolling_residuals` outputs (`resid`, `betas["Mkt-RF"]`, `idio_var`) or the cached close panel (52-wk high), all strictly rolling, past-only.
2. Per date: winsorize at 1/99 (`config.winsorize_pct`), French-12 sector-demean (`get_sector_map`), cross-sectional z-score, the exact `build_signal` sequence (steps 2-4).
3. Composite = equal-weight average of the 2-3 sector-neutral z-scores, then re-z-score per date so the alpha entering the optimizer stays O(1) (mirrors `signal.py` lines 88-91).
4. **Structural lag:** `composite.shift(1)` (no skip-gap needed, these are slow signals; the reversal sleeve's `signal_gap` exists only to dodge bid-ask bounce at the 5-day horizon). Optionally smooth with a single ~21-day span before the lag.

**Degrees of freedom held near zero.** Exactly 2-3 ex-ante factors; fixed equal weights; lookbacks pinned to existing conventions (252 momentum window, 21 skip, `factor_window`=90 for betas, 21-day idio-vol smoothing); no per-stock parameters; no new config knobs beyond at most one smoothing span and one rebalance gap. This is the explicit defense against the failure mode that broke reversal (solver-sensitive near a turnover cliff).

**Neutralization, the self-neutralization trap and its fix.** `portfolio.py:optimize_book` line 65 enforces `cp.abs(B.T @ w) <= config.beta_tol` across **all six** `config.factors`. Feeding the residual-momentum/low-risk composite through the reversal `Config` unchanged would **zero out the very tilts the sleeve harvests** (UMD/Mkt-RF), collapsing it back into a residual-reversal-like book. Two acceptable resolutions, in priority order:
- **(Recommended) Transparent rank-and-inverse-vol long/short, no optimizer.** Long top-decile / short bottom-decile of the composite, dollar-neutral, sector-neutral by construction, position-sized inverse to trailing idio-vol, gross capped to match the other sleeves. This avoids both the self-neutralization trap and the cvxpy solver fragility that hurt reversal's reproducibility, and is the most honest/reproducible build.
- **(Alternative) A dedicated `Config` via `dataclasses.replace`** whose neutrality `factors` tuple is `("Mkt-RF",)` (or `("Mkt-RF","SMB")`) only, neutralizing market (and optionally size) while *allowing* the style exposures the sleeve is paid to hold, with `lam_to` raised substantially for low turnover. Never reuse the reversal `Config` wholesale.

Add lag tests mirroring the reversal sleeve: assert the composite at t is a function only of data strictly before t.

## 5 Portfolio construction and cost integration (reuse existing machinery)

**Construction.** Default to the rank-and-inverse-vol long/short (section 4). If the optimizer route is chosen instead, reuse `optimize_book` with the dedicated Mkt-only-neutral `Config` and `w_prev` carried between rebalances. Either way: dollar-neutral, French-12 sector-neutral, market-beta-neutral, gross capped to the existing convention, per-name cap from `max_w`.

**Rebalance cadence.** Rebalance every ~21 trading days (monthly), holding weights constant between rebalances, mirror the `trend.py` `REBAL_GAP`/`ffill` pattern (lines 120-123). Slow 252-day signals plus monthly rebalancing give an order-of-magnitude lower turnover than the reversal sleeve's daily churn, which is the core cost-robustness argument.

**Costs (reused unchanged).** Run net-of-cost PnL through `costs.py`: Corwin-Schultz half-spread (`corwin_schultz_spread`) + Almgren sqrt-impact (`compute_rebalance_cost`/`build_cost_panel`), with `participation = |Δw| / AUM/ADV`, on the **same** pinned universe and AUM (`aum=25e6`, matching `build_reports.py`). Apply costs only on rebalance dates. The sleeve must clear the existing `cost_stress_test` at 2x costs comfortably.

**Disclosed cost gap (short-leg borrow).** `costs.py` models spread + impact only, **not** stock-borrow / hard-to-borrow fees. The short leg of a low-risk factor concentrates in high-beta/high-vol names that are costlier to borrow, so short-leg net returns are **optimistically biased**. Disclose; mitigate by restricting shorts to the most-liquid ADV quartile. Shared capacity: a third market-neutral equity book on the same top-N names shares the reversal sleeve's ~$10-25M capacity, it does not multiply it. Report, do not ignore.

## 6 Validation and overfitting controls

**Pre-registration first (before any backtest).** Commit to git a one-page block in `STRATEGY_SELECTION.md` stating: the hypothesis (section 1), the exact factor set and fixed weights (section 2), pinned lookbacks, the rebalance cadence, the freeze date, and the verbatim kill criteria (section 7). The commit hash is the registration timestamp.

**Fresh freeze (do not reuse the existing window as this sleeve's OOS).** Pre-register a freeze at the sleeve's build date forward. The frozen in-sample window is <= 2024-12-31 for an apples-to-apples 3-sleeve comparison, but the sleeve's *honest OOS track starts at its build date*, do not retroactively claim the existing 2025-01-01 live window (against which the other sleeves were finalized) as this sleeve's clean out-of-sample.

**Trial-count repair (the critic's must-fix).** Current `data/trials.jsonl` holds 8 non-comparable reversal trials mixing `universe_size` 300 and 1000, `sr_star = max = 0.457` vs `E[max] = norm.ppf(1−1/8) = 1.15`, the existing reversal sleeve already *fails* its own deflated-Sharpe gate. Action: **archive** that log; create a separate, comparable trials log for this sleeve; log **every** variant tried (each factor-2 choice, the optional factor-3 IC check, the equal-vs-inverse-vol panel, monthly-vs-quarterly), not just the winner. Pre-register the enumeration so the count is fixed and honest (target ~ 8-12 trials).

**Deflated Sharpe fix.** `deflated_sharpe` currently sets `sr_star = max(trials)`, for a single candidate the correct input is **this sleeve's own Sharpe**, deflated by the honest trial count `T`, with `E[max] = norm.ppf(1−1/T)`. Patch `deflated_sharpe` so SR* is the candidate's Sharpe (not the max across heterogeneous strategies). Gate: DSR >= 0.95 pass, 0.80-0.95 warn, < 0.80 kill.

**CPCV (reuse with lengthened purge/embargo).** Use `cpcv_splits` C(6,2)=15 paths, but call with **purge=21** (>= one monthly rebalance) and **embargo~0.05** (~3 months, >= signal half-life) instead of the reversal defaults (purge=5, embargo=0.01), a 252-day signal leaks across folds otherwise. Gate: `pct_positive` >= 0.75 and mean OOS Sharpe > 0.

**PBO (new, `validation.py` gap).** Add `probability_of_backtest_overfitting()` via CSCV (Bailey-Borwein-Lopez de Prado-Zhu 2017): split the trials x CPCV-subperiod performance matrix into combinatorial train/test halves, pick the in-sample-best trial per split, record its OOS rank, report PBO = fraction of splits where the IS-best lands below the OOS median. Gate: PBO <= 0.20.

**Orthogonality gate (measured, not assumed).** Regress the sleeve's daily returns on {trend, reversal, Mkt-RF, UMD} with HAC t-stats. Require |corr| < 0.3 to *each* existing sleeve (full-sample) and bounded 63-day rolling correlations; UMD must not be the dominant exposure (guard against accidentally building a second anti-momentum bet). Specifically test correlation to the trend sleeve's bond legs.

**Diversification-lift test (the headline, under BOTH weightings).** Add the sleeve to the dict passed to `combine.diversification_report({"trend":..., "reversal":..., "multifactor":...}, freeze=...)`. Report, for `scheme="equal"` **and** `scheme="risk_parity"`: the full 3x3 correlation matrix and 63-day rolling pairwise correlations; 2-sleeve vs 3-sleeve combined in-sample Sharpe and max drawdown; and the honest live 2-vs-3-sleeve track. Success bar: a drawdown reduction that holds under **both** weightings; a win under only one is reported as fragile, not as a result.

## 7 Expected outcome ranges and explicit kill criteria

**Realistic, literature-grounded expectations (not a fantasy number).** All factors are published, crowded, and decayed ~26-58% post-publication (McLean-Pontiff 2016; Harvey-Liu-Zhu 2016). Honest expectation, net of the Corwin-Schultz+Almgren model on this 300-name large-cap-only free-data universe:
- **In-sample net Sharpe ~ 0.0 to 0.2** (could be slightly negative).
- **Live net Sharpe ~ 0.0 to 0.2, possibly negative**, the sleeve may, like reversal, fail live; that is an accepted outcome to be reported honestly.
- The value proposition is **diversification / drawdown reduction** and a partial hedge of reversal's momentum-short, **not** standalone return. Low-risk/BAB underperform in growth-melt-up regimes (2020-21) and are rate-sensitive.

**Kill criteria (pre-committed; any one triggers abandonment, not gate-relaxation):**
1. |full-sample corr| >= 0.3 to trend **or** to reversal.
2. Residual momentum reintroduces |corr to trend| >= 0.3 or a dominant UMD loading (kill the factor; do **not** fall back to raw momentum).
3. Own deflated Sharpe < 0.80 on the honest trial count.
4. CPCV OOS `pct_positive` < 0.75 **or** mean OOS Sharpe <= 0.
5. PBO > 0.20.
6. Three-sleeve drawdown does not improve under **both** risk-parity and equal-weight.
7. Sleeve does not survive 2x costs (`cost_stress_test` = fail).
8. In-sample Sharpe is solver/seed/ticker-order-sensitive (re-run under ±1 ticker count and solver-order perturbation; the lesson from reversal, never publish a number that moves run-to-run).

## 8 Honest limitations

- **Price-only by necessity.** This is a price-factor sleeve, not a "full multi-factor" (FF5/q-factor) model; free data forbids the four fundamental premia. Never market it as more than it is.
- **Survivorship bias** (current constituents applied historically) flatters the low-risk factor most; biased *upward*; unfixable without paid point-in-time data. Disclosed, with a top-150/300 sensitivity as partial mitigation.
- **Decayed/crowded premia.** Expected net Sharpe is low and may be negative; the sleeve may fail live exactly as reversal did.
- **Short-leg borrow cost unmodeled**, net short-leg returns are optimistically biased; mitigated by liquid-quartile short restriction, disclosed regardless.
- **Size dropped**, not solved, only an ADV/price proxy exists, and it confounds with the cost model.
- **Shared capacity** with the reversal sleeve (~$10-25M); a third equity-neutral book does not multiply capacity.
- **Tiny live sample.** ~17 live months ~ one factor cycle; the live Sharpe has a huge standard error and is partly regime luck. Credibility rests on CPCV+DSR+PBO and literature, not the short live track.
- **yfinance data-quality tail** (split/adjust quirks; the data layer flags >50% one-day moves). A thin edge could be a few mis-adjusted names, check robustness to winsorization and to dropping flagged names.

## 9 Phased build plan (mapped to the existing repo)

**Phase 0, Pre-registration.** Write the section 1/section 2/section 7 block into `STRATEGY_SELECTION.md`; archive `data/trials.jsonl` -> `data/trials_reversal_archive.jsonl`; create a fresh trials log for this sleeve. Commit before any backtest. *(no new code)*

**Phase 1, Factor builders (`residrev/multifactor.py`, new).** ~3 small functions: `residual_momentum(resid)` (rolling-sum-with-skip over the existing residual panel), `low_risk(betas, idio_var)` (BAB or low-idio-vol; one of), optional `high_52w(close_panel)`. Plus `build_composite_alpha(...)` (reuse `signal.py` winsorize/sector-demean/z-score helpers; equal-weight + re-z-score). Reuse `rolling_residuals` output wholesale, no new data, no new residual code.

**Phase 2, Sleeve backtest (`residrev/multifactor.py` or a thin `backtest_multifactor`).** Default rank-and-inverse-vol long/short with monthly `REBAL_GAP`/`ffill` hold (mirror `trend.py`); net-of-cost via `costs.py` on the pinned universe at `aum=25e6`. Produce a daily net return `pd.Series` matching the `combine` sleeve-dict contract. (If the optimizer route is chosen, add a dedicated Mkt-only-neutral `Config` via `dataclasses.replace`.)

**Phase 3, Validation harness.** Patch `validation.py`: fix `deflated_sharpe` SR* semantics; add `probability_of_backtest_overfitting()` (CSCV); allow `cpcv_oos_sharpe`/`cpcv_splits` to take purge=21/embargo=0.05 for this sleeve. Run orthogonality regression and the full pre-trust checklist; log every trial.

**Phase 4, Combiner + report integration.** No change to `combine.py` (already takes an arbitrary sleeve dict). In `build_reports.py`: add a `multifactor_sleeve()` builder beside `reversal_sleeve()`/`trend_sleeve()`; add `"multifactor"` to the `sleeves` dict in `compute()`; the JSON/site loop over `series` already generalizes. Gate inclusion in the live report on Phase 3 passing.

**Phase 5, Tests (`tests/test_multifactor.py`, new).** Mirror existing suite: no-look-ahead lag assertions (composite[t] uses only data < t); factor-build correctness; equal-weight composite scale; sector-neutrality of the composite; reproducibility under ±1 ticker / solver-order perturbation; a `combine` 3-sleeve smoke test. Keep the 266-test suite green.

**Phase 6, Derived analytics + site wiring.** `build_derived.py` already derives tear-sheet analytics from the *frozen* `report.json` curves without recomputing, extend `BOOKS = ("reversal","trend","combined")` to include `"multifactor"` once it is in the frozen block, plus the 3-sleeve correlation matrix and the 2-vs-3-sleeve diversification delta. Add the sleeve to the dashboard pages and the methodology write-up, including the honest negative if gates fail.

**Decision point.** If any kill criterion in section 7 fires, do **not** add the sleeve to the live report, document the negative in `STRATEGY_SELECTION.md` (consistent with the project's honest-reporting discipline) and stop. The build only reaches Phase 6 inclusion after Phase 3 passes cleanly.

---

## 10 Result: built, tested, and rejected by its own gates

The sleeve was built exactly as pre-registered (residual 12-1 momentum + low idiosyncratic volatility, fixed equal weight, decile long/short with inverse-idio-vol risk weighting within each leg, monthly rebalance, net of the Corwin-Schultz + Almgren cost model) and run on the pinned S&P 500 universe, 2018 to 2026-04. It **fails its pre-registered kill criteria and is NOT shipped to the live book.**

In-sample (to 2024-12-31), net of costs:

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| In-sample net Sharpe | -0.71 | > 0 | fail |
| Keystone factor IC (residual 12-1 momentum) | -0.023 (t_HAC -3.25) | positive | fail (significant, wrong sign) |
| Deflated Sharpe (corrected; 4 trials) | 0.015 | >= 0.80 | fail |
| CPCV OOS (15 paths, purge 21 / embargo 0.05) | 7% positive, mean OOS Sharpe -0.78 | >= 75% positive | fail |
| Sharpe at 2x costs | -1.09 | > 0 | fail |
| PBO | 0.22 | <= 0.20 | warn |

All four pre-registered variants (idio-vol or BAB, with and without the 52-week-high factor) are negative in-sample (-0.48 to -0.71).

**Why (mechanism, not a bug).** The keystone factor is residual 12-1 momentum, computed on residuals that already have the UMD (momentum) factor stripped out. On the large-cap S&P 500 over this window, the leftover idiosyncratic 12-month component **mean-reverts rather than trends**: residual winners underperform (IC -0.023, t -3.25). Long residual momentum is the wrong direction here. The low-risk factors are also weak or negative (low-volatility and betting-against-beta were hurt in the 2020-21 growth melt-up and the 2025 high-beta rally). The implementation sign was verified against forward returns, so this is a genuine negative result, not a defect.

**Decision.** Per kill criteria #3, #4, and #7, the sleeve is rejected. It is not added to the combined book or the live dashboard. This is the project's second pre-registered, honestly-reported negative result. The code (`residrev/multifactor.py`), its unit tests (`tests/test_multifactor.py`), and the corrected validation machinery (a Deflated Sharpe that now uses the candidate's own Sharpe deflated by an honest trial count, plus a new PBO / CSCV check in `validation.py`) are retained for reproducibility. The corrected Deflated Sharpe also confirms the existing reversal sleeve does not clear the deflated-Sharpe bar on its archived trials, which the live methodology page had flagged as "pending audit."

**What it would take to revisit (deliberately not done here).** The finding points at a long-horizon *residual reversal* effect on large caps. Testing that is a NEW hypothesis and must be pre-registered with a fresh forward freeze; reusing this window after seeing the result would be exactly the back-fitting the project is built to avoid.