"""End-to-end output verification for a residrev run (Step 19, Phase 4).

Usage:
    python verify_outputs.py [RUN_ID]

If RUN_ID is omitted, the most recently modified directory under data/results/
is used. Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import pandas as pd


def resolve_run_id(argv: list[str]) -> str:
    if len(argv) > 1:
        return argv[1]
    dirs = [d for d in glob.glob("data/results/*") if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("No run directories found under data/results/")
    latest = max(dirs, key=os.path.getmtime)
    return os.path.basename(latest)


run_id = resolve_run_id(sys.argv)
base = f"data/results/{run_id}"
print(f"Verifying run: {run_id}\n")

# 1. All expected output files exist
expected_files = [
    f"{base}/summary.json",
    f"{base}/pnl.parquet",
    f"{base}/positions.parquet",
    f"{base}/factor_exposures.parquet",
    f"{base}/ic_series.parquet",
    f"{base}/research_note.md",
    f"{base}/report.html",
    f"{base}/run.log",
]
for f in expected_files:
    assert os.path.exists(f), f"MISSING: {f}"
print("1. All output files present [OK]")

# 2. summary.json — no NaNs, key metrics are real numbers
with open(f"{base}/summary.json") as fh:
    s = json.load(fh)

# NOTE: real schema uses 'mean_daily_ic' (the Step 19 template said 'mean_ic').
for key in ["gross_sharpe", "net_sharpe", "max_drawdown", "annual_turnover", "mean_daily_ic"]:
    val = s[key]
    assert val is not None, f"{key} is None"
    assert not math.isnan(val), f"{key} is NaN"
    assert not math.isinf(val), f"{key} is Inf"
print("2. summary.json metrics are real numbers [OK]")
print(json.dumps({k: round(v, 4) for k, v in s.items() if isinstance(v, float)}, indent=2))

# 3. pnl.parquet — non-empty, no all-NaN columns
pnl = pd.read_parquet(f"{base}/pnl.parquet")
assert len(pnl) > 200, f"PnL has only {len(pnl)} rows — suspiciously short"
assert not pnl["net_pnl"].isna().all(), "net_pnl is all NaN"
print(f"3. PnL: {len(pnl)} rows, net cumulative: {pnl['net_pnl'].sum():.6f} [OK]")

# 4. positions.parquet — weights sum near zero (dollar-neutral) on most dates
positions = pd.read_parquet(f"{base}/positions.parquet")
weight_sums = positions.sum(axis=1)
dollar_neutral_frac = (weight_sums.abs() < 0.05).mean()
assert dollar_neutral_frac > 0.9, "Weights are not dollar-neutral on >90% of days"
# Guard against a degenerate (empty-book) portfolio that passes neutrality trivially.
gross = positions.abs().sum(axis=1)
assert gross.mean() > 0.1, f"Mean gross leverage {gross.mean():.4f} — book is effectively empty"
print(f"4. Dollar-neutral on {dollar_neutral_frac*100:.1f}% of days; mean gross={gross.mean():.3f} [OK]")

# 5. ic_series.parquet — mean IC is non-zero
ic = pd.read_parquet(f"{base}/ic_series.parquet")
mean_ic = ic.iloc[:, 0].mean()
assert abs(mean_ic) > 1e-5, f"Mean IC is effectively zero: {mean_ic}"
print(f"5. Mean IC: {mean_ic:.4f} [OK]")

# 6. research_note.md — required sections present
with open(f"{base}/research_note.md", encoding="utf-8") as fh:
    note = fh.read()
required_sections = ["Hypothesis", "Data", "Methodology", "Results", "Robustness", "Limitations"]
for section in required_sections:
    assert section in note, f"research_note.md missing section: {section}"
print("6. research_note.md has all required sections [OK]")

# 7. report.html — non-empty, contains Chart.js
with open(f"{base}/report.html", encoding="utf-8") as fh:
    html = fh.read()
assert len(html) > 10_000, f"report.html is suspiciously small ({len(html)} bytes)"
assert "chart.js" in html.lower() or "Chart" in html, "report.html missing Chart.js"
print(f"7. report.html: {len(html):,} bytes [OK]")

print("\n=== ALL VERIFICATION CHECKS PASSED ===")
