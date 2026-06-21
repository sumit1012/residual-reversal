# How to Run the Residual Reversal Strategy

---

## Prerequisites

- Python 3.11+
- Node.js (only needed if you reinstall gsd)
- Git
- ~2 GB disk space (price cache)
- Internet connection for first run (data fetching)

---

## 1. Setup (one-time)

```bash
# Navigate to the project
cd "C:\Users\thakk\Documents\JOb search\wallstreetquants\Project_Residual_Reversal"

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Verify dependencies
pip install -r requirements.txt
```

Confirm the environment is clean:
```bash
python -c "import cvxpy, yfinance, pandas, statsmodels; print('All imports OK')"
```

---

## 2. Quickest Run (S&P 500, all defaults)

```bash
python -m residrev.run
```

This fetches the current S&P 500 constituent list from Wikipedia, pulls 7 years of price data, and runs the full pipeline. **Expect 30–90 minutes on first run** (price download + rolling OLS on 500+ tickers × 1500 dates). Subsequent runs are much faster because prices are cached in `cache/prices/`.

---

## 3. Run With All Outputs

```bash
python -m residrev.run --eda --report
```

Flags:
- `--eda` — runs the EDA module and saves exhibits to `data/results/<run_id>/eda/`
- `--report` — generates the research note at `data/results/<run_id>/research_note.md`

---

## 4. All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--start-date` | `2018-01-01` | Backtest start date (YYYY-MM-DD) |
| `--end-date` | `2024-12-31` | Backtest end date |
| `--universe-size` | `1000` | Target number of liquid stocks per day |
| `--signal-k` | `5` | Reversal lookback in days |
| `--gamma` | `5.0` | Risk aversion in the optimizer |
| `--lam-to` | `10.0` | Turnover penalty coefficient |
| `--tickers-file` | *(S&P 500)* | Path to a newline-separated list of tickers |
| `--eda` | off | Run EDA and save exhibits |
| `--report` | off | Generate the research note |
| `--no-checklist` | off | Skip the pre-trust validation checklist |

**Example — shorter backtest with custom tickers:**
```bash
python -m residrev.run \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --signal-k 3 \
  --tickers-file my_tickers.txt \
  --report
```

**Example — quick smoke test with a small date range:**
```bash
python -m residrev.run \
  --start-date 2022-01-01 \
  --end-date 2023-12-31 \
  --no-checklist
```

---

## 5. Using a Custom Ticker File

Create a plain text file, one ticker per line:

```
AAPL
MSFT
GOOGL
AMZN
NVDA
...
```

Pass it with `--tickers-file path/to/tickers.txt`. Tickers are automatically uppercased and deduplicated. yfinance format: use `-` instead of `.` for class shares (e.g., `BRK-B` not `BRK.B`).

---

## 6. What Happens During a Run

```
[INFO] Run ID: 20240615_143022
[INFO] Fetching 503 tickers from Wikipedia S&P 500 list
[INFO] Pulling prices... (this takes ~10-30 min on first run)
[INFO] Computing ADV and universe membership
[INFO] Downloading Ken French FF5+UMD factors
[INFO] Fetching sector map from SEC EDGAR
[INFO] Running rolling OLS residualization...
[INFO] Progress: 50/1500 dates complete
[INFO] Progress: 100/1500 dates complete
...
[INFO] Building reversal signal
[INFO] Running backtest optimization loop...
[INFO] Backtest complete: 1487 trading days simulated
[INFO] Saving outputs to data/results/20240615_143022/
[INFO] Trial logged to data/trials.jsonl
[INFO] Pre-trust checklist: 4 PASS  1 WARN  0 FAIL
```

Progress is logged every 50 dates. If it looks frozen, it's likely in the OLS loop (the slowest step, ~20–40 min).

---

## 7. Outputs

All outputs land in `data/results/<run_id>/`:

| File | Contents |
|------|----------|
| `summary.json` | All key metrics: Sharpe, drawdown, IC, turnover, costs, per-year Sharpe |
| `pnl.parquet` | Daily gross PnL, net PnL, and cost in bps |
| `positions.parquet` | Full (dates × tickers) weight panel |
| `factor_exposures.parquet` | Net factor exposure per date |
| `ic_series.parquet` | Daily cross-sectional IC |
| `research_note.md` | Structured research note (if `--report` used) |
| `eda/` | EDA exhibits as CSVs (if `--eda` used) |

Trial parameters and summary are also appended to `data/trials.jsonl` — one JSON line per run — so you can compare runs across parameter sweeps.

---

## 8. Reading the Results

Load the summary:
```python
import json
with open("data/results/<run_id>/summary.json") as f:
    s = json.load(f)
print(f"Net Sharpe: {s['net_sharpe']:.2f}")
print(f"Gross Sharpe: {s['gross_sharpe']:.2f}")
print(f"Max Drawdown: {s['max_drawdown']*100:.1f}%")
print(f"Annual Turnover: {s['annual_turnover']*100:.0f}%")
```

Load the PnL and plot cumulative returns:
```python
import pandas as pd
import matplotlib.pyplot as plt

pnl = pd.read_parquet("data/results/<run_id>/pnl.parquet")
pnl[["gross_pnl", "net_pnl"]].cumsum().plot(title="Cumulative PnL")
plt.show()
```

Load all trials and compare Sharpes across parameter sweeps:
```python
import pandas as pd

trials = pd.read_json("data/trials.jsonl", lines=True)
print(trials[["run_id", "summary"]].assign(
    net_sharpe=lambda df: df["summary"].apply(lambda s: s["net_sharpe"])
).sort_values("net_sharpe", ascending=False))
```

---

## 9. Running the Tests

```bash
# All 275 tests
pytest tests/ -v

# Single module
pytest tests/test_signal.py -v

# With coverage report
pytest tests/ --cov=residrev --cov-report=term-missing

# Fast check (no slow integration-adjacent tests)
pytest tests/ -v -x  # stop on first failure
```

---

## 10. Common Issues

**"yfinance returns empty DataFrame for some tickers"**
Normal. The Stooq fallback fires automatically. If both fail, the ticker is skipped and logged. Check the logs for which tickers were dropped.

**"Optimization failed — returning zero weights"**
Happens occasionally on dates where the problem is infeasible (too-tight constraints relative to the universe size). The backtest continues; that date gets zero PnL and zero cost. If it happens frequently, try increasing `--universe-size` or reducing `--gamma`.

**"EDGAR rate limit / timeout in get_sector_map"**
SEC EDGAR allows 10 requests/second. The code sleeps 0.1s between calls. If you still hit limits, run again — the sector map is cached to `data/sector_map.json` after the first successful fetch.

**"Backtest seems frozen"**
Check that progress logs are printing every 50 dates. The rolling OLS loop takes 20–40 minutes. If no log has appeared in 10+ minutes, the process may have OOM-killed — try reducing `--universe-size` to 300 for a faster test run.

**Price cache is stale (missing recent dates)**
The cache validates date coverage. If your `--end-date` is beyond what's cached, it re-fetches the missing tickers automatically.

---

## 11. Re-running With New Parameters (Parameter Sweep)

Each run appends to `data/trials.jsonl` so nothing is overwritten. To sweep signal lookback:

```bash
for k in 3 5 7 10; do
  python -m residrev.run --signal-k $k --no-checklist
done
```

Then read `data/trials.jsonl` to compare (see Section 8). The Deflated Sharpe Ratio in the checklist automatically accounts for the number of trials logged — so run the checklist on the best configuration after the sweep.

---

*Project: Factor-Neutral Residual Reversal | residrev/ package | Python 3.11+*
