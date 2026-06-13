"""Tests for residrev/report.py — synthetic fixtures, no real backtest."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.report import generate_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    pnl: pd.Series
    gross_pnl: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs_bps: pd.Series
    factor_exposures: pd.DataFrame
    ic_series: pd.Series
    meta: dict


def _make_result(
    n: int = 252,
    pnl_mean: float = 0.001,
    cost_bps: float = 2.0,
    seed: int = 42,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    gross_pnl = pd.Series(rng.normal(pnl_mean, 0.01, n), index=dates, name="gross_pnl")
    costs = pd.Series(np.full(n, cost_bps), index=dates, name="costs_bps")
    net_pnl = gross_pnl - costs / 10_000
    factors = ["Mkt-RF", "SMB", "HML"]
    fe = pd.DataFrame(rng.normal(0, 1e-4, (n, 3)), index=dates, columns=factors)
    ic = pd.Series(rng.normal(0.03, 0.05, n), index=dates, name="ic")
    turnover = pd.Series(np.full(n, 0.1), index=dates, name="turnover")
    positions = pd.DataFrame(np.zeros((n, 10)), index=dates)

    return BacktestResult(
        pnl=net_pnl,
        gross_pnl=gross_pnl,
        positions=positions,
        turnover=turnover,
        costs_bps=costs,
        factor_exposures=fe,
        ic_series=ic,
        meta={
            "start_date": "2020-01-02",
            "end_date": "2021-01-04",
            "universe_size_mean": 487.0,
        },
    )


def _make_summary() -> dict:
    return {
        "net_sharpe": 1.23,
        "gross_sharpe": 1.85,
        "net_annual_return": 0.087,
        "gross_annual_return": 0.135,
        "max_drawdown": -0.052,
        "annual_turnover": 45.6,
        "mean_daily_ic": 0.0312,
        "ic_tstat": 3.45,
        "mean_cost_bps": 2.0,
        "total_cost_bps_pa": 504.0,
        "per_year_sharpe": {2020: 1.45, 2021: 0.98},
        "factor_exposures": {"Mkt-RF": 0.001, "SMB": -0.0002, "HML": 0.0003},
        "n_trading_days": 252,
        "start_date": "2020-01-02",
        "end_date": "2021-01-04",
    }


def _make_checklist() -> dict:
    return {
        "checks": {
            "skip_day_test": {"status": "pass", "note": "uniform weekday Sharpes"},
            "cost_stress_test": {"status": "pass", "sharpe_1x": 1.23, "sharpe_2x": 0.65},
            "factor_crash_stress": {
                "status": "pass",
                "covid": {"sharpe": 0.50, "max_dd": -0.02},
                "rate_hike": {"note": "outside backtest range"},
            },
            "deflated_sharpe": {
                "status": "pass",
                "dsr": 0.97,
                "n_trials": 5,
                "best_sharpe": 1.23,
                "e_max_sharpe": 1.10,
            },
            "cpcv_oos_sharpe": {
                "status": "pass",
                "oos_sharpes": [0.8, 1.0, 1.2, 1.1, 0.9, 1.3, 0.7, 1.05, 1.15, 0.95, 1.0, 1.1, 0.85, 1.2, 1.0],
                "mean": 1.02,
                "std": 0.16,
                "median": 1.0,
                "min": 0.7,
                "max": 1.3,
                "pct_positive": 1.0,
                "n_paths": 15,
            },
        },
        "n_pass": 5,
        "n_warn": 0,
        "n_fail": 0,
        "overall": "pass",
    }


@pytest.fixture
def result():
    return _make_result()


@pytest.fixture
def summary():
    return _make_summary()


@pytest.fixture
def checklist():
    return _make_checklist()


@pytest.fixture
def config():
    return Config()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_returns_nonempty_string(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert isinstance(md, str)
        assert len(md) > 100

    def test_all_eight_section_headers(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        for i in range(1, 9):
            assert f"## {i}." in md, f"Section {i} header missing"

    def test_net_sharpe_appears_verbatim(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert f"{summary['net_sharpe']:.2f}" in md

    def test_signal_k_in_methodology(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        methodology_start = md.index("## 4.")
        methodology_end = md.index("## 5.")
        methodology = md[methodology_start:methodology_end]
        assert str(config.signal_k) in methodology

    def test_per_year_sharpe_table(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "2020" in md
        assert "Per-year Sharpe" in md

    def test_writes_file_to_output_path(self, result, summary, checklist, config, tmp_path):
        out = str(tmp_path / "note.md")
        generate_report(result, summary, checklist, config, out)
        assert os.path.exists(out)

    def test_file_matches_returned_string(self, result, summary, checklist, config, tmp_path):
        out = str(tmp_path / "note.md")
        md = generate_report(result, summary, checklist, config, out)
        with open(out, encoding="utf-8") as f:
            on_disk = f.read()
        assert md == on_disk

    def test_handles_checklist_none(self, result, summary, config, tmp_path):
        md = generate_report(result, summary, None, config, str(tmp_path / "note.md"))
        assert isinstance(md, str)
        assert len(md) > 100

    def test_max_drawdown_formatted_as_percentage(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "%" in md
        assert f"{summary['max_drawdown']*100:.1f}%" in md

    def test_all_three_limitations(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        md_lower = md.lower()
        assert "survivorship" in md_lower
        assert "decay" in md_lower
        assert "execution" in md_lower

    def test_creates_parent_directories(self, result, summary, checklist, config, tmp_path):
        out = str(tmp_path / "nested" / "dir" / "note.md")
        generate_report(result, summary, checklist, config, out)
        assert os.path.exists(out)

    def test_cpcv_distribution_in_robustness(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "CPCV" in md
        assert "Mean OOS Sharpe" in md

    def test_cost_sensitivity_table_present(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "Cost multiplier" in md
        assert "Annual return" in md
        assert "Max drawdown" in md
        assert "Breakeven" in md

    def test_deflated_sharpe_in_robustness(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "DSR" in md
        assert "0.97" in md

    def test_universe_size_from_meta(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "487" in md

    def test_factor_crash_stress_in_robustness(self, result, summary, checklist, config, tmp_path):
        md = generate_report(result, summary, checklist, config, str(tmp_path / "note.md"))
        assert "COVID" in md

    def test_checklist_none_robustness_still_present(self, result, summary, config, tmp_path):
        md = generate_report(result, summary, None, config, str(tmp_path / "note.md"))
        assert "## 6." in md
        assert "CPCV" in md
