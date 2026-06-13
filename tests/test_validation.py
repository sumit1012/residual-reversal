"""Tests for residrev.validation — CPCV, DSR, and pre-trust checklist."""

from __future__ import annotations

import inspect
import json
import tempfile
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from residrev.validation import (
    cost_stress_test,
    cpcv_splits,
    deflated_sharpe,
    factor_crash_stress,
    run_pre_trust_checklist,
    skip_day_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dates(n: int = 500, start: str = "2019-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n, freq="B")


def _make_result(
    n: int = 500,
    start: str = "2019-01-01",
    mean: float = 0.001,
    std: float = 0.01,
    cost_bps: float = 5.0,
    seed: int = 42,
):
    """Build a minimal BacktestResult-like object for testing."""
    rng = np.random.RandomState(seed)
    dates = _make_dates(n, start)
    gross = pd.Series(rng.normal(mean, std, n), index=dates, name="gross_pnl")
    costs = pd.Series(cost_bps, index=dates, name="costs_bps")
    pnl = gross - costs / 10_000
    pnl.name = "pnl"

    @dataclass
    class _FakeResult:
        pnl: pd.Series
        gross_pnl: pd.Series
        costs_bps: pd.Series
        positions: pd.DataFrame = None
        turnover: pd.Series = None
        factor_exposures: pd.DataFrame = None
        ic_series: pd.Series = None
        meta: dict = None

    return _FakeResult(pnl=pnl, gross_pnl=gross, costs_bps=costs)


def _make_config(**overrides):
    """Build a minimal Config-like object for testing."""

    @dataclass
    class _FakeConfig:
        cpcv_n_groups: int = 6
        cpcv_k_test: int = 2
        cpcv_embargo: float = 0.01
        trials_log: str = "data/trials.jsonl"

    return _FakeConfig(**overrides)


def _write_trials_log(path: str, sharpes: list[float]) -> None:
    with open(path, "w") as f:
        for s in sharpes:
            f.write(json.dumps({"net_sharpe": s}) + "\n")


# ---------------------------------------------------------------------------
# CPCV splits
# ---------------------------------------------------------------------------


class TestCpcvSplits:
    def test_yields_correct_number_of_splits(self):
        dates = _make_dates(300)
        splits = list(cpcv_splits(dates, n_groups=6, k_test=2))
        assert len(splits) == comb(6, 2)

    def test_train_test_disjoint(self):
        dates = _make_dates(300)
        for train, test in cpcv_splits(dates, n_groups=6, k_test=2):
            overlap = set(train) & set(test)
            assert len(overlap) == 0, "Train and test must be disjoint"

    def test_purge_removes_dates_before_test_group(self):
        dates = _make_dates(300)
        purge = 5
        n_groups = 6
        T = len(dates)
        group_size = T // n_groups

        for train, test in cpcv_splits(dates, n_groups=n_groups, k_test=2, purge=purge):
            train_set = set(train)
            test_indices = sorted([dates.get_loc(d) for d in test])
            # Find the start of each contiguous test block
            blocks = []
            block_start = test_indices[0]
            for i in range(1, len(test_indices)):
                if test_indices[i] != test_indices[i - 1] + 1:
                    blocks.append(block_start)
                    block_start = test_indices[i]
            blocks.append(block_start)

            for block_start_idx in blocks:
                for offset in range(1, purge + 1):
                    purge_idx = block_start_idx - offset
                    if purge_idx >= 0:
                        assert dates[purge_idx] not in train_set, (
                            f"Date at index {purge_idx} should be purged "
                            f"(within {purge} days before test block at {block_start_idx})"
                        )

    def test_embargo_removes_dates_after_test_group(self):
        dates = _make_dates(300)
        embargo = 0.01
        T = len(dates)
        embargo_days = max(1, int(T * embargo))

        for train, test in cpcv_splits(dates, n_groups=6, k_test=2, embargo=embargo):
            train_set = set(train)
            test_indices = sorted([dates.get_loc(d) for d in test])
            # Find the end of each contiguous test block
            blocks_end = []
            for i in range(len(test_indices)):
                if i == len(test_indices) - 1 or test_indices[i + 1] != test_indices[i] + 1:
                    blocks_end.append(test_indices[i])

            for block_end_idx in blocks_end:
                for offset in range(1, embargo_days + 1):
                    embargo_idx = block_end_idx + offset
                    if embargo_idx < T:
                        assert dates[embargo_idx] not in train_set, (
                            f"Date at index {embargo_idx} should be embargoed "
                            f"(within {embargo_days} days after test block ending at {block_end_idx})"
                        )

    def test_all_dates_appear_in_at_least_one_test_split(self):
        dates = _make_dates(300)
        all_test_dates = set()
        for _, test in cpcv_splits(dates, n_groups=6, k_test=2):
            all_test_dates.update(test)
        assert set(dates) == all_test_dates, "Every date must appear in at least one test split"

    def test_is_generator(self):
        assert inspect.isgeneratorfunction(cpcv_splits)


# ---------------------------------------------------------------------------
# Skip-day test
# ---------------------------------------------------------------------------


class TestSkipDayTest:
    def test_pass_when_uniform_weekday_sharpes(self):
        result = _make_result(n=1000, mean=0.001, std=0.01, seed=42)
        out = skip_day_test(result)
        assert out["status"] == "pass"
        assert "by_weekday" in out
        assert len(out["by_weekday"]) == 5

    def test_warn_when_one_weekday_differs(self):
        dates = _make_dates(1000)
        rng = np.random.RandomState(99)
        pnl_vals = rng.normal(0.0005, 0.01, len(dates))
        # Make Monday returns much higher
        for i, d in enumerate(dates):
            if d.weekday() == 0:
                pnl_vals[i] += 0.03
        pnl = pd.Series(pnl_vals, index=dates, name="pnl")
        gross = pnl.copy()
        gross.name = "gross_pnl"
        costs = pd.Series(0.0, index=dates, name="costs_bps")

        @dataclass
        class _R:
            pnl: pd.Series
            gross_pnl: pd.Series
            costs_bps: pd.Series

        result = _R(pnl=pnl, gross_pnl=gross, costs_bps=costs)
        out = skip_day_test(result)
        assert out["status"] in ("warn", "fail")


# ---------------------------------------------------------------------------
# Cost stress test
# ---------------------------------------------------------------------------


class TestCostStressTest:
    def test_fail_when_marginal_strategy(self):
        # Very small mean, high costs -> 2x costs should push Sharpe negative
        result = _make_result(n=500, mean=0.0001, std=0.01, cost_bps=50.0, seed=77)
        out = cost_stress_test(result)
        assert out["status"] == "fail"
        assert out["sharpe_2x"] < 0

    def test_pass_when_robust_strategy(self):
        result = _make_result(n=500, mean=0.005, std=0.01, cost_bps=1.0, seed=42)
        out = cost_stress_test(result)
        assert out["status"] == "pass"
        assert out["sharpe_2x"] > 0.3


# ---------------------------------------------------------------------------
# Factor crash stress
# ---------------------------------------------------------------------------


class TestFactorCrashStress:
    def test_skip_note_for_out_of_range(self):
        # Backtest only covers 2023 — both stress periods are outside range
        result = _make_result(n=250, start="2023-01-01", seed=42)
        out = factor_crash_stress(result)
        for period in ("covid", "rate_hike"):
            assert "note" in out[period]
        assert out["status"] == "pass"

    def test_pass_with_data_covering_periods(self):
        # 2019 to 2023 — covers both stress periods
        result = _make_result(n=1200, start="2019-01-01", mean=0.001, std=0.01, seed=42)
        out = factor_crash_stress(result)
        assert out["status"] in ("pass", "warn", "fail")
        assert "sharpe" in out["covid"]


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------


class TestDeflatedSharpe:
    def test_skip_when_no_file(self):
        result = _make_result()
        out = deflated_sharpe("/nonexistent/path.jsonl", result)
        assert out["status"] == "skip"

    def test_dsr_in_zero_one(self):
        result = _make_result(n=500, mean=0.002, std=0.01, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in [0.5, 0.8, 1.0, 1.2, 0.3]:
                f.write(json.dumps({"net_sharpe": s}) + "\n")
            path = f.name

        out = deflated_sharpe(path, result)
        assert out["status"] != "skip"
        assert 0.0 <= out["dsr"] <= 1.0
        Path(path).unlink()

    def test_dsr_higher_with_fewer_trials(self):
        result = _make_result(n=500, mean=0.002, std=0.01, seed=42)

        # Few trials
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_trials_log(f.name, [0.8, 1.0])
            path_few = f.name

        # Many trials — same best Sharpe but more multiple-testing bias
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            _write_trials_log(f.name, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1] * 5)
            path_many = f.name

        out_few = deflated_sharpe(path_few, result)
        out_many = deflated_sharpe(path_many, result)

        assert out_few["dsr"] > out_many["dsr"], (
            f"DSR with fewer trials ({out_few['dsr']:.3f}) should exceed "
            f"DSR with many trials ({out_many['dsr']:.3f})"
        )
        Path(path_few).unlink()
        Path(path_many).unlink()

    def test_skip_with_one_trial(self):
        result = _make_result()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"net_sharpe": 1.0}) + "\n")
            path = f.name

        out = deflated_sharpe(path, result)
        assert out["status"] == "skip"
        Path(path).unlink()


# ---------------------------------------------------------------------------
# run_pre_trust_checklist
# ---------------------------------------------------------------------------


class TestRunPreTrustChecklist:
    def test_returns_correct_overall_status(self):
        result = _make_result(n=500, mean=0.005, std=0.01, cost_bps=1.0, seed=42)
        config = _make_config(trials_log="/nonexistent/path.jsonl")
        out = run_pre_trust_checklist(result, config)
        assert out["overall"] in ("pass", "warn", "fail")
        assert "checks" in out

    def test_counts_sum_to_total(self):
        result = _make_result(n=500, mean=0.005, std=0.01, cost_bps=1.0, seed=42)
        config = _make_config(trials_log="/nonexistent/path.jsonl")
        out = run_pre_trust_checklist(result, config)
        total_checks = len(out["checks"])
        counted = out["n_pass"] + out["n_warn"] + out["n_fail"]
        # skip counts are not in the three main buckets
        n_skip = sum(1 for c in out["checks"].values() if c["status"] == "skip")
        assert counted + n_skip == total_checks

    def test_fail_overrides_warn(self):
        # Marginal strategy — cost stress should fail
        result = _make_result(n=500, mean=0.0001, std=0.01, cost_bps=50.0, seed=77)
        config = _make_config(trials_log="/nonexistent/path.jsonl")
        out = run_pre_trust_checklist(result, config)
        assert out["overall"] == "fail"
