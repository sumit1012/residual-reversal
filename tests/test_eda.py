"""Tests for residrev.eda — all synthetic data, no network or disk I/O required."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from residrev.eda import (
    compute_adv_stats,
    compute_ic_at_k,
    compute_return_stats,
    compute_universe_turnover,
    run_eda,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(tickers: list[str], dates: pd.DatetimeIndex, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    prices = {}
    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        volume = rng.integers(1_000_000, 5_000_000, len(dates)).astype(float)
        prices[ticker] = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
    return prices


def _make_membership(tickers: list[str], dates: pd.DatetimeIndex, all_true: bool = True) -> pd.DataFrame:
    return pd.DataFrame(all_true, index=dates, columns=tickers)


def _make_adv(tickers: list[str], dates: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.uniform(1e6, 1e8, size=(len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# compute_return_stats
# ---------------------------------------------------------------------------

class TestComputeReturnStats:
    def test_output_columns(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers = [f"T{i}" for i in range(60)]
        prices = _make_prices(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_return_stats(prices, membership)
        expected_cols = {"mean", "std", "skew", "kurt", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "thin_cross_section"}
        assert expected_cols.issubset(set(result.columns))

    def test_output_shape(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        tickers = [f"T{i}" for i in range(60)]
        prices = _make_prices(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_return_stats(prices, membership)
        # First row has NaN returns (pct_change), still included
        assert len(result) == len(dates)

    def test_membership_filtering(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers_all = [f"T{i}" for i in range(80)]
        member_tickers = tickers_all[:60]
        non_member_tickers = tickers_all[60:]

        prices = _make_prices(tickers_all, dates, seed=5)
        # Give non-members extreme returns so they'd dominate the mean
        for t in non_member_tickers:
            prices[t]["Close"] = prices[t]["Close"] * 1000

        membership = pd.DataFrame(False, index=dates, columns=tickers_all)
        membership[member_tickers] = True

        result_filtered = compute_return_stats(prices, membership)

        # Compare vs result with all members — should be different if filtering works
        membership_all = _make_membership(tickers_all, dates)
        result_all = compute_return_stats(prices, membership_all)

        # Mean should differ because non-members have extreme values
        assert not result_filtered["mean"].dropna().equals(result_all["mean"].dropna())

    def test_thin_cross_section_flag(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        # Only 10 tickers → always thin
        tickers = [f"T{i}" for i in range(10)]
        prices = _make_prices(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_return_stats(prices, membership)
        # All rows should be flagged thin (10 < 50)
        assert result["thin_cross_section"].all()

    def test_not_thin_when_enough_members(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        tickers = [f"T{i}" for i in range(60)]
        prices = _make_prices(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_return_stats(prices, membership)
        # Should have at least some non-thin rows (60 >= 50)
        assert not result["thin_cross_section"].all()


# ---------------------------------------------------------------------------
# compute_adv_stats
# ---------------------------------------------------------------------------

class TestComputeAdvStats:
    def test_output_shape(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        tickers = [f"T{i}" for i in range(50)]
        adv = _make_adv(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_adv_stats(adv, membership)
        assert len(result) == len(dates)

    def test_output_columns(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        tickers = [f"T{i}" for i in range(30)]
        adv = _make_adv(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_adv_stats(adv, membership)
        assert {"mean", "std", "p25", "p50", "p75", "min", "max"}.issubset(set(result.columns))

    def test_p50_between_min_and_max(self):
        dates = pd.date_range("2023-01-01", periods=15, freq="B")
        tickers = [f"T{i}" for i in range(40)]
        adv = _make_adv(tickers, dates)
        membership = _make_membership(tickers, dates)
        result = compute_adv_stats(adv, membership).dropna()
        assert (result["p50"] >= result["min"]).all()
        assert (result["p50"] <= result["max"]).all()


# ---------------------------------------------------------------------------
# compute_ic_at_k
# ---------------------------------------------------------------------------

class TestComputeIcAtK:
    def _make_signal_fwd(self, n_dates: int = 60, n_tickers: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        signal = pd.DataFrame(rng.normal(size=(n_dates, n_tickers)), index=dates, columns=tickers)
        fwd = pd.DataFrame(rng.normal(size=(n_dates, n_tickers)), index=dates, columns=tickers)
        return signal, fwd

    def test_one_row_per_k(self):
        signal, fwd = self._make_signal_fwd()
        k_values = [1, 3, 5, 10, 21]
        result = compute_ic_at_k(signal, fwd, k_values=k_values)
        assert list(result.index) == k_values

    def test_t_stat_sign_matches_mean_ic(self):
        signal, fwd = self._make_signal_fwd(seed=42)
        result = compute_ic_at_k(signal, fwd, k_values=[1, 5]).dropna()
        for _, row in result.iterrows():
            if not math.isnan(row["t_stat"]) and not math.isnan(row["mean_ic"]):
                assert math.copysign(1, row["t_stat"]) == math.copysign(1, row["mean_ic"])

    def test_perfect_rank_correlation(self):
        """IC should be near 1.0 when fwd_returns[i+k] == signal[i] for all i."""
        n_dates, n_tickers = 60, 50
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        rng = np.random.default_rng(7)
        data = pd.DataFrame(rng.normal(size=(n_dates, n_tickers)), index=dates, columns=tickers)
        # shift(1) means fwd.iloc[i+1] == data.iloc[i] → perfect correlation at k=1
        fwd = data.shift(1)
        result = compute_ic_at_k(data, fwd, k_values=[1])
        assert result.loc[1, "mean_ic"] > 0.95

    def test_default_k_values(self):
        signal, fwd = self._make_signal_fwd()
        result = compute_ic_at_k(signal, fwd)
        assert list(result.index) == [1, 3, 5, 10, 21]

    def test_n_obs_positive(self):
        signal, fwd = self._make_signal_fwd(n_dates=30)
        result = compute_ic_at_k(signal, fwd, k_values=[1, 5])
        assert (result["n_obs"] > 0).all()


# ---------------------------------------------------------------------------
# compute_universe_turnover
# ---------------------------------------------------------------------------

class TestComputeUniverseTurnover:
    def test_zero_on_no_change_dates(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        tickers = ["A", "B", "C", "D"]
        membership = pd.DataFrame(True, index=dates, columns=tickers)
        result = compute_universe_turnover(membership)
        # All dates after first should be 0 (no changes)
        assert (result.iloc[1:] == 0).all()

    def test_first_row_is_nan(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        tickers = ["A", "B", "C"]
        membership = pd.DataFrame(True, index=dates, columns=tickers)
        result = compute_universe_turnover(membership)
        assert math.isnan(result.iloc[0])

    def test_correct_count_on_known_change(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        tickers = ["A", "B", "C", "D"]
        # Day 0: A,B,C members; Day 1: B,C,D members (A exits, D enters)
        membership = pd.DataFrame(
            [
                [True, True, True, False],
                [False, True, True, True],
                [False, True, True, True],
            ],
            index=dates,
            columns=tickers,
        )
        result = compute_universe_turnover(membership)
        # Day 1: 1 entry (D) + 1 exit (A) = 2 changes; size = 3 → turnover = 2/3
        assert abs(result.iloc[1] - 2 / 3) < 1e-9
        # Day 2: no changes → 0
        assert result.iloc[2] == 0.0


# ---------------------------------------------------------------------------
# run_eda
# ---------------------------------------------------------------------------

class TestRunEda:
    def _setup(self, tmp_path):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers = [f"T{i}" for i in range(60)]
        prices = _make_prices(tickers, dates)
        membership = _make_membership(tickers, dates)
        adv = _make_adv(tickers, dates)
        return prices, membership, adv, dates, tickers, str(tmp_path)

    def test_returns_all_keys(self, tmp_path):
        prices, membership, adv, _, _, out = self._setup(tmp_path)
        result = run_eda(prices, membership, adv, output_dir=out)
        assert set(result.keys()) == {"return_stats", "adv_stats", "universe_turnover", "ic_profile"}

    def test_saves_csvs_to_output_dir(self, tmp_path):
        prices, membership, adv, _, _, out = self._setup(tmp_path)
        run_eda(prices, membership, adv, output_dir=out)
        import os
        files = os.listdir(out)
        assert "return_stats.csv" in files
        assert "adv_stats.csv" in files
        assert "universe_turnover.csv" in files

    def test_ic_profile_none_when_signal_missing(self, tmp_path):
        prices, membership, adv, _, _, out = self._setup(tmp_path)
        result = run_eda(prices, membership, adv, output_dir=out)
        assert result["ic_profile"] is None

    def test_ic_profile_computed_when_provided(self, tmp_path):
        prices, membership, adv, dates, tickers, out = self._setup(tmp_path)
        rng = np.random.default_rng(3)
        signal = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
        fwd = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
        result = run_eda(prices, membership, adv, signal=signal, fwd_returns=fwd, output_dir=out)
        assert result["ic_profile"] is not None
        assert len(result["ic_profile"]) == 5  # default k_values

    def test_ic_csv_saved_when_provided(self, tmp_path):
        prices, membership, adv, dates, tickers, out = self._setup(tmp_path)
        rng = np.random.default_rng(3)
        signal = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
        fwd = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
        run_eda(prices, membership, adv, signal=signal, fwd_returns=fwd, output_dir=out)
        import os
        assert "ic_profile.csv" in os.listdir(out)
