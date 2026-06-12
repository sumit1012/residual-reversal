"""Tests for residrev.conditioning."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.conditioning import compute_amihud, compute_vix_regime, get_vix, ic_by_bucket
from residrev.signal import compute_ic, compute_ic_tstat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return Config()


def _make_prices(n_tickers: int = 5, n_days: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    prices = {}
    for t in tickers:
        close = 100 * np.exp(rng.normal(0, 0.01, n_days).cumsum())
        volume = rng.integers(100_000, 5_000_000, n_days).astype(float)
        prices[t] = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
    return prices, tickers, dates


def _make_universe(tickers, dates, all_true: bool = True):
    univ = pd.DataFrame(True, index=dates, columns=tickers)
    if not all_true:
        univ[tickers[-2:]] = False
    return univ


# ---------------------------------------------------------------------------
# compute_amihud
# ---------------------------------------------------------------------------


class TestComputeAmihud:
    def test_output_shape_matches_universe(self, config):
        prices, tickers, dates = _make_prices()
        universe = _make_universe(tickers, dates)
        q, z = compute_amihud(prices, universe, config)
        assert q.shape == universe.shape
        assert z.shape == universe.shape

    def test_non_universe_members_are_nan(self, config):
        prices, tickers, dates = _make_prices()
        universe = _make_universe(tickers, dates, all_true=False)
        q, z = compute_amihud(prices, universe, config)
        excluded = tickers[-2:]
        assert q[excluded].isna().all().all(), "excluded tickers should be NaN in quintile"
        assert z[excluded].isna().all().all(), "excluded tickers should be NaN in z_score"

    def test_quintile_values_in_valid_range(self, config):
        prices, tickers, dates = _make_prices(n_tickers=10, n_days=60)
        universe = _make_universe(tickers, dates)
        q, _ = compute_amihud(prices, universe, config)
        valid = q.stack().dropna()
        assert len(valid) > 0
        assert valid.between(1, config.n_illiq_buckets).all(), (
            f"quintile values out of range: {sorted(valid.unique())}"
        )

    def test_zscore_approximately_normalized(self, config):
        prices, tickers, dates = _make_prices(n_tickers=20, n_days=60)
        universe = _make_universe(tickers, dates)
        _, z = compute_amihud(prices, universe, config)
        last_row = z.iloc[-1].dropna()
        assert len(last_row) >= 5, "need valid stocks for normalization check"
        assert abs(last_row.mean()) < 0.5, f"z-score mean {last_row.mean():.3f} not near 0"
        assert 0.3 < last_row.std() < 2.0, f"z-score std {last_row.std():.3f} unexpected"

    def test_zero_dollar_volume_produces_nan_not_inf(self, config):
        prices, tickers, dates = _make_prices(n_tickers=3, n_days=30)
        prices[tickers[0]].loc[prices[tickers[0]].index[5:10], "Volume"] = 0.0
        universe = _make_universe(tickers, dates)
        q, z = compute_amihud(prices, universe, config)
        assert not np.isinf(q.values[~np.isnan(q.values)]).any(), "inf in quintile"
        assert not np.isinf(z.values[~np.isnan(z.values)]).any(), "inf in z_score"

    def test_empty_prices_returns_all_nan(self, config):
        _, tickers, dates = _make_prices()
        universe = _make_universe(tickers, dates)
        q, z = compute_amihud({}, universe, config)
        assert q.isna().all().all()
        assert z.isna().all().all()
        assert q.shape == universe.shape
        assert z.shape == universe.shape


# ---------------------------------------------------------------------------
# get_vix
# ---------------------------------------------------------------------------


class TestGetVix:
    def test_loads_from_parquet_cache(self, tmp_path):
        """Returns cached VIX without hitting any network source."""
        cfg = Config(data_dir=str(tmp_path))
        fake = pd.Series(
            [15.0, 20.0, 25.0],
            index=pd.date_range("2020-01-02", periods=3, freq="B"),
            name="VIX",
        )
        (tmp_path / "vix.parquet").parent.mkdir(parents=True, exist_ok=True)
        fake.to_frame().to_parquet(tmp_path / "vix.parquet")

        vix = get_vix(cfg)
        assert isinstance(vix, pd.Series)
        assert vix.name == "VIX"
        assert len(vix) == 3

    def test_falls_back_to_yfinance_when_fred_fails(self, tmp_path):
        """FRED error triggers yfinance fallback and result is still a valid Series."""
        cfg = Config(data_dir=str(tmp_path))
        fake_close = pd.Series(
            [18.0, 22.0],
            index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]),
        )
        fake_df = pd.DataFrame({"Close": fake_close})

        with (
            patch("residrev.conditioning.pdr") as mock_pdr,
            patch("residrev.conditioning.yf") as mock_yf,
        ):
            mock_pdr.get_data_fred.side_effect = Exception("FRED unavailable")
            mock_yf.download.return_value = fake_df
            vix = get_vix(cfg)

        assert isinstance(vix, pd.Series)
        assert vix.name == "VIX"
        assert len(vix) == 2

    def test_returns_tz_naive_index(self, tmp_path):
        """Index must be tz-naive regardless of source timezone."""
        cfg = Config(data_dir=str(tmp_path))
        # Simulate yfinance returning a tz-aware index
        tz_idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"]).tz_localize("UTC")
        fake_close = pd.Series([18.0, 22.0], index=tz_idx)
        fake_df = pd.DataFrame({"Close": fake_close})

        with (
            patch("residrev.conditioning.pdr") as mock_pdr,
            patch("residrev.conditioning.yf") as mock_yf,
        ):
            mock_pdr.get_data_fred.side_effect = Exception("FRED down")
            mock_yf.download.return_value = fake_df
            vix = get_vix(cfg)

        assert vix.index.tz is None


# ---------------------------------------------------------------------------
# compute_vix_regime
# ---------------------------------------------------------------------------


class TestComputeVixRegime:
    def test_regime_values_in_valid_range(self, config):
        rng = np.random.default_rng(0)
        vix = pd.Series(
            rng.uniform(10, 50, 500), index=pd.date_range("2018-01-01", periods=500)
        )
        regime = compute_vix_regime(vix, config)
        valid = regime.dropna()
        assert len(valid) > 0
        assert valid.isin([1.0, 2.0, 3.0]).all(), f"unexpected values: {valid.unique()}"

    def test_first_vix_regime_window_dates_are_nan(self, config):
        rng = np.random.default_rng(1)
        n = config.vix_regime_window + 100
        vix = pd.Series(rng.uniform(10, 50, n), index=pd.date_range("2018-01-01", periods=n))
        regime = compute_vix_regime(vix, config)
        assert regime.iloc[: config.vix_regime_window].isna().all(), (
            "first vix_regime_window dates must be NaN"
        )

    def test_past_only_adding_future_data_does_not_change_prior_regimes(self, config):
        """Past-only invariant: regime at t is unchanged by appending t+1."""
        rng = np.random.default_rng(2)
        n = config.vix_regime_window + 50
        vix = pd.Series(rng.uniform(10, 50, n), index=pd.date_range("2018-01-01", periods=n))
        regime_orig = compute_vix_regime(vix, config)

        # Append an extreme future value — should not affect any prior regime
        extra_date = vix.index[-1] + pd.Timedelta(days=1)
        vix_ext = pd.concat([vix, pd.Series([200.0], index=[extra_date])])
        regime_ext = compute_vix_regime(vix_ext, config)

        pd.testing.assert_series_equal(
            regime_orig,
            regime_ext.reindex(vix.index),
            check_names=False,
        )

    def test_returns_float_dtype_for_nan_support(self, config):
        rng = np.random.default_rng(3)
        vix = pd.Series(rng.uniform(10, 50, 300), index=pd.date_range("2018-01-01", periods=300))
        regime = compute_vix_regime(vix, config)
        assert pd.api.types.is_float_dtype(regime), f"expected float dtype, got {regime.dtype}"


# ---------------------------------------------------------------------------
# ic_by_bucket
# ---------------------------------------------------------------------------


class TestIcByBucket:
    def test_date_level_buckets_returns_one_row_per_label(self):
        """Series buckets (VIX-style) → one row per unique label."""
        rng = np.random.default_rng(10)
        dates = pd.date_range("2020-01-01", periods=200)
        tickers = [f"T{i}" for i in range(50)]
        signal = pd.DataFrame(rng.normal(0, 1, (200, 50)), index=dates, columns=tickers)
        fwd = pd.DataFrame(rng.normal(0, 1, (200, 50)), index=dates, columns=tickers)
        buckets = pd.Series(
            rng.integers(1, 4, 200).astype(float), index=dates
        )

        result = ic_by_bucket(signal, fwd, buckets, compute_ic, compute_ic_tstat)
        assert set(result.index.tolist()) == {1.0, 2.0, 3.0}
        assert list(result.columns) == ["mean_ic", "std_ic", "t_stat_hac", "n_dates"]

    def test_stock_level_buckets_returns_one_row_per_quintile(self):
        """DataFrame buckets (Amihud-style) → one row per quintile label.

        Use 200 tickers so each quintile has ~40 stocks per date, satisfying
        compute_ic's minimum of 30 valid pairs.
        """
        rng = np.random.default_rng(11)
        n_tickers = 200
        dates = pd.date_range("2020-01-01", periods=200)
        tickers = [f"T{i}" for i in range(n_tickers)]
        signal = pd.DataFrame(rng.normal(0, 1, (200, n_tickers)), index=dates, columns=tickers)
        fwd = pd.DataFrame(rng.normal(0, 1, (200, n_tickers)), index=dates, columns=tickers)
        bucket_vals = rng.integers(1, 6, (200, n_tickers)).astype(float)
        buckets_df = pd.DataFrame(bucket_vals, index=dates, columns=tickers)

        result = ic_by_bucket(signal, fwd, buckets_df, compute_ic, compute_ic_tstat)
        assert len(result) == 5, f"expected 5 rows, got {len(result)}"
        assert list(result.columns) == ["mean_ic", "std_ic", "t_stat_hac", "n_dates"]

    def test_known_perfect_ic_in_bucket_1_zero_ic_in_bucket_2(self):
        """Bucket 1: fwd = signal (high IC). Bucket 2: fwd is random (≈zero IC)."""
        rng = np.random.default_rng(12)
        n_stocks = 50
        dates_b1 = pd.date_range("2020-01-02", periods=100, freq="B")
        dates_b2 = pd.date_range("2020-06-01", periods=100, freq="B")
        all_dates = dates_b1.append(dates_b2)
        tickers = [f"T{i}" for i in range(n_stocks)]

        sig_vals = rng.normal(0, 1, (200, n_stocks))
        fwd_vals = sig_vals.copy()
        fwd_vals[100:] = rng.normal(0, 1, (100, n_stocks))  # bucket 2: random

        signal = pd.DataFrame(sig_vals, index=all_dates, columns=tickers)
        fwd = pd.DataFrame(fwd_vals, index=all_dates, columns=tickers)
        buckets = pd.Series([1.0] * 100 + [2.0] * 100, index=all_dates)

        result = ic_by_bucket(signal, fwd, buckets, compute_ic, compute_ic_tstat)
        assert result.loc[1.0, "mean_ic"] > 0.5, (
            f"bucket 1 should have high IC, got {result.loc[1.0, 'mean_ic']:.3f}"
        )
        assert abs(result.loc[2.0, "mean_ic"]) < 0.3, (
            f"bucket 2 should have near-zero IC, got {result.loc[2.0, 'mean_ic']:.3f}"
        )

    def test_empty_buckets_returns_empty_dataframe_with_correct_columns(self):
        """Empty input returns correctly structured empty DataFrame."""
        result = ic_by_bucket(
            pd.DataFrame(dtype=float),
            pd.DataFrame(dtype=float),
            pd.Series(dtype=float),
            lambda s, f: pd.Series(dtype=float),
            lambda ic: 0.0,
        )
        assert list(result.columns) == ["mean_ic", "std_ic", "t_stat_hac", "n_dates"]
        assert len(result) == 0
