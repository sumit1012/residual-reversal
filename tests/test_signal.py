"""Tests for residrev.signal — synthetic data, no real I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.signal import build_signal, compute_ic, compute_ic_decay, compute_ic_tstat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = [f"T{i}" for i in range(60)]  # 60 tickers for cross-sectional tests
N_DATES = 100
DATES = pd.bdate_range("2020-01-01", periods=N_DATES)


def _cfg(k: int = 5) -> Config:
    return Config(signal_k=k, winsorize_pct=0.01)


def _make_resid(
    tickers: list[str] | None = None,
    n_dates: int = N_DATES,
    seed: int = 42,
) -> pd.DataFrame:
    tickers = tickers or TICKERS
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(n_dates, len(tickers))),
        index=pd.bdate_range("2020-01-01", periods=n_dates),
        columns=tickers,
    )


def _make_sector_map(tickers: list[str] | None = None) -> dict[str, str]:
    tickers = tickers or TICKERS
    sectors = ["Energy", "Finance", "Tech", "Health", "Utils"]
    return {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}


# ---------------------------------------------------------------------------
# build_signal
# ---------------------------------------------------------------------------


class TestBuildSignal:
    def test_output_shapes_match_input(self):
        resid = _make_resid()
        raw, tradeable = build_signal(resid, _make_sector_map(), _cfg())
        assert raw.shape == resid.shape
        assert tradeable.shape == resid.shape

    def test_negates_cumulative_residuals(self):
        """Positive residual sum → negative raw signal (before z-scoring)."""
        cfg = _cfg(k=3)
        tickers = [f"T{i}" for i in range(60)]
        dates = pd.bdate_range("2020-01-01", periods=10)
        resid = pd.DataFrame(0.01, index=dates, columns=tickers)
        resid.iloc[:, 0] = 0.05  # T0 has large positive residuals

        raw, _ = build_signal(resid, _make_sector_map(tickers), cfg)
        # After z-scoring, T0 should still be the most negative
        valid_row = raw.iloc[cfg.signal_k - 1 :]
        assert (valid_row["T0"].dropna() < 0).all()

    def test_first_k_minus_1_rows_nan(self):
        cfg = _cfg(k=5)
        resid = _make_resid()
        raw, _ = build_signal(resid, _make_sector_map(), cfg)
        assert raw.iloc[: cfg.signal_k - 1].isna().all().all()

    def test_tradeable_is_raw_shifted_by_gap(self):
        resid = _make_resid()
        cfg = _cfg()
        raw, tradeable = build_signal(resid, _make_sector_map(), cfg)
        # tradeable[t] should equal raw[t-(1+signal_gap)] (1-day lag + skip-day gap)
        shift_n = 1 + cfg.signal_gap
        pd.testing.assert_frame_equal(
            tradeable.iloc[shift_n:].reset_index(drop=True),
            raw.iloc[:-shift_n].reset_index(drop=True),
        )

    def test_winsorization_clips_extremes(self):
        cfg = _cfg(k=1)
        tickers = [f"T{i}" for i in range(100)]
        dates = pd.bdate_range("2020-01-01", periods=5)
        rng = np.random.RandomState(7)
        resid = pd.DataFrame(
            rng.normal(0, 0.01, size=(5, 100)), index=dates, columns=tickers
        )
        # Inject extreme outlier
        resid.iloc[2, 0] = 10.0
        resid.iloc[2, 1] = -10.0

        sector_map = {t: "Sec" for t in tickers}
        raw, _ = build_signal(resid, sector_map, cfg)
        # After z-score the extremes are tamed; the raw max should be bounded
        row = raw.iloc[2].dropna()
        assert row.max() < 10.0  # winsorization tames extremes vs unbounded

    def test_sector_demeaning_zero_sector_mean(self):
        """After sector-demeaning, each sector's mean should be ~0.

        Smoothing disabled here: this test targets the sector-demean step in
        isolation. The downstream rolling-mean smoothing deliberately mixes dates
        and re-z-scores globally, so it does not preserve exact per-sector zero
        means (portfolio sector-neutrality is enforced by the optimizer, not the
        raw signal).
        """
        cfg = Config(signal_k=1, winsorize_pct=0.01, signal_smooth_span=1)
        tickers = [f"T{i}" for i in range(60)]
        resid = _make_resid(tickers, n_dates=20)
        sector_map = _make_sector_map(tickers)

        raw, _ = build_signal(resid, sector_map, cfg)
        sector_labels = pd.Series(sector_map)

        # Check a row where data is valid (after k-1)
        row = raw.iloc[5]
        for sector in sector_labels.unique():
            members = sector_labels[sector_labels == sector].index
            vals = row[members].dropna()
            if len(vals) > 1:
                assert abs(vals.mean()) < 1e-10

    def test_zscore_mean_zero_std_one(self):
        cfg = _cfg(k=1)
        resid = _make_resid()
        raw, _ = build_signal(resid, _make_sector_map(), cfg)

        for i in range(cfg.signal_k, len(raw)):
            row = raw.iloc[i].dropna()
            if len(row) > 1:
                assert abs(row.mean()) < 1e-10
                assert abs(row.std() - 1.0) < 0.05  # within tolerance


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------


class TestComputeIc:
    def test_nan_when_fewer_than_30_pairs(self):
        tickers = [f"T{i}" for i in range(20)]
        dates = pd.bdate_range("2020-01-01", periods=5)
        signal = pd.DataFrame(1.0, index=dates, columns=tickers)
        fwd = pd.DataFrame(1.0, index=dates, columns=tickers)
        ic = compute_ic(signal, fwd)
        assert ic.isna().all()

    def test_perfect_positive_correlation(self):
        tickers = [f"T{i}" for i in range(50)]
        dates = pd.bdate_range("2020-01-01", periods=5)
        rng = np.random.RandomState(0)
        vals = rng.randn(5, 50)
        signal = pd.DataFrame(vals, index=dates, columns=tickers)
        fwd = pd.DataFrame(vals, index=dates, columns=tickers)
        ic = compute_ic(signal, fwd)
        assert (ic.dropna() > 0.99).all()

    def test_perfect_negative_correlation(self):
        tickers = [f"T{i}" for i in range(50)]
        dates = pd.bdate_range("2020-01-01", periods=5)
        rng = np.random.RandomState(0)
        vals = rng.randn(5, 50)
        signal = pd.DataFrame(vals, index=dates, columns=tickers)
        fwd = pd.DataFrame(-vals, index=dates, columns=tickers)
        ic = compute_ic(signal, fwd)
        assert (ic.dropna() < -0.99).all()


# ---------------------------------------------------------------------------
# compute_ic_tstat
# ---------------------------------------------------------------------------


class TestComputeIcTstat:
    def test_sign_matches_mean_ic(self):
        rng = np.random.RandomState(1)
        ic = pd.Series(rng.normal(0.05, 0.1, size=200))
        t = compute_ic_tstat(ic)
        assert t > 0  # positive mean → positive t

        ic_neg = pd.Series(rng.normal(-0.05, 0.1, size=200))
        t_neg = compute_ic_tstat(ic_neg)
        assert t_neg < 0

    def test_zero_ic_near_zero_tstat(self):
        ic = pd.Series(np.zeros(100))
        t = compute_ic_tstat(ic)
        assert abs(t) < 1e-10


# ---------------------------------------------------------------------------
# compute_ic_decay
# ---------------------------------------------------------------------------


class TestComputeIcDecay:
    def test_output_shape(self):
        tickers = [f"T{i}" for i in range(50)]
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.RandomState(3)
        signal = pd.DataFrame(
            rng.randn(100, 50), index=dates, columns=tickers
        )
        returns = pd.DataFrame(
            rng.randn(100, 50) * 0.01, index=dates, columns=tickers
        )
        decay = compute_ic_decay(signal, returns, max_lag=10)
        assert decay.shape == (10, 4)
        assert list(decay.columns) == ["mean_ic", "std_ic", "t_stat_hac", "n_obs"]
        assert list(decay.index) == list(range(1, 11))

    def test_lag1_matches_compute_ic(self):
        tickers = [f"T{i}" for i in range(50)]
        dates = pd.bdate_range("2020-01-01", periods=80)
        rng = np.random.RandomState(5)
        signal = pd.DataFrame(
            rng.randn(80, 50), index=dates, columns=tickers
        )
        returns = pd.DataFrame(
            rng.randn(80, 50) * 0.01, index=dates, columns=tickers
        )

        decay = compute_ic_decay(signal, returns, max_lag=3)

        fwd_1 = returns.shift(-1).rolling(1).sum()
        ic_direct = compute_ic(signal, fwd_1)
        expected_mean = ic_direct.dropna().mean()

        assert decay.loc[1, "mean_ic"] == pytest.approx(expected_mean, abs=1e-10)
