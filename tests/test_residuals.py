"""Tests for residrev.residuals — synthetic data, no real I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.residuals import build_return_panel, compute_idio_vol, rolling_residuals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["A", "B", "C", "D", "E"]
N_DATES = 150
DATES = pd.bdate_range("2020-01-01", periods=N_DATES)


def _make_config() -> Config:
    return Config(factor_window=20, min_obs=10)


def _make_prices(
    tickers: list[str] | None = None,
    n_dates: int = N_DATES,
    base_close: float = 100.0,
) -> dict[str, pd.DataFrame]:
    tickers = tickers or TICKERS
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rng = np.random.RandomState(42)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        rets = rng.normal(0, 0.02, size=n_dates)
        close = base_close * np.exp(np.cumsum(rets))
        out[t] = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(n_dates, 1_000_000.0),
            },
            index=dates,
        )
    return out


def _make_universe(
    tickers: list[str] | None = None,
    n_dates: int = N_DATES,
    all_true: bool = True,
) -> pd.DataFrame:
    tickers = tickers or TICKERS
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    if all_true:
        return pd.DataFrame(True, index=dates, columns=tickers)
    vals = np.ones((n_dates, len(tickers)), dtype=bool)
    vals[:, -1] = False  # last ticker excluded
    return pd.DataFrame(vals, index=dates, columns=tickers)


def _make_factors(n_dates: int = N_DATES) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rng = np.random.RandomState(99)
    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    data = {f: rng.normal(0, 0.005, size=n_dates) for f in factor_names}
    data["RF"] = np.full(n_dates, 0.0001)
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# build_return_panel
# ---------------------------------------------------------------------------


class TestBuildReturnPanel:
    def test_output_shape_matches_universe(self):
        prices = _make_prices()
        universe = _make_universe()
        panel = build_return_panel(prices, universe)
        assert panel.shape == (N_DATES, len(TICKERS))
        assert list(panel.columns) == TICKERS

    def test_log_returns_correct(self):
        prices = _make_prices(["X"], n_dates=5)
        universe = _make_universe(["X"], n_dates=5)
        panel = build_return_panel(prices, universe)
        close = prices["X"]["Close"]
        expected = np.log(close.iloc[1] / close.iloc[0])
        assert panel.iloc[1, 0] == pytest.approx(expected, rel=1e-10)

    def test_first_row_is_nan(self):
        prices = _make_prices()
        universe = _make_universe()
        panel = build_return_panel(prices, universe)
        assert panel.iloc[0].isna().all()

    def test_nan_where_universe_false(self):
        prices = _make_prices()
        universe = _make_universe(all_true=False)
        panel = build_return_panel(prices, universe)
        assert panel["E"].isna().all()
        assert not panel["A"].iloc[1:].isna().all()


# ---------------------------------------------------------------------------
# rolling_residuals
# ---------------------------------------------------------------------------


class TestRollingResiduals:
    def test_output_shapes(self):
        cfg = _make_config()
        prices = _make_prices()
        universe = _make_universe()
        returns = build_return_panel(prices, universe)
        factors = _make_factors()

        resid, betas, idio_var = rolling_residuals(returns, factors, cfg)

        assert resid.shape == returns.shape
        assert idio_var.shape == returns.shape
        for name in list(cfg.factors) + ["intercept"]:
            assert name in betas
            assert betas[name].shape == returns.shape

    def test_first_window_rows_all_nan(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()

        resid, betas, idio_var = rolling_residuals(returns, factors, cfg)

        assert resid.iloc[: cfg.factor_window].isna().all().all()
        assert idio_var.iloc[: cfg.factor_window].isna().all().all()
        for beta_df in betas.values():
            assert beta_df.iloc[: cfg.factor_window].isna().all().all()

    def test_no_look_ahead(self):
        """Perturbing returns[t+1] must not change resid[t]."""
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()

        resid_orig, _, _ = rolling_residuals(returns.copy(), factors, cfg)

        t_check = cfg.factor_window + 10
        returns_perturbed = returns.copy()
        returns_perturbed.iloc[t_check + 1] = 999.0

        resid_perturbed, _, _ = rolling_residuals(returns_perturbed, factors, cfg)

        pd.testing.assert_series_equal(
            resid_orig.iloc[t_check],
            resid_perturbed.iloc[t_check],
            check_names=False,
        )

    def test_nan_stock_gets_nan_residual(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()

        returns["A"] = np.nan

        resid, betas, idio_var = rolling_residuals(returns, factors, cfg)

        assert resid["A"].isna().all()
        assert idio_var["A"].isna().all()

    def test_known_case_stock_equals_factor(self):
        """If a stock's return equals a factor return exactly, residual ≈ 0."""
        cfg = Config(factor_window=20, min_obs=10)
        factors = _make_factors()

        returns = build_return_panel(_make_prices(), _make_universe())
        rf = factors["RF"].values
        mkt = factors["Mkt-RF"].values
        for col in returns.columns:
            returns[col] = mkt + rf

        resid, _, _ = rolling_residuals(returns, factors, cfg)

        valid = resid.iloc[cfg.factor_window :].dropna(how="all")
        max_abs_resid = valid.abs().max().max()
        assert max_abs_resid < 0.01

    def test_betas_dict_keys(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()

        _, betas, _ = rolling_residuals(returns, factors, cfg)

        expected_keys = set(cfg.factors) | {"intercept"}
        assert set(betas.keys()) == expected_keys

    def test_idio_var_non_negative(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()

        _, _, idio_var = rolling_residuals(returns, factors, cfg)

        valid = idio_var.dropna(how="all")
        assert (valid.fillna(0) >= 0).all().all()

    def test_excess_returns_rf_subtraction(self):
        """Verify that RF is subtracted: with large RF, intercept shifts."""
        cfg = Config(factor_window=20, min_obs=10)
        factors = _make_factors()
        returns = build_return_panel(_make_prices(), _make_universe())

        _, betas_base, _ = rolling_residuals(returns, factors, cfg)

        factors_high_rf = factors.copy()
        factors_high_rf["RF"] = 0.05

        _, betas_high, _ = rolling_residuals(returns, factors_high_rf, cfg)

        intercept_base = betas_base["intercept"].iloc[cfg.factor_window :].mean().mean()
        intercept_high = betas_high["intercept"].iloc[cfg.factor_window :].mean().mean()
        assert abs(intercept_high - intercept_base) > 0.01


# ---------------------------------------------------------------------------
# compute_idio_vol
# ---------------------------------------------------------------------------


class TestComputeIdioVol:
    def test_output_non_negative(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()
        _, _, idio_var = rolling_residuals(returns, factors, cfg)

        idio_vol = compute_idio_vol(idio_var)
        valid = idio_vol.dropna(how="all")
        assert (valid.fillna(0) >= 0).all().all()

    def test_shape_preserved(self):
        cfg = _make_config()
        returns = build_return_panel(_make_prices(), _make_universe())
        factors = _make_factors()
        _, _, idio_var = rolling_residuals(returns, factors, cfg)

        idio_vol = compute_idio_vol(idio_var)
        assert idio_vol.shape == idio_var.shape

    def test_rolling_window_respected(self):
        data = pd.DataFrame({"A": [1.0, 4.0, 9.0, 16.0, 25.0]})
        vol = compute_idio_vol(data, window=3)
        expected_t2 = np.sqrt((1 + 4 + 9) / 3)
        assert vol.iloc[2, 0] == pytest.approx(expected_t2, rel=1e-10)
