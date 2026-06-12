"""Tests for residrev.portfolio — cvxpy mean-variance optimizer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.portfolio import (
    _build_sector_constraints,
    _validate_weights,
    optimize_book,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config():
    """Config for a tiny 6-stock, 2-factor test."""
    return Config(
        factors=("F1", "F2"),
        gamma=1.0,
        lam_to=0.0,
        max_w=0.5,
        gross_cap=2.0,
        beta_tol=1e-3,
        sector_tol=1e-3,
    )


@pytest.fixture()
def small_inputs(small_config):
    """Synthetic 6-stock, 2-factor problem."""
    tickers = ["A", "B", "C", "D", "E", "F"]
    np.random.seed(42)
    alpha = pd.Series([0.05, -0.03, 0.04, -0.02, 0.01, -0.01], index=tickers)
    betas = {
        "F1": pd.Series(np.random.randn(6) * 0.5, index=tickers),
        "F2": pd.Series(np.random.randn(6) * 0.5, index=tickers),
    }
    idio_var = pd.Series(np.random.rand(6) * 0.01 + 0.001, index=tickers)
    factor_cov = np.eye(2) * 0.0004
    sector_labels = {
        "A": "Tech",
        "B": "Tech",
        "C": "Health",
        "D": "Health",
        "E": "Energy",
        "F": "Energy",
    }
    return alpha, betas, idio_var, factor_cov, sector_labels


# ---------------------------------------------------------------------------
# optimize_book
# ---------------------------------------------------------------------------

class TestOptimizeBook:

    def test_returns_series_with_correct_index(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        assert isinstance(w, pd.Series)
        assert sorted(w.index.tolist()) == sorted(alpha.index.tolist())

    def test_dollar_neutrality(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        assert abs(w.sum()) < 1e-4

    def test_position_cap(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        assert w.abs().max() <= small_config.max_w + 1e-5

    def test_gross_leverage(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        assert w.abs().sum() <= small_config.gross_cap + 1e-5

    def test_factor_exposure_near_zero(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        tickers = sorted(w.index.tolist())
        for factor in small_config.factors:
            beta_vec = betas[factor].reindex(tickers).values
            exposure = abs(w.reindex(tickers).values @ beta_vec)
            assert exposure < small_config.beta_tol + 1e-5

    def test_high_alpha_gets_positive_weight(self, small_config):
        """With identical betas/risk, higher alpha should get higher weight."""
        tickers = ["HI", "LO", "HI2", "LO2"]
        alpha = pd.Series([1.0, -1.0, 1.0, -1.0], index=tickers)
        betas = {
            "F1": pd.Series([0.0, 0.0, 0.0, 0.0], index=tickers),
            "F2": pd.Series([0.0, 0.0, 0.0, 0.0], index=tickers),
        }
        idio_var = pd.Series([0.01, 0.01, 0.01, 0.01], index=tickers)
        factor_cov = np.eye(2) * 0.0004
        sector_labels = {"HI": "A", "LO": "A", "HI2": "B", "LO2": "B"}
        w = optimize_book(alpha, betas, idio_var, factor_cov, sector_labels, small_config)
        assert w["HI"] > 0
        assert w["LO"] < 0

    def test_turnover_penalty_keeps_weights_close(self, small_inputs):
        """With large turnover penalty, weights stay near w_prev."""
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        cfg = Config(
            factors=("F1", "F2"),
            gamma=1.0,
            lam_to=1000.0,
            max_w=0.5,
            gross_cap=2.0,
            beta_tol=1e-3,
            sector_tol=1e-3,
        )
        w_prev = pd.Series(0.0, index=sorted(alpha.index))
        w = optimize_book(
            alpha, betas, idio_var, factor_cov, sector_labels, cfg, w_prev=w_prev
        )
        assert np.abs(w.values).sum() < 0.01

    def test_all_solvers_fail_returns_zeros(self, small_inputs, small_config):
        alpha, betas, idio_var, factor_cov, sector_labels = small_inputs
        with patch("residrev.portfolio.cp.Problem") as mock_prob:
            instance = mock_prob.return_value
            instance.status = "infeasible"
            instance.solve = lambda **kw: None
            w = optimize_book(
                alpha, betas, idio_var, factor_cov, sector_labels, small_config
            )
        assert (w == 0.0).all()
        assert len(w) == len(alpha)

    def test_empty_intersection_returns_empty(self, small_config):
        alpha = pd.Series([0.1], index=["X"])
        betas = {"F1": pd.Series([0.5], index=["Y"]), "F2": pd.Series([0.5], index=["Y"])}
        idio_var = pd.Series([0.01], index=["Z"])
        sector_labels = {"W": "Tech"}
        w = optimize_book(alpha, betas, idio_var, np.eye(2), sector_labels, small_config)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# _build_sector_constraints
# ---------------------------------------------------------------------------

class TestBuildSectorConstraints:

    def test_skips_single_member_sectors(self):
        import cvxpy as cp
        w = cp.Variable(5)
        tickers = ["A", "B", "C", "D", "E"]
        sector_labels = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health", "E": "Solo"}
        cfg = Config(factors=("F1",), sector_tol=1e-3)
        constraints = _build_sector_constraints(w, tickers, sector_labels, cfg)
        assert len(constraints) == 2  # Tech and Health, not Solo

    def test_all_single_member_returns_empty(self):
        import cvxpy as cp
        w = cp.Variable(3)
        tickers = ["A", "B", "C"]
        sector_labels = {"A": "S1", "B": "S2", "C": "S3"}
        cfg = Config(factors=("F1",), sector_tol=1e-3)
        constraints = _build_sector_constraints(w, tickers, sector_labels, cfg)
        assert len(constraints) == 0


# ---------------------------------------------------------------------------
# _validate_weights
# ---------------------------------------------------------------------------

class TestValidateWeights:

    def test_dollar_neutral_value(self):
        w = np.array([0.1, -0.05, -0.05])
        tickers = ["A", "B", "C"]
        B = np.zeros((3, 1))
        sector_labels = {"A": "X", "B": "X", "C": "X"}
        cfg = Config(factors=("F1",))
        diag = _validate_weights(w, tickers, B, sector_labels, cfg)
        assert abs(diag["dollar_neutral"] - 0.0) < 1e-10

    def test_gross_leverage_value(self):
        w = np.array([0.3, -0.2, -0.1])
        tickers = ["A", "B", "C"]
        B = np.zeros((3, 1))
        sector_labels = {"A": "X", "B": "X", "C": "X"}
        cfg = Config(factors=("F1",))
        diag = _validate_weights(w, tickers, B, sector_labels, cfg)
        assert abs(diag["gross_leverage"] - 0.6) < 1e-10

    def test_max_position_value(self):
        w = np.array([0.02, -0.01, -0.01])
        tickers = ["A", "B", "C"]
        B = np.zeros((3, 1))
        sector_labels = {"A": "X", "B": "X", "C": "X"}
        cfg = Config(factors=("F1",))
        diag = _validate_weights(w, tickers, B, sector_labels, cfg)
        assert abs(diag["max_position"] - 0.02) < 1e-10

    def test_factor_exposure_reported(self):
        w = np.array([0.5, -0.5])
        tickers = ["A", "B"]
        B = np.array([[1.0], [0.5]])
        sector_labels = {"A": "X", "B": "X"}
        cfg = Config(factors=("F1",))
        diag = _validate_weights(w, tickers, B, sector_labels, cfg)
        expected = abs(0.5 * 1.0 + (-0.5) * 0.5)
        assert abs(diag["factor_exposures"]["F1"] - expected) < 1e-10
