"""cvxpy mean-variance optimizer with neutrality constraints."""

from __future__ import annotations

import logging
import time

import cvxpy as cp
import numpy as np
import pandas as pd

from residrev.config import Config

logger = logging.getLogger(__name__)


def optimize_book(
    alpha: pd.Series,
    betas: dict[str, pd.Series],
    idio_var: pd.Series,
    factor_cov: np.ndarray,
    sector_labels: dict[str, str],
    config: Config,
    w_prev: pd.Series | None = None,
) -> pd.Series:
    """Solve for optimal portfolio weights via mean-variance optimization.

    Returns pd.Series of weights indexed by ticker.
    """
    tickers = _align_tickers(alpha, betas, idio_var, sector_labels)
    if len(tickers) == 0:
        logger.warning("optimize_book: no common tickers after alignment")
        return pd.Series(dtype=float)
    if len(tickers) < 50:
        logger.warning("optimize_book: only %d tickers after alignment", len(tickers))

    N = len(tickers)
    K = len(config.factors)
    alpha_vec = alpha.reindex(tickers).values.astype(float)
    B = np.column_stack(
        [betas[f].reindex(tickers).values.astype(float) for f in config.factors]
    )
    D_diag = idio_var.reindex(tickers).values.astype(float)
    D_diag = np.maximum(D_diag, 1e-10)
    w_prev_vec = (
        w_prev.reindex(tickers, fill_value=0.0).values.astype(float)
        if w_prev is not None
        else np.zeros(N)
    )

    w = cp.Variable(N)

    factor_risk = cp.quad_form(B.T @ w, factor_cov)
    idio_risk = cp.sum_squares(cp.multiply(np.sqrt(D_diag), w))
    risk = factor_risk + idio_risk

    turnover = cp.norm1(w - w_prev_vec)

    objective = cp.Maximize(
        alpha_vec @ w - (config.gamma / 2) * risk - config.lam_to * turnover
    )

    constraints = [
        cp.sum(w) == 0,
        cp.abs(B.T @ w) <= config.beta_tol,
        w >= -config.max_w,
        w <= config.max_w,
        cp.norm1(w) <= config.gross_cap,
    ]
    constraints.extend(
        _build_sector_constraints(w, tickers, sector_labels, config)
    )

    prob = cp.Problem(objective, constraints)

    solvers = [cp.CLARABEL, cp.OSQP, cp.SCS]
    solved = False
    for solver in solvers:
        try:
            t0 = time.perf_counter()
            prob.solve(solver=solver)
            elapsed = time.perf_counter() - t0
            if prob.status in ("optimal", "optimal_inaccurate"):
                logger.info(
                    "optimize_book: solved with %s in %.3fs (status=%s)",
                    solver, elapsed, prob.status,
                )
                solved = True
                break
            logger.info(
                "optimize_book: %s returned status=%s, trying next",
                solver, prob.status,
            )
        except cp.SolverError as exc:
            logger.info("optimize_book: %s raised SolverError: %s", solver, exc)

    if not solved:
        logger.warning("optimize_book: all solvers failed, returning zeros")
        return pd.Series(0.0, index=tickers)

    weights = np.array(w.value).flatten()
    diag = _validate_weights(weights, tickers, B, sector_labels, config)
    logger.info(
        "optimize_book: dollar_neutral=%.2e, max_pos=%.4f, "
        "gross=%.4f, sector_max=%.4f",
        diag["dollar_neutral"],
        diag["max_position"],
        diag["gross_leverage"],
        diag["sector_max_exposure"],
    )
    return pd.Series(weights, index=tickers)


def _align_tickers(
    alpha: pd.Series,
    betas: dict[str, pd.Series],
    idio_var: pd.Series,
    sector_labels: dict[str, str],
) -> list[str]:
    """Return sorted intersection of tickers present in all inputs."""
    common = set(alpha.dropna().index)
    for factor_beta in betas.values():
        common &= set(factor_beta.dropna().index)
    common &= set(idio_var.dropna().index)
    common &= set(sector_labels.keys())
    return sorted(common)


def _build_sector_constraints(
    w: cp.Variable,
    tickers: list[str],
    sector_labels: dict[str, str],
    config: Config,
) -> list:
    """Build sector-neutrality constraints, skipping single-member sectors."""
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    sectors: dict[str, list[int]] = {}
    for t in tickers:
        sec = sector_labels.get(t, "Other")
        sectors.setdefault(sec, []).append(ticker_to_idx[t])

    constraints = []
    for sec, idx_list in sectors.items():
        if len(idx_list) < 2:
            continue
        constraints.append(cp.sum(w[idx_list]) == 0)
    return constraints


def _validate_weights(
    w: np.ndarray,
    tickers: list[str],
    B: np.ndarray,
    sector_labels: dict[str, str],
    config: Config,
) -> dict:
    """Check solved weights for constraint satisfaction."""
    dollar_neutral = float(np.abs(w.sum()))
    max_position = float(np.abs(w).max())
    gross_leverage = float(np.abs(w).sum())

    factor_exposures = {}
    for i, f in enumerate(config.factors):
        exposure = float(np.abs(w @ B[:, i]))
        factor_exposures[f] = exposure

    sector_exposures: dict[str, float] = {}
    for t, sec in sector_labels.items():
        if t not in tickers:
            continue
        idx = tickers.index(t)
        sector_exposures[sec] = sector_exposures.get(sec, 0.0) + w[idx]
    sector_max_exposure = (
        float(max(abs(v) for v in sector_exposures.values()))
        if sector_exposures
        else 0.0
    )

    if dollar_neutral > 1e-4:
        logger.warning("_validate_weights: dollar_neutral=%.4e exceeds 1e-4", dollar_neutral)
    if max_position > config.max_w + 1e-5:
        logger.warning("_validate_weights: max_position=%.4f exceeds max_w=%.4f", max_position, config.max_w)
    if gross_leverage > config.gross_cap + 1e-5:
        logger.warning("_validate_weights: gross_leverage=%.4f exceeds gross_cap=%.4f", gross_leverage, config.gross_cap)

    return {
        "dollar_neutral": dollar_neutral,
        "max_position": max_position,
        "gross_leverage": gross_leverage,
        "factor_exposures": factor_exposures,
        "sector_max_exposure": sector_max_exposure,
    }
