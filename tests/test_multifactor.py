"""Unit tests for the multi-factor sleeve. Synthetic data only (fast, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.residuals import compute_idio_vol
from residrev.multifactor import (
    _xs_standardize, residual_momentum, low_risk_idiovol, low_risk_bab, high_52w,
    build_composite_alpha, multifactor_weights, MOM_LOOKBACK, MOM_SKIP,
)


@pytest.fixture
def synth():
    rng = np.random.default_rng(7)
    n_dates, n_names = 400, 60
    idx = pd.date_range("2019-01-01", periods=n_dates, freq="B")
    cols = [f"S{i:02d}" for i in range(n_names)]
    resid = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_names)), index=idx, columns=cols)
    idio_var = pd.DataFrame(rng.uniform(1e-4, 4e-4, (n_dates, n_names)), index=idx, columns=cols)
    beta = pd.DataFrame(rng.normal(1.0, 0.3, (n_dates, n_names)), index=idx, columns=cols)
    sector_map = {c: f"SEC{i % 4}" for i, c in enumerate(cols)}
    return dict(idx=idx, cols=cols, resid=resid, idio_var=idio_var, beta=beta, sector_map=sector_map)


def test_residual_momentum_is_12_minus_1(synth):
    resid = synth["resid"]
    mom = residual_momentum(resid, lookback=MOM_LOOKBACK, skip=MOM_SKIP)
    t = 300
    expected = resid.iloc[t - MOM_LOOKBACK + 1 : t - MOM_SKIP + 1].sum()
    np.testing.assert_allclose(mom.iloc[t].values, expected.values, rtol=1e-9, atol=1e-12)


def test_residual_momentum_past_only(synth):
    resid = synth["resid"].copy()
    mom1 = residual_momentum(resid)
    j = 320
    perturbed = resid.copy()
    perturbed.iloc[j:] += 5.0          # change everything from row j onward
    mom2 = residual_momentum(perturbed)
    # rows strictly before j must be untouched (no look-ahead into the future)
    pd.testing.assert_frame_equal(mom1.iloc[:j], mom2.iloc[:j])


def test_low_risk_signs(synth):
    iv = low_risk_idiovol(synth["idio_var"])
    np.testing.assert_allclose(iv.values, -compute_idio_vol(synth["idio_var"]).values, equal_nan=True)
    bab = low_risk_bab(synth["beta"])
    np.testing.assert_allclose(bab.values, -synth["beta"].values)


def test_xs_standardize_zscored_and_sector_demeaned(synth):
    cfg = Config()
    z = _xs_standardize(residual_momentum(synth["resid"]), synth["sector_map"], cfg.winsorize_pct)
    row = z.iloc[350].dropna()
    assert abs(row.mean()) < 1e-9                     # cross-sectional mean ~0
    assert abs(row.std(ddof=0) - 1.0) < 0.05          # ~unit std
    # each sector's mean is ~0 after demeaning (pre-zscore property preserved by linear z)
    sectors = pd.Series(synth["sector_map"])
    for s in sectors.unique():
        members = [c for c in row.index if synth["sector_map"][c] == s]
        if len(members) > 1:
            assert abs(row[members].mean()) < 0.6     # demeaned, much tighter than raw


def test_composite_is_lagged_one_day(synth):
    cfg = Config()
    raws = {"mom": residual_momentum(synth["resid"]),
            "lr": low_risk_idiovol(synth["idio_var"])}
    alpha = build_composite_alpha(raws, synth["sector_map"], cfg)
    # reconstruct the un-lagged composite and confirm alpha[t] == composite[t-1]
    zs = [_xs_standardize(r, synth["sector_map"], cfg.winsorize_pct) for r in raws.values()]
    comp = np.nanmean(np.stack([z.values for z in zs]), axis=0)
    comp = pd.DataFrame(comp, index=zs[0].index, columns=zs[0].columns)
    comp = comp.sub(comp.mean(axis=1), axis=0).div(comp.std(axis=1).replace(0, np.nan), axis=0)
    pd.testing.assert_frame_equal(alpha.iloc[1:], comp.shift(1).iloc[1:])
    assert alpha.iloc[0].isna().all()                 # first row is NaN after the lag


def test_weights_neutral_and_capped(synth):
    cfg = Config()
    raws = {"mom": residual_momentum(synth["resid"]),
            "lr": low_risk_idiovol(synth["idio_var"])}
    alpha = build_composite_alpha(raws, synth["sector_map"], cfg)
    idio_vol = compute_idio_vol(synth["idio_var"])
    w = multifactor_weights(alpha, idio_vol, synth["sector_map"], cfg)
    active = w.loc[w.abs().sum(axis=1) > 0]
    assert (w.abs().max().max()) <= cfg.max_w + 1e-9          # per-name cap respected (exact)
    assert active.sum(axis=1).abs().max() < 0.05              # dollar-neutral (~)
    # sector-neutral: each sector nets ~0 on active days
    sectors = pd.Series(synth["sector_map"])
    last = active.iloc[-1]
    for s in sectors.unique():
        members = [c for c in last.index if synth["sector_map"][c] == s]
        assert abs(last[members].sum()) < 0.05


def test_weights_long_short_balanced(synth):
    cfg = Config()
    raws = {"mom": residual_momentum(synth["resid"]), "lr": low_risk_idiovol(synth["idio_var"])}
    alpha = build_composite_alpha(raws, synth["sector_map"], cfg)
    w = multifactor_weights(alpha, compute_idio_vol(synth["idio_var"]), synth["sector_map"], cfg)
    last = w.iloc[-1]
    assert (last > 0).any() and (last < 0).any()              # genuine long/short book
    assert abs(last[last > 0].sum() + last[last < 0].sum()) < 0.05


def test_high_52w_in_unit_interval(synth):
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (400, 60)), axis=0),
                         index=synth["idx"], columns=synth["cols"])
    h = high_52w(close).dropna()
    assert (h.values <= 1.0 + 1e-9).all() and (h.values > 0).all()  # ratio to trailing max


def test_weights_deterministic(synth):
    cfg = Config()
    raws = {"mom": residual_momentum(synth["resid"]), "lr": low_risk_idiovol(synth["idio_var"])}
    alpha = build_composite_alpha(raws, synth["sector_map"], cfg)
    iv = compute_idio_vol(synth["idio_var"])
    w1 = multifactor_weights(alpha, iv, synth["sector_map"], cfg)
    w2 = multifactor_weights(alpha, iv, synth["sector_map"], cfg)
    pd.testing.assert_frame_equal(w1, w2)
