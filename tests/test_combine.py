"""Tests for residrev.combine — synthetic return streams."""
import numpy as np
import pandas as pd

from residrev import combine


def _two_sleeves(n=500, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    a = pd.Series(rng.normal(0.0004, 0.005, n), index=dates)
    b = pd.Series(rng.normal(0.0003, 0.011, n), index=dates)
    return {"reversal": a, "trend": b}


def test_align_sleeves_common_dates():
    s = _two_sleeves()
    s["trend"] = s["trend"].iloc[5:]  # misalign
    panel = combine.align_sleeves(s)
    assert list(panel.columns) == ["reversal", "trend"]
    assert panel.isna().sum().sum() == 0


def test_sleeve_weights_sum_to_one():
    s = _two_sleeves()
    panel = combine.align_sleeves(s)
    for scheme in ("equal", "risk_parity"):
        w = combine.sleeve_weights(panel, scheme=scheme)
        row_sums = w.sum(axis=1).iloc[80:]  # after the vol warmup
        assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_combine_outputs_and_targets_vol():
    s = _two_sleeves()
    out = combine.combine(s, scheme="risk_parity", target_vol=0.10)
    assert isinstance(out["combined"], pd.Series)
    realized_vol = out["combined"].std() * np.sqrt(252)
    assert 0.04 < realized_vol < 0.20  # roughly near the 10% target


def test_diversification_report_structure():
    s = _two_sleeves()
    df = combine.diversification_report(s, freeze="2019-06-01")
    for book in ("reversal", "trend", "combined_equal", "combined_risk_parity"):
        assert book in df.index
    for col in ("full_sharpe", "insample_sharpe", "live_sharpe", "max_dd_pct"):
        assert col in df.columns
    assert df.attrs["correlation"].shape == (2, 2)


def test_combination_lowers_drawdown_for_uncorrelated_sleeves():
    """Two near-uncorrelated positive-drift sleeves: combined max DD <= worse sleeve's."""
    s = _two_sleeves(seed=7)
    df = combine.diversification_report(s, freeze="2019-06-01")
    worst_sleeve_dd = min(df.loc["reversal", "max_dd_pct"], df.loc["trend", "max_dd_pct"])
    assert df.loc["combined_risk_parity", "max_dd_pct"] >= worst_sleeve_dd  # less negative
