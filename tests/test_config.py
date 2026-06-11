"""10 tests for residrev.config.Config."""

import dataclasses
import json

import pytest

from residrev.config import Config


def test_default_instantiation():
    """Config instantiates with defaults without error."""
    cfg = Config()
    assert cfg.universe_size == 1000


def test_frozen_raises_on_assignment():
    """Assigning to a field raises FrozenInstanceError."""
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.universe_size = 500  # type: ignore[misc]


def test_to_dict_has_all_keys():
    """to_dict returns a dict containing every Config field."""
    cfg = Config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    for f in dataclasses.fields(cfg):
        assert f.name in d, f"Missing key: {f.name}"


def test_from_dict_round_trip():
    """from_dict(to_dict(cfg)) reconstructs an equal Config."""
    cfg = Config()
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_to_json_is_valid_json():
    """to_json returns a valid JSON string."""
    cfg = Config()
    s = cfg.to_json()
    assert isinstance(s, str)
    parsed = json.loads(s)  # must not raise
    assert isinstance(parsed, dict)


def test_from_json_round_trip():
    """from_json(to_json(cfg)) reconstructs an equal Config."""
    cfg = Config()
    assert Config.from_json(cfg.to_json()) == cfg


def test_custom_override():
    """Custom keyword arguments override defaults correctly."""
    cfg = Config(signal_k=3, gamma=2.5, universe_size=500)
    assert cfg.signal_k == 3
    assert cfg.gamma == 2.5
    assert cfg.universe_size == 500
    # Other fields remain at defaults
    assert cfg.adv_window == 63


def test_factors_is_tuple_of_six_strings():
    """factors field is a tuple of exactly 6 strings."""
    cfg = Config()
    assert isinstance(cfg.factors, tuple)
    assert len(cfg.factors) == 6
    assert all(isinstance(f, str) for f in cfg.factors)


def test_numeric_fields_are_positive():
    """All numeric fields that should be positive are > 0."""
    cfg = Config()
    positive_fields = [
        "universe_size", "adv_window", "hysteresis_buffer",
        "factor_window", "min_obs", "signal_k",
        "gamma", "lam_to", "max_w", "gross_cap",
        "sigma_f_window", "eta_impact", "adv_participation_cap",
        "cs_smooth_window", "amihud_window", "vix_regime_window",
        "n_illiq_buckets", "n_vol_buckets",
        "cpcv_n_groups", "cpcv_k_test",
    ]
    for name in positive_fields:
        val = getattr(cfg, name)
        assert val > 0, f"{name} should be positive, got {val}"


def test_string_paths_are_non_empty():
    """trials_log and cache_dir are non-empty strings."""
    cfg = Config()
    assert isinstance(cfg.trials_log, str) and len(cfg.trials_log) > 0
    assert isinstance(cfg.cache_dir, str) and len(cfg.cache_dir) > 0
