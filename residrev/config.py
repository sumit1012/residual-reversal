"""Single source of truth for all project parameters."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # Universe & data
    universe_size: int = 300  # most-liquid N names: low enough cost for reversal alpha to survive
    adv_window: int = 63
    hysteresis_buffer: int = 200
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    cache_dir: str = "cache/prices"
    data_dir: str = "data"

    # Factor model
    factor_window: int = 90
    min_obs: int = 60
    factors: tuple = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD")

    # Signal
    signal_k: int = 5
    winsorize_pct: float = 0.01
    signal_smooth_span: int = 5  # past-only rolling-mean smoothing of the signal (1 = off)
    signal_gap: int = 2  # skip the most-recent `signal_gap` days (avoid bid-ask-bounce reversal)

    # Portfolio optimizer
    gamma: float = 5.0
    lam_to: float = 3.0
    max_w: float = 0.02
    gross_cap: float = 2.0
    beta_tol: float = 1e-3
    sector_tol: float = 1e-3
    sigma_f_window: int = 252

    # Costs
    aum: float = 1e8  # portfolio AUM (USD); drives market-impact via participation = Δw·AUM/ADV
    eta_impact: float = 0.5
    adv_participation_cap: float = 0.10
    cs_smooth_window: int = 21

    # Conditioning
    amihud_window: int = 21
    vix_regime_window: int = 252
    n_illiq_buckets: int = 5
    n_vol_buckets: int = 3

    # Validation
    cpcv_n_groups: int = 6
    cpcv_k_test: int = 2
    cpcv_embargo: float = 0.01
    trials_log: str = "data/trials.jsonl"

    def to_dict(self) -> dict:
        """Serialize to plain dict (JSON-safe, for trial logging)."""
        d = dataclasses.asdict(self)
        # tuple fields become lists via asdict; convert back for round-trip fidelity
        d["factors"] = list(d["factors"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Deserialize from plain dict."""
        kw = dict(d)
        if "factors" in kw:
            kw["factors"] = tuple(kw["factors"])
        return cls(**kw)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Config":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
