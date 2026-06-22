"""YAML experiment configuration (the run knobs of Methods, "Experimental protocol").

A :class:`Config` fixes the code, distance, noise regimes, physical error rate
``p_eval``, the threshold-curve grid ``p_grid``, Monte-Carlo ``trials`` and
neural ``train_trials``, the abstention threshold and the syndrome read-out
rate. ``configs/smoke.yaml`` is the fast CI configuration; ``configs/full.yaml``
is the reported scale (distance 9, 40,000 trials/regime) quoted in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 0
    code: str = "repetition"          # "repetition" or "surface"
    distance: int = 5                 # code distance d
    regimes: List[str] = field(default_factory=lambda: [
        "iid", "biased", "burst", "correlated", "measurement_error",
    ])
    p_eval: float = 0.12              # physical error rate for the headline table
    p_grid: List[float] = field(default_factory=lambda: [
        0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20,
    ])
    trials: int = 3000                # Monte-Carlo trials per (regime, decoder)
    train_trials: int = 2000          # syndromes used to fit the neural decoder
    abstain_threshold: float = 0.5    # commit only when confidence >= this
    q_meas: float = 0.05              # syndrome read-out error rate

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
