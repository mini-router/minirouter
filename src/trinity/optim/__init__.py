"""Evolutionary training (separable CMA-ES) + baseline optimizers."""
from __future__ import annotations

from trinity.optim.sep_cmaes import SepCMAES, default_popsize, run
from trinity.optim.baselines import (
    RSConfig,
    SFTConfig,
    ReinforceConfig,
    run_random_search,
    run_sft,
    run_reinforce,
)

__all__ = [
    "SepCMAES",
    "default_popsize",
    "run",
    "RSConfig",
    "SFTConfig",
    "ReinforceConfig",
    "run_random_search",
    "run_sft",
    "run_reinforce",
]
