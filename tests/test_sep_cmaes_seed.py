"""Offline reproducibility tests for SepCMAES seeding (``trinity.optim.sep_cmaes``).

Regression coverage for the seed==0 bug: pycma ignores ``seed==0`` and leaves
numpy's global RNG unseeded, so the documented reproducible default was in fact
non-reproducible. These tests are gated on ``cma`` being importable (skipped
otherwise) and make no network/GPU calls.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cma")

from trinity.optim.sep_cmaes import SepCMAES  # noqa: E402


def _first_pop(seed):
    return np.asarray(SepCMAES(n=32, sigma0=0.1, seed=seed, maxiter=5).ask())


def test_seed_zero_is_reproducible():
    # The documented default (seed=0) must be repeatable.
    assert np.allclose(_first_pop(0), _first_pop(0))


def test_nonzero_seed_still_reproducible():
    assert np.allclose(_first_pop(7), _first_pop(7))


def test_distinct_seeds_differ():
    # Remapping 0 must not collide with the remap target's own behavior in a way
    # that makes different seeds produce identical first populations.
    assert not np.allclose(_first_pop(0), _first_pop(1))


def test_caller_seed_value_is_preserved():
    # self.seed keeps the caller's original value even though pycma receives the
    # remapped one.
    assert SepCMAES(n=8, sigma0=0.1, seed=0, maxiter=1).seed == 0
