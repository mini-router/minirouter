"""Regression: SepCMAES.best() ships the distribution mean under noisy fitness.

King / submission PRs save ``best_theta.npy`` from ``es.best()``. With a noisy
objective, pycma ``xbest`` is the luckiest eval; ``xfavorite`` (mean) is the
estimate that generalizes. See issues #234 / #69.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.optim.sep_cmaes import SepCMAES


def _noise_free_sphere(x: np.ndarray, target: np.ndarray) -> float:
    d = x - target
    return -float(np.dot(d, d))


def test_best_returns_xfavorite_when_available() -> None:
    n = 32
    opt = SepCMAES(n=n, sigma0=0.5, seed=0, maxiter=5, popsize=8)
    rng = np.random.default_rng(1)
    target = rng.standard_normal(n) * 0.1

    while not opt.stop():
        sols = opt.ask()
        fits = [_noise_free_sphere(x, target) for x in sols]
        opt.tell(sols, fits)

    shipped, _ = opt.best()
    favorite = np.asarray(opt._es.result.xfavorite, dtype=float)
    assert shipped.shape == (n,)
    np.testing.assert_allclose(shipped, favorite, rtol=0, atol=0)


def test_shipped_mean_beats_xbest_on_noise_free_fitness() -> None:
    """Under additive eval noise, noise-free J(xfavorite) >= J(xbest)."""
    n = 64
    opt = SepCMAES(n=n, sigma0=0.4, seed=2, maxiter=12, popsize=12)
    rng = np.random.default_rng(3)
    target = rng.standard_normal(n) * 0.05
    noise_rng = np.random.default_rng(4)

    while not opt.stop():
        sols = opt.ask()
        fits = []
        for x in sols:
            clean = _noise_free_sphere(x, target)
            # Large noise so the luckiest eval is often not the true best.
            fits.append(clean + float(noise_rng.normal(0.0, 0.35)))
        opt.tell(sols, fits)

    shipped, _ = opt.best()
    xbest, _ = opt.incumbent()
    j_ship = _noise_free_sphere(shipped, target)
    j_xbest = _noise_free_sphere(xbest, target)
    assert j_ship >= j_xbest - 1e-12, (
        f"expected noise-free J(ship) >= J(xbest); got {j_ship:.6f} < {j_xbest:.6f}"
    )


def test_incumbent_matches_pycma_xbest() -> None:
    n = 16
    opt = SepCMAES(n=n, sigma0=0.3, seed=5, maxiter=3, popsize=6)
    target = np.zeros(n)

    while not opt.stop():
        sols = opt.ask()
        opt.tell(sols, [_noise_free_sphere(x, target) for x in sols])

    xbest, fbest = opt.incumbent()
    lib_x = np.asarray(opt._es.result.xbest, dtype=float)
    lib_f = -float(opt._es.result.fbest)
    np.testing.assert_allclose(xbest, lib_x)
    assert fbest == pytest.approx(lib_f)
