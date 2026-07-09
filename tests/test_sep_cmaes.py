"""Tests for the sep-CMA-ES recommendation policy (#69).

Under TRINITY's stochastic minibatch fitness, ``SepCMAES.best()`` must return the
distribution mean (pycma ``xfavorite``) — the noise-averaged recommendation — and
not the single luckiest evaluated candidate (``xbest``), which overfits sampling
noise when it is saved as ``best_theta.npy``.

CPU-only and torch-free; skipped cleanly on boxes without the optional ``cma``
package (it is a runtime dependency, so CI installs it).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

pytest.importorskip("cma", reason="pycma is required to exercise SepCMAES")

from trinity.optim.sep_cmaes import SepCMAES, run  # noqa: E402


def _drive(opt: SepCMAES, target: np.ndarray, gens: int) -> None:
    """Run a few deterministic ask/tell generations toward ``target``."""
    for _ in range(gens):
        sols = opt.ask()
        fits = [-float(np.sum((x - target) ** 2)) for x in sols]
        opt.tell(sols, fits)


def test_best_returns_distribution_mean_not_xbest():
    """best() must ship the CMA-ES mean (xfavorite), not the noisy xbest."""
    opt = SepCMAES(n=8, sigma0=0.5, seed=0, maxiter=20)
    target = np.arange(8) * 0.1
    _drive(opt, target, gens=8)

    result = opt._es.result
    # The two candidates genuinely differ here, so the test is meaningful.
    assert not np.allclose(result.xbest, result.xfavorite)

    best_x, _ = opt.best()
    assert np.allclose(best_x, result.xfavorite), "best() must return xfavorite (the mean)"
    assert not np.allclose(best_x, result.xbest), "best() must NOT return xbest (the noisy incumbent)"


def test_best_raises_before_first_tell():
    """The no-evaluation-yet contract is preserved after the xfavorite switch."""
    opt = SepCMAES(n=4, sigma0=0.3, seed=0, maxiter=5)
    with pytest.raises(RuntimeError):
        opt.best()


def test_best_reports_best_observed_fitness():
    """The reported fitness is the running max of observed (maximization) fitness."""
    opt = SepCMAES(n=6, sigma0=0.4, seed=0, maxiter=20)
    target = np.zeros(6)
    _drive(opt, target, gens=6)

    _, best_f = opt.best()
    assert best_f == opt._best_f
    # Every observed fitness is <= 0 for this objective; the best is the largest.
    assert best_f <= 0.0


def test_run_history_best_fitness_stays_monotone():
    """run()'s logged best_fitness must remain non-decreasing (S7 smoke contract)."""
    rng = np.random.default_rng(0)
    target = rng.standard_normal(16)

    def objective(x):
        return -float(np.sum((x - target) ** 2))

    best_x, best_f, history = run(objective, 16, sigma0=0.5, maxiter=25, seed=0)
    series = [h["best_fitness"] for h in history]
    assert all(series[i] <= series[i + 1] + 1e-12 for i in range(len(series) - 1))
    assert history[-1]["best_fitness"] > history[0]["best_fitness"]
    assert best_x.shape == (16,)
