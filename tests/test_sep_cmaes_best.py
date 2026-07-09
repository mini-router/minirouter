"""Regression tests for SepCMAES.best() — issue #69.

`best()` must return the CMA-ES *distribution mean* (``xfavorite``), NOT the
single best-*evaluated* candidate (``xbest``). Under a noisy objective — which is
exactly what the TRINITY binary reward is (a small sampled minibatch, sampled
policy) — ``xbest`` is the luckiest draw of the run and overfits sampling noise,
so shipping it as ``best_theta.npy`` degrades the submission.

Two levels of check:
  * structural — best() == the mean, incumbent() == xbest, and they differ;
  * behavioral — averaged over many seeds, the mean's *noise-free* fitness beats
    the lucky incumbent's (the acceptance criterion in the issue). The advantage
    is an expectation property (noise cancels in the mean), so it is asserted in
    aggregate, not per-seed.

pycma (`cma`) is required; the test skips cleanly when it is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

pytest.importorskip("cma")

from trinity.optim.sep_cmaes import SepCMAES  # noqa: E402

# Noisy-optimization regime tuned so the mean-vs-lucky gap is decisive (z ~ 3.6
# over 40 seeds) yet the run stays sub-second. See PR #69 discussion.
_N = 32
_NOISE = 0.8
_SIGMA0 = 0.5
_MAXITER = 40


def _run_noisy(seed: int):
    """Optimize a noisy sphere; return (opt, mean_x, xbest, clean_fn, target)."""
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(_N) * 0.1

    def clean(x):  # maximize; noise-free peak (0.0) at ``target``
        return -float(np.sum((np.asarray(x) - target) ** 2))

    opt = SepCMAES(n=_N, sigma0=_SIGMA0, seed=seed + 1, maxiter=_MAXITER)
    noise_rng = np.random.default_rng(1000 + seed)
    while not opt.stop():
        sols = opt.ask()
        fits = [clean(x) + float(noise_rng.normal(0.0, _NOISE)) for x in sols]
        opt.tell(sols, fits)

    mean_x, _ = opt.best()
    xbest, _ = opt.incumbent()
    return opt, mean_x, xbest, clean, target


def test_best_raises_before_first_tell():
    opt = SepCMAES(n=_N, sigma0=0.1, seed=0, maxiter=5)
    with pytest.raises(RuntimeError):
        opt.best()
    with pytest.raises(RuntimeError):
        opt.incumbent()


def test_best_returns_distribution_mean_not_incumbent():
    """best() is the mean (xfavorite); incumbent() is xbest; they differ."""
    opt, mean_x, xbest, _, _ = _run_noisy(seed=0)

    # best() must be the distribution mean, not the best-evaluated candidate.
    assert np.allclose(mean_x, np.asarray(opt._es.result.xfavorite, dtype=float))
    # incumbent() must be pycma's xbest (the raw best eval), exposed for logging.
    assert np.allclose(xbest, np.asarray(opt._es.result.xbest, dtype=float))
    # Under a noisy objective the two are genuinely different vectors — proving
    # best() is not silently returning xbest again.
    assert not np.allclose(mean_x, xbest)


def test_mean_beats_lucky_incumbent_under_noise():
    """Averaged over seeds, the mean's noise-free fitness >= the incumbent's.

    This is the issue #69 acceptance criterion: best() (the mean) must not be
    worse than the luckiest single evaluation once the sampling noise it exploits
    is removed.
    """
    n_seeds = 40
    clean_mean = np.empty(n_seeds)
    clean_xbest = np.empty(n_seeds)
    for i in range(n_seeds):
        _, mean_x, xbest, clean, _ = _run_noisy(seed=i)
        clean_mean[i] = clean(mean_x)
        clean_xbest[i] = clean(xbest)

    # Aggregate (expectation) form: the mean is the better recommendation.
    assert clean_mean.mean() > clean_xbest.mean(), (
        f"mean's avg noise-free fitness {clean_mean.mean():.4f} did not beat the "
        f"lucky incumbent's {clean_xbest.mean():.4f} — best() may be returning xbest"
    )


def test_best_is_finite_and_right_shape():
    _, mean_x, _, _, _ = _run_noisy(seed=3)
    assert mean_x.shape == (_N,)
    assert np.isfinite(mean_x).all()
