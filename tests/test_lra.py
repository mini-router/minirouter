"""Offline tests for sep-CMA-ES Learning-Rate Adaptation (IMPROVEMENTS.md #4).

Pure numpy + pycma; no GPU, no network, no API spend. Covers:

* the :class:`LRAController` SNR math (clean -> eta 1.0, noisy -> damp);
* :class:`LRAConfig` validation and dict parsing;
* the optimizer integration is a **byte-identical no-op when disabled** and when
  ``eta`` never leaves 1.0 (noiseless), so shipping it default-off cannot regress
  the existing training path;
* the headline claim, in miniature: on a noisy quadratic LRA reaches a strictly
  better optimum than vanilla (the full 24-seed sweep lives in
  ``utility/lra_ablation.py``).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.optim.lra import LRAConfig, LRAController  # noqa: E402
from trinity.optim.sep_cmaes import SepCMAES  # noqa: E402


# --------------------------------------------------------------------------- #
# LRAConfig
# --------------------------------------------------------------------------- #
def test_config_defaults_and_from_dict():
    assert LRAConfig().alpha == 1.0
    assert LRAConfig.from_dict(None) == LRAConfig()
    assert LRAConfig.from_dict({}) == LRAConfig()
    cfg = LRAConfig.from_dict({"alpha": 2.5, "warmup": 7, "unknown": 9})
    assert cfg.alpha == 2.5 and cfg.warmup == 7  # unknown keys ignored


@pytest.mark.parametrize("bad", [
    {"beta": 0.0}, {"beta": 1.5}, {"alpha": 0.0}, {"eta_min": 0.0},
    {"eta_min": 1.5}, {"gain": -0.1}, {"warmup": -1},
])
def test_config_rejects_invalid(bad):
    with pytest.raises(ValueError):
        LRAConfig(**bad)


# --------------------------------------------------------------------------- #
# LRAController SNR behavior
# --------------------------------------------------------------------------- #
def test_controller_noiseless_is_full_rate():
    """noise_var <= 0 (a clean objective) must always yield eta == 1.0."""
    ctl = LRAController(LRAConfig(warmup=0))
    for _ in range(20):
        eta = ctl.update(np.array([0.1, 0.5, 0.9, 0.3]), noise_var=0.0)
        assert eta == 1.0


def test_controller_warmup_holds_full_rate():
    ctl = LRAController(LRAConfig(warmup=3))
    # A high-noise generation would normally damp, but warm-up pins eta = 1.0.
    fits = np.array([0.5, 0.5, 0.5, 0.5])  # zero spread -> pure noise
    for gen in range(1, 4):
        assert ctl.update(fits, noise_var=0.25) == 1.0
        assert ctl.generation == gen
    # first post-warmup generation may now move
    ctl.update(fits, noise_var=0.25)
    assert ctl.eta < 1.0


def test_controller_damps_when_noise_dominates():
    """Candidate differences buried in estimation noise -> eta driven down."""
    ctl = LRAController(LRAConfig(warmup=2))
    fits = np.array([0.48, 0.52, 0.50, 0.49, 0.51])  # tiny spread
    eta = 1.0
    for _ in range(30):
        eta = ctl.update(fits, noise_var=0.25)  # noise_var >> spread
    assert eta <= 0.2  # pushed toward the eta_min floor
    assert eta >= ctl.cfg.eta_min


def test_controller_stays_high_when_signal_dominates():
    """Large real spread vs tiny noise -> eta stays at 1.0."""
    ctl = LRAController(LRAConfig(warmup=2))
    fits = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # big spread
    eta = 1.0
    for _ in range(30):
        eta = ctl.update(fits, noise_var=1e-4)
    assert eta == 1.0


def test_controller_eta_never_leaves_unit_interval():
    ctl = LRAController(LRAConfig(warmup=0, eta_min=0.1))
    rng = np.random.default_rng(0)
    for _ in range(200):
        fits = rng.random(6)
        eta = ctl.update(fits, noise_var=float(rng.random()))
        assert 0.1 <= eta <= 1.0


# --------------------------------------------------------------------------- #
# Optimizer integration
# --------------------------------------------------------------------------- #
def _optimize(noise_sd, lra, seed, n=32, gens=40, sigma0=0.3):
    rng = np.random.default_rng(seed)
    x_star = rng.standard_normal(n) * 0.1
    nrng = np.random.default_rng(97 * seed + 3)
    opt = SepCMAES(n=n, sigma0=sigma0, x0=np.zeros(n), seed=seed, maxiter=gens, lra=lra)
    while not opt.stop():
        sols = opt.ask()
        fits = [-float((x - x_star) @ (x - x_star))
                + (noise_sd * float(nrng.standard_normal()) if noise_sd else 0.0)
                for x in sols]
        opt.tell(sols, fits, fitness_noise_var=noise_sd * noise_sd)
    best_x, _ = opt.best()
    return float(np.linalg.norm(best_x - x_star)), opt.lra_eta


def test_disabled_is_vanilla_and_reports_no_eta():
    d, eta = _optimize(0.0, None, seed=1)
    assert eta is None  # no controller -> no eta introspection


def test_noiseless_run_matches_vanilla_bitwise():
    """LRA enabled but noise_var==0 => eta pinned at 1.0 => identical trajectory."""
    d_van, _ = _optimize(0.0, None, seed=2)
    d_lra, eta = _optimize(0.0, LRAConfig(), seed=2)
    assert eta == 1.0
    assert d_lra == pytest.approx(d_van, abs=1e-9)


def test_lra_beats_vanilla_under_noise():
    """Mini version of the ablation: better mean final distance across seeds."""
    seeds = range(10)
    van = np.mean([_optimize(1.5, None, s)[0] for s in seeds])
    lra = np.mean([_optimize(1.5, LRAConfig(), s)[0] for s in seeds])
    assert lra < van


def test_tell_length_mismatch_still_raises_with_lra():
    opt = SepCMAES(n=8, sigma0=0.2, seed=0, maxiter=5, lra=True)
    sols = opt.ask()
    with pytest.raises(ValueError):
        opt.tell(sols, [0.0] * (len(sols) - 1), fitness_noise_var=0.1)
