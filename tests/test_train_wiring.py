"""Wiring tests for warm-start (#2) + shaped fitness (#3) integration.

Torch-free (no GPU on the dev box): exercises the numpy/config contracts that
train.py relies on — warm-theta length validation, FitnessConfig parsing/defaults,
and the eval-stays-binary invariant at the grader level.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

from trinity.coordinator import params as P  # noqa: E402
from trinity.coordinator import warmstart as WS  # noqa: E402
from trinity.optim.fitness import FitnessConfig  # noqa: E402
from trinity.orchestration import reward as R  # noqa: E402

_CONFIG = _REPO / "configs" / "trinity.yaml"


# ---- warm-theta loading + length validation (what _resolve_x0 calls) --------
def test_warmstart_theta_roundtrip(tmp_path):
    spec = P.make_spec()
    Wa = np.full((3, spec.head_shape[1]), 0.1)
    theta = WS.pack_warmstart_theta(Wa, spec)
    p = tmp_path / "warm.npy"
    np.save(p, theta)
    loaded = WS.load_warmstart_theta(str(p), spec)
    assert loaded.shape == (spec.n_total,)
    assert np.allclose(loaded, theta)


def test_warmstart_theta_wrong_length_rejected(tmp_path):
    spec = P.make_spec()
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros(spec.n_total - 5))   # layout mismatch
    with pytest.raises(ValueError):
        WS.load_warmstart_theta(str(p), spec)


def test_warmstart_theta_length_matches_default_init():
    spec = P.make_spec()
    assert WS.pack_warmstart_theta(np.zeros((3, spec.head_shape[1])), spec).size == \
        P.initial_theta(spec).size == spec.n_total


# ---- fitness config parsing + defaults (training-only shaping) --------------
def test_fitness_config_defaults_off():
    # from_dict(None) and {} reproduce the original mean-binary behavior knobs.
    for d in (None, {}):
        cfg = FitnessConfig.from_dict(d)
        assert cfg.enable_reweight is False
        assert cfg.format_bonus == 0.05 and cfg.turn_penalty == 0.05
        assert cfg.hero_dense is False


def test_fitness_config_from_real_yaml():
    cfg = yaml.safe_load(_CONFIG.read_text())
    fc = FitnessConfig.from_dict(cfg.get("fitness"))
    # the committed config keeps reweighting OFF (safe default)
    assert fc.enable_reweight is False
    # a fully-zeroed shaping config must be a perfect no-op
    off = FitnessConfig(enable_reweight=False, format_bonus=0.0, turn_penalty=0.0, hero_dense=False)
    assert off.shaping_active is False


# ---- eval-stays-binary invariant (grader level) ----------------------------
def test_eval_grader_is_binary():
    # The eval path scores with reward.score_text and must be pure {0,1}.
    good = R.score_text("math500", "The answer is \\boxed{4}.", "4")
    bad = R.score_text("math500", "The answer is \\boxed{5}.", "4")
    assert good in (0.0, 1.0) and bad in (0.0, 1.0)
    assert good == 1.0 and bad == 0.0
