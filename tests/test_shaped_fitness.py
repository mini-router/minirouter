"""Offline, numpy-only unit tests for the TRAINING-ONLY shaped CMA-ES fitness
(improvement #3).

These tests import ONLY the pure shaping cores from ``trinity.optim.fitness`` and
the format predicate from ``trinity.orchestration.reward``. They make NO live API
calls and import NO torch (directly or transitively) — they run on the dev box
with numpy alone.

Coverage mandated by the plan:
  * shaped_reward formula, incl. "correct always outranks wrong"
  * shaped_reward collapses to plain binary when bonuses are zero
  * variance_reweight up-weights high-variance tasks
  * variance_reweight is a no-op (uniform) when all rewards are equal / disabled
  * default-OFF config reproduces a plain task mean exactly
  * the EVAL path (reward.score / reward.score_text) stays pure binary
"""
import sys

import numpy as np
import pytest

from trinity.optim import fitness as F
from trinity.optim.fitness import FitnessConfig, shaped_reward, variance_reweight
from trinity.orchestration import reward as R


# ---------------------------------------------------------------------------
# Torch-free invariant: nothing imported here may pull in torch.
# ---------------------------------------------------------------------------
def test_no_torch_imported():
    assert "torch" not in sys.modules, "shaped-fitness tests must not import torch"


# ---------------------------------------------------------------------------
# FitnessConfig
# ---------------------------------------------------------------------------
def test_config_defaults_and_from_dict():
    d = FitnessConfig()
    assert d.enable_reweight is False
    assert d.format_bonus == 0.05
    assert d.turn_penalty == 0.05
    assert d.hero_dense is False
    assert d.shaping_active is True  # nonzero bonuses

    empty = FitnessConfig.from_dict(None)
    assert empty == FitnessConfig()
    assert FitnessConfig.from_dict({}) == FitnessConfig()

    custom = FitnessConfig.from_dict(
        {"enable_reweight": True, "format_bonus": 0.1, "turn_penalty": 0.0, "hero_dense": True}
    )
    assert custom.enable_reweight is True
    assert custom.format_bonus == 0.1
    assert custom.turn_penalty == 0.0
    assert custom.hero_dense is True


def test_shaping_inactive_when_zeroed():
    cfg = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=False)
    assert cfg.shaping_active is False


# ---------------------------------------------------------------------------
# shaped_reward
# ---------------------------------------------------------------------------
def test_shaped_reward_formula_exact():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.05)
    # correct=1, has_answer=True, 1 turn of 5 -> 1 + 0.05*1 - 0.05*0 = 1.05
    assert shaped_reward(1, True, 1, 5, cfg) == pytest.approx(1.05)
    # correct=1, has_answer=True, 5 turns of 5 -> 1 + 0.05 - 0.05*(4/4) = 1.0
    assert shaped_reward(1, True, 5, 5, cfg) == pytest.approx(1.0)
    # correct=0, has_answer=True, 3 turns of 5 -> 0 + 0.05 - 0.05*(2/4) = 0.025
    assert shaped_reward(0, True, 3, 5, cfg) == pytest.approx(0.025)
    # correct=0, has_answer=False, 5 turns of 5 -> 0 - 0.05 = -0.05
    assert shaped_reward(0, False, 5, 5, cfg) == pytest.approx(-0.05)


def test_shaped_reward_turn_penalty_denominator_guard():
    # max_turns=1 -> denom = max(1, 0) = 1, single-turn frac = 0 (no penalty).
    cfg = FitnessConfig(format_bonus=0.0, turn_penalty=0.05)
    assert shaped_reward(1, False, 1, 1, cfg) == pytest.approx(1.0)
    # num_turns > max_turns is clamped to a full penalty, never overshoots.
    assert shaped_reward(1, False, 99, 5, cfg) == pytest.approx(1.0 - 0.05)


def test_correct_always_outranks_wrong():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.05)
    # worst-case correct (max penalty, no format bonus) vs best-case wrong
    # (format bonus, no penalty): 0.95 must still beat 0.05.
    worst_correct = shaped_reward(1, False, 5, 5, cfg)
    best_wrong = shaped_reward(0, True, 1, 5, cfg)
    assert worst_correct > best_wrong
    # Exhaustive sweep over plausible turn counts.
    for nt in range(1, 6):
        for ha_c in (True, False):
            for ha_w in (True, False):
                c = shaped_reward(1, ha_c, nt, 5, cfg)
                w = shaped_reward(0, ha_w, 1, 5, cfg)
                assert c > w


def test_shaped_reward_collapses_to_binary_when_off():
    cfg = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=False)
    for correct in (0, 1):
        for ha in (True, False):
            for nt in (1, 3, 5):
                assert shaped_reward(correct, ha, nt, 5, cfg) == float(correct)


# ---------------------------------------------------------------------------
# variance_reweight
# ---------------------------------------------------------------------------
def test_variance_reweight_upweights_high_variance_tasks():
    cfg = FitnessConfig(enable_reweight=True)
    # task 0: everyone wrong (sigma 0). task 1: split (high sigma). task 2: everyone right (sigma 0).
    m = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    w = variance_reweight(m, cfg)
    assert w.shape == (3,)
    # The split (high-variance) task is up-weighted above the flat tasks.
    assert w[1] > w[0]
    assert w[1] > w[2]
    # Flat tasks (below-mean sigma) sit near the 0.5 floor; the disputed one near the top.
    assert w[0] < 1.0 and w[2] < 1.0
    assert w[1] > 1.0


def test_variance_reweight_noop_when_all_equal():
    cfg = FitnessConfig(enable_reweight=True)
    # All-equal rewards -> all sigma == 0 == mean_sigma -> uniform weights.
    m = np.ones((5, 4))
    w = variance_reweight(m, cfg)
    assert np.allclose(w, 1.0)
    m0 = np.zeros((3, 2))
    assert np.allclose(variance_reweight(m0, cfg), 1.0)
    # Every task equally variable -> still uniform.
    m_eq = np.array([[0.0, 0.0], [1.0, 1.0]])  # each column sigma identical
    assert np.allclose(variance_reweight(m_eq, cfg), 1.0)


def test_variance_reweight_noop_when_disabled():
    cfg = FitnessConfig(enable_reweight=False)
    m = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])  # task 1 high variance
    w = variance_reweight(m, cfg)
    assert np.allclose(w, 1.0), "disabled reweight must yield uniform weights"


def test_variance_reweight_empty_tasks():
    cfg = FitnessConfig(enable_reweight=True)
    w = variance_reweight(np.zeros((3, 0)), cfg)
    assert w.shape == (0,)


# ---------------------------------------------------------------------------
# weighted candidate fitness: default-OFF == plain mean
# ---------------------------------------------------------------------------
def test_default_off_weighted_mean_equals_plain_mean():
    cfg = FitnessConfig()  # enable_reweight False
    per_task_rows = [
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
    ]
    matrix = np.vstack(per_task_rows)
    weights = variance_reweight(matrix, cfg)  # uniform under default-OFF
    for row in per_task_rows:
        weighted = F._candidate_fitness(row, weights)
        assert weighted == pytest.approx(float(row.mean()))


def test_reweighted_mean_differs_from_plain_when_on():
    cfg = FitnessConfig(enable_reweight=True)
    rows = [
        np.array([1.0, 1.0, 1.0]),  # task0 flat-1, task1 split, task2 flat-1
        np.array([1.0, 0.0, 1.0]),
    ]
    matrix = np.vstack(rows)
    w = variance_reweight(matrix, cfg)
    # The candidate that scored the disputed task differently from the mean
    # gets a fitness that is NOT the plain mean (the split task is reweighted).
    wf = F._candidate_fitness(rows[1], w)
    assert wf != pytest.approx(float(rows[1].mean()))


# ---------------------------------------------------------------------------
# reward.has_answer (format predicate reused by the format bonus)
# ---------------------------------------------------------------------------
def test_has_answer_math_choice_code():
    assert R.has_answer("math500", r"the answer is \boxed{42}") is True
    assert R.has_answer("math500", "the result is 17 apples") is True
    assert R.has_answer("math500", "no numbers here at all") is False
    assert R.has_answer("mmlu", "The answer is (C).") is True
    assert R.has_answer("mmlu", "I am not sure which option.") is False
    assert R.has_answer("livecodebench", "```python\ndef f(): pass\n```") is True
    assert R.has_answer("livecodebench", "just prose, no code") is False
    assert R.has_answer("math500", "") is False
    assert R.has_answer("unknown_bench", "anything") is False


# ---------------------------------------------------------------------------
# EVAL PATH INVARIANT: reward.score / score_text stay pure binary.
# ---------------------------------------------------------------------------
def test_eval_path_stays_binary():
    # The eval-facing scorer must only ever return 0.0 or 1.0, regardless of any
    # training-side shaping config (which it does not consult at all).
    assert R.score_text("math500", r"\boxed{42}", "42") == 1.0
    assert R.score_text("math500", r"\boxed{41}", "42") == 0.0
    assert R.score_text("mmlu", "The answer is B.", "B") == 1.0
    assert R.score_text("mmlu", "The answer is A.", "B") == 0.0
    for v in (
        R.score_text("math500", r"\boxed{42}", "42"),
        R.score_text("math500", "wrong", "42"),
        R.score_text("mmlu", "B", "B"),
    ):
        assert v in (0.0, 1.0)


def test_shaped_module_does_not_mutate_eval_scorer():
    # reward.score must be unaffected by importing/constructing FitnessConfig.
    _ = FitnessConfig(format_bonus=0.5, turn_penalty=0.5, enable_reweight=True)
    assert R.score_text("math500", r"\boxed{7}", "7") == 1.0
    assert R.score_text("math500", r"\boxed{8}", "7") == 0.0
