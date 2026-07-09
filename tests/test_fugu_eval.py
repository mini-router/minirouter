"""Offline tests for the per-query binary emitted by ``trinity.fugu.eval``.

No network, no GPU: ``propose_and_run`` and ``is_correct`` are monkeypatched, so
``evaluate()`` runs against scripted votes.

``per_query_binary`` is the input ``scripts/oracle_ceiling.py`` consumes as
``--trinity-per-query`` for its McNemar test, so a tied ballot must resolve to 0.
A 50/50 query banked as 1 is the partial credit the module docstring rules out.
"""
from __future__ import annotations

import asyncio

import trinity.fugu.eval as fe
from trinity.fugu.eval import _majority, evaluate
from trinity.fugu.workflow import WorkflowRun
from trinity.types import Task


# --- the pure rule -------------------------------------------------------


def test_majority_is_strict_so_a_tie_is_not_a_win():
    assert _majority([1, 0]) == 0
    assert _majority([0, 1]) == 0
    assert _majority([1, 1, 0, 0]) == 0


def test_majority_of_unanimous_and_clear_ballots():
    assert _majority([1]) == 1
    assert _majority([0]) == 0
    assert _majority([1, 1]) == 1
    assert _majority([0, 0]) == 0
    assert _majority([1, 1, 0]) == 1  # 2/3 -> win
    assert _majority([1, 0, 0]) == 0  # 1/3 -> loss


def test_majority_never_counts_a_half_correct_ballot_as_solved():
    """Any exactly-half ballot is a loss, at every even width."""
    for half in range(1, 5):
        votes = [1] * half + [0] * half
        assert _majority(votes) == 0, votes


# --- evaluate() end to end ------------------------------------------------


_TASK = Task(task_id="q1", benchmark="math500", prompt="2+2?", answer="4")


def _install_scripted_votes(monkeypatch, votes, *, cost_per_run=0.0):
    """Make each rep return a run costing ``cost_per_run``, graded by ``votes``."""
    model_tokens = {"m": (1_000_000, 0)} if cost_per_run else {}

    async def fake_propose_and_run(*a, **kw):
        return WorkflowRun(workflow=None, parsed_ok=True, model_tokens=dict(model_tokens))

    remaining = list(votes)

    def fake_is_correct(run, task):
        return remaining.pop(0)

    monkeypatch.setattr(fe, "propose_and_run", fake_propose_and_run)
    monkeypatch.setattr(fe, "is_correct", fake_is_correct)


def _run(**kw):
    return asyncio.run(evaluate(object(), [_TASK], object(), ["m"], **kw))


def test_even_reps_tie_is_recorded_as_unsolved(monkeypatch):
    """--reps 2 with a 1-0 split must not bank the query as solved."""
    _install_scripted_votes(monkeypatch, [1, 0])
    res = _run(reps=2)
    assert res.per_task["q1"]["acc"] == 0.5
    assert res.per_query_binary["q1"] == 0


def test_odd_reps_truncated_by_the_spend_cap_still_breaks_ties_correctly(monkeypatch):
    """The cap can end a task mid-reps, so an ODD --reps can still tie.

    Each rep costs $1.00. With reps=3 and cap_usd=1.50 the meter aborts after the
    second rep, leaving a 2-vote ballot [1, 0] -- a tie, which must score 0.
    """
    _install_scripted_votes(monkeypatch, [1, 0, 1], cost_per_run=1.0)
    res = _run(reps=3, prices={"m": (1.0, 0.0)}, cap_usd=1.50)

    assert res.aborted is True
    assert res.per_task["q1"]["reps_correct"] == [1, 0]  # third rep never ran
    assert res.per_query_binary["q1"] == 0


def test_clear_majority_still_scores_one(monkeypatch):
    _install_scripted_votes(monkeypatch, [1, 1, 0])
    res = _run(reps=3)
    assert res.per_query_binary["q1"] == 1
