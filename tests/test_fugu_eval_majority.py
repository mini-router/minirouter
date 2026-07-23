"""Offline tests for the per-query strict-majority rule in ``trinity.fugu.eval``.

Regression coverage for issue #83: a tied ballot (e.g. votes ``[1, 0]``) must
NOT be banked as a solved query in ``per_query_binary``, because that value feeds
``scripts/oracle_ceiling.py`` and a coin-flip counted as ``1`` is exactly the
partial credit this harness's contract forbids. Pure asyncio, no network/LLMs:
``propose_and_run`` and ``is_correct`` are stubbed to script the votes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import trinity.fugu.eval as E
from trinity.types import Task


def _fake_run():
    """A minimal object that satisfies CostMeter.add_run and the parse-gate."""
    return SimpleNamespace(
        parsed_ok=True,
        model_tokens={},
        n_llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _run_eval(monkeypatch, votes, reps):
    """Drive evaluate() for a single task whose reps grade to ``votes``."""
    seq = iter(votes)

    async def fake_propose_and_run(*a, **k):
        return _fake_run()

    monkeypatch.setattr(E, "propose_and_run", fake_propose_and_run)
    # is_correct is called once per rep, in order; hand back the scripted votes.
    monkeypatch.setattr(E, "is_correct", lambda run, task: next(seq))

    task = Task(task_id="q1", benchmark="math500", prompt="?", answer="x")
    res = asyncio.run(
        E.evaluate(
            conductor=None,
            tasks=[task],
            pool=None,
            pool_models=["m"],
            reps=reps,
        )
    )
    return res.per_query_binary["q1"]


@pytest.mark.parametrize(
    "votes, reps, expected",
    [
        ([1, 0], 2, 0),          # even tie -> not solved
        ([1, 1, 0, 0], 4, 0),    # even tie -> not solved
        ([1, 0, 0], 3, 0),       # clear minority
        ([1, 1, 0], 3, 1),       # clear majority
        ([1], 1, 1),             # single correct
        ([0], 1, 0),             # single wrong
        ([1, 1], 2, 1),          # unanimous correct
    ],
)
def test_strict_majority(monkeypatch, votes, reps, expected):
    assert _run_eval(monkeypatch, votes, reps) == expected


def test_truncated_odd_ballot_tie_is_not_solved(monkeypatch):
    # reps=3 requested but only 2 votes recorded (spend cap would truncate);
    # simulate by scripting exactly 2 votes for a 2-iteration run.
    assert _run_eval(monkeypatch, [1, 0], 2) == 0
