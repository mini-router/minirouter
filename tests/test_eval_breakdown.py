"""Offline unit tests for the optional per-item breakdown recording in eval.py.

Mirrors the mocking pattern in test_eval_batching.py (fake run_trajectory, no
network/GPU). Confirms the new `records` parameter is purely additive: the
returned pooled score is unchanged, and each recorded item carries the task's
benchmark/subtask/score.
"""
from __future__ import annotations

import asyncio

from trinity.eval import _score_policy, _score_submission_policy, _task_record
from trinity.types import Task, Trajectory


def _bbh_task(idx: int, subtask: str) -> Task:
    return Task(
        task_id=f"bbh-{subtask}-{idx}",
        benchmark="bbh",
        prompt=f"question {idx}",
        answer="(A)",
        meta={"subtask": subtask, "source": "toy"},
    )


def test_score_policy_records_are_additive_and_do_not_change_score(monkeypatch):
    async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
        return Trajectory(task=task, final_answer="Answer: (A)")

    monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)

    tasks = [_bbh_task(0, "sports_understanding"), _bbh_task(1, "navigate")]

    async def run():
        records: list[dict] = []
        score = await _score_policy(
            tasks,
            policy=None,
            pool=None,
            pool_models=[],
            sample=False,
            batch_size=2,
            records=records,
            max_turns=1,
            max_tokens=1,
            reasoning=None,
        )
        return score, records

    score, records = asyncio.run(run())

    assert score == 1.0
    assert len(records) == 2
    by_id = {r["task_id"]: r for r in records}
    assert by_id["bbh-sports_understanding-0"]["subtask"] == "sports_understanding"
    assert by_id["bbh-navigate-1"]["subtask"] == "navigate"
    assert all(r["benchmark"] == "bbh" for r in records)
    assert all(r["score"] == 1.0 for r in records)


def test_score_policy_records_default_none_is_a_noop(monkeypatch):
    async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
        return Trajectory(task=task, final_answer="Answer: (A)")

    monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)

    async def run():
        return await _score_policy(
            [_bbh_task(0, "navigate")],
            policy=None,
            pool=None,
            pool_models=[],
            sample=False,
            batch_size=1,
            max_turns=1,
            max_tokens=1,
            reasoning=None,
        )

    score = asyncio.run(run())
    assert score == 1.0


def test_score_submission_policy_records_populate(monkeypatch):
    async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
        return Trajectory(task=task, final_answer="Answer: (B)")

    monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)

    async def run():
        records: list[dict] = []
        score = await _score_submission_policy(
            [_bbh_task(0, "web_of_lies")],
            policy=None,
            pool=None,
            pool_models=[],
            sample=False,
            batch_size=1,
            records=records,
            max_turns=1,
            max_tokens=1,
            reasoning=None,
        )
        return score, records

    score, records = asyncio.run(run())
    assert score == 0.0  # gold is "(A)", model answered "(B)"
    assert records[0]["subtask"] == "web_of_lies"
    assert records[0]["score"] == 0.0


def test_task_record_handles_missing_meta():
    task = Task(task_id="t0", benchmark="math500", prompt="2+2?", answer="4")
    traj = Trajectory(task=task, final_answer="\\boxed{4}")

    record = _task_record(traj)

    assert record == {"task_id": "t0", "benchmark": "math500", "subtask": None, "score": 1.0}
