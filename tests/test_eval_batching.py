from __future__ import annotations

import asyncio

from trinity.eval import _score_submission_policy
from trinity.types import Task, Trajectory


def _task(idx: int) -> Task:
    return Task(
        task_id=f"math-{idx}",
        benchmark="math500",
        prompt=f"question {idx}",
        answer="4",
    )


def test_submission_eval_runs_items_in_batches(monkeypatch):
    async def run() -> tuple[float, int]:
        active = 0
        max_active = 0
        started = 0
        gate = asyncio.Event()

        async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
            nonlocal active, max_active, started
            active += 1
            max_active = max(max_active, active)
            started += 1
            if started >= 2:
                gate.set()
            await gate.wait()
            await asyncio.sleep(0)
            active -= 1
            return Trajectory(task=task, final_answer="\\boxed{4}")

        monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)

        tasks = [_task(i) for i in range(4)]
        score = await _score_submission_policy(
            tasks,
            policy=None,
            pool=None,
            pool_models=[],
            sample=False,
            batch_size=2,
            max_turns=1,
            max_tokens=1,
            reasoning=None,
        )
        return score, max_active

    score, max_active = asyncio.run(run())

    assert score == 1.0
    assert max_active == 2


def test_submission_eval_keeps_going_after_item_failure(monkeypatch):
    async def run() -> float:
        started = []

        async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
            started.append(task.task_id)
            if task.task_id == "math-1":
                raise RuntimeError("boom")
            return Trajectory(task=task, final_answer="\\boxed{4}")

        monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)

        tasks = [_task(1), _task(2)]
        score = await _score_submission_policy(
            tasks,
            policy=None,
            pool=None,
            pool_models=[],
            sample=False,
            batch_size=2,
            max_turns=1,
            max_tokens=1,
            reasoning=None,
        )
        assert started == ["math-1", "math-2"]
        return score

    score = asyncio.run(run())

    assert score == 0.5
