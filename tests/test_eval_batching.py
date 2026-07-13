from __future__ import annotations

import asyncio

from trinity.eval import _score_policy, _score_submission_policy
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


def _scores_each_trajectory_once(score_fn, monkeypatch):
    """Drive ``score_fn`` and assert ``reward.score`` runs exactly once per task.

    Scoring a code trajectory executes the candidate in a subprocess sandbox, so
    a redundant second call doubles evaluation cost and — for nondeterministic
    candidates — can make the logged per-item score disagree with the aggregate.
    """
    calls: dict[str, int] = {}

    async def fake_run_trajectory(task, policy, pool, pool_models, **kwargs):
        return Trajectory(task=task, final_answer="\\boxed{4}")

    def counting_score(traj) -> float:
        calls[traj.task.task_id] = calls.get(traj.task.task_id, 0) + 1
        return 1.0

    monkeypatch.setattr("trinity.eval.run_trajectory", fake_run_trajectory)
    monkeypatch.setattr("trinity.orchestration.reward.score", counting_score)

    tasks = [_task(i) for i in range(4)]
    score = asyncio.run(
        score_fn(
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
    )

    assert score == 1.0
    # One score() per task — not two (no redundant re-scoring on aggregation).
    assert calls == {f"math-{i}": 1 for i in range(4)}


def test_submission_eval_scores_each_trajectory_once(monkeypatch):
    _scores_each_trajectory_once(_score_submission_policy, monkeypatch)


def test_eval_scores_each_trajectory_once(monkeypatch):
    _scores_each_trajectory_once(_score_policy, monkeypatch)
