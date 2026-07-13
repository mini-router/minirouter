from __future__ import annotations

import asyncio

from trinity.orchestration.session import TrajectoryTimeoutError, run_trajectory
from trinity.types import Task
from trinity.types import Role


class _Policy:
    def decide(self, transcript_text: str, *, sample: bool, rng=None):
        return 0, Role.WORKER


class _Pool:
    def describe_model(self, model: str):
        return "openrouter", model

    async def chat(self, model, messages, **kwargs):
        await asyncio.sleep(0.05)
        return type("R", (), {"text": "answer", "prompt_tokens": 1, "completion_tokens": 1})()


def test_run_trajectory_times_out_and_reports_route(capsys):
    task = Task(task_id="task-1", benchmark="math500", prompt="question", answer="4")

    try:
        asyncio.run(
            run_trajectory(
                task,
                _Policy(),
                _Pool(),
                ["google/gemma-4-31B-turbo-TEE"],
                max_turns=1,
                request_timeout_s=0.001,
            )
        )
        raised = None
    except TrajectoryTimeoutError as exc:
        raised = exc

    out = capsys.readouterr().out
    assert "provider=openrouter" in out
    assert "model=google/gemma-4-31B-turbo-TEE" in out
    assert "task=task-1" in out
    assert raised is not None
    assert "timeout waiting for provider=openrouter" in str(raised)
