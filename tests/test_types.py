"""Offline unit tests for shared coordinator data types (``trinity.types``).

These dataclasses and enums are the integration contract across llm /
coordinator / roles / orchestration / optim. Locking field semantics and
computed properties prevents silent breakage in the coordination loop.
"""
from __future__ import annotations

from trinity.types import ROLE_ORDER, Role, Task, Trajectory, TurnRecord


def test_role_enum_values_and_string_coercion():
    assert Role.THINKER.value == "thinker"
    assert Role.WORKER.value == "worker"
    assert Role.VERIFIER.value == "verifier"
    assert Role("worker") is Role.WORKER


def test_role_order_matches_spec_head_layout():
    assert ROLE_ORDER == (Role.THINKER, Role.WORKER, Role.VERIFIER)
    assert len(ROLE_ORDER) == 3


def test_task_defaults_meta_to_empty_dict():
    task = Task(task_id="t1", benchmark="math500", prompt="2+2?", answer="4")
    assert task.meta == {}
    task.meta["split"] = "test"
    assert task.meta["split"] == "test"


def test_turn_record_stores_verifier_fields():
    rec = TurnRecord(
        turn=2,
        agent_name="glm-5p2",
        role=Role.VERIFIER,
        raw_output="Looks good.\nVERDICT: ACCEPT",
        processed_output="Looks good.\nVERDICT: ACCEPT",
        verdict="ACCEPT",
        prompt_tokens=12,
        completion_tokens=7,
    )
    assert rec.turn == 2
    assert rec.agent_name == "glm-5p2"
    assert rec.role is Role.VERIFIER
    assert rec.verdict == "ACCEPT"
    assert rec.prompt_tokens == 12
    assert rec.completion_tokens == 7


def test_trajectory_defaults_and_token_sum():
    task = Task(task_id="t1", benchmark="mmlu", prompt="Q?", answer="B")
    turns = [
        TurnRecord(
            turn=1,
            agent_name="deepseek-v4-pro",
            role=Role.WORKER,
            raw_output="B",
            processed_output="B",
            completion_tokens=5,
            prompt_tokens=10,
        ),
        TurnRecord(
            turn=2,
            agent_name="deepseek-v4-pro",
            role=Role.VERIFIER,
            raw_output="ok",
            processed_output="ok",
            verdict="REVISE",
            completion_tokens=3,
            prompt_tokens=8,
        ),
    ]
    traj = Trajectory(
        task=task,
        turns=turns,
        final_answer="B",
        reward=1.0,
        terminated_by="max_turns",
    )
    assert traj.n_turns == 2
    assert traj.total_completion_tokens == 8
    assert traj.final_answer == "B"
    assert traj.reward == 1.0
    assert traj.terminated_by == "max_turns"


def test_empty_trajectory_properties():
    task = Task(task_id="empty", benchmark="math500", prompt="?", answer="1")
    traj = Trajectory(task=task)
    assert traj.turns == []
    assert traj.n_turns == 0
    assert traj.total_completion_tokens == 0
    assert traj.final_answer == ""
    assert traj.reward is None
    assert traj.terminated_by == "max_turns"
