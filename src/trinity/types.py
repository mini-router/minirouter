"""Shared data types for the TRINITY coordinator.

These are the integration contract used across llm / coordinator / roles /
orchestration / optim. Keep field names stable — many modules import these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    THINKER = "thinker"
    WORKER = "worker"
    VERIFIER = "verifier"


# Canonical ordering for the head's role logits (indices n_models .. n_models+2).
ROLE_ORDER: tuple[Role, ...] = (Role.THINKER, Role.WORKER, Role.VERIFIER)


@dataclass
class Task:
    """One benchmark item. `answer` is whatever the reward checker needs."""

    task_id: str
    benchmark: str          # "math500" | "mmlu" | "livecodebench" | "gpqa" | ...
    prompt: str             # the question text presented to a pool LLM
    answer: object          # reference answer / test spec consumed by reward.score
    meta: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    turn: int               # 1-indexed
    agent_name: str         # pool model short name, e.g. "deepseek-v4-pro"
    role: Role
    raw_output: str         # M_k (verbatim model output)
    processed_output: str   # O_k (post-processed, appended to transcript)
    verdict: str | None = None   # "ACCEPT" | "REVISE" | None (verifier turns only)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Trajectory:
    task: Task
    turns: list[TurnRecord] = field(default_factory=list)
    final_answer: str = ""
    reward: float | None = None          # set by reward.score; in {0, 1}
    terminated_by: str = "max_turns"     # "accept" | "max_turns"

    @property
    def total_completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)
