"""Offline unit tests for the inner coordination loop (SPEC §2/§4, smoke S4).

``orchestration/session.run_trajectory`` is the per-query routing loop: up to K
turns of (policy picks agent+role) -> (pool answers) -> (append), with a Verifier
ACCEPT terminating early only once a Worker output exists (SPEC §0.3.5). It is
provider/torch-agnostic by design ("tests pass a mock" / "a stub"), but the only
thing exercising it was the S4 smoke script — no pytest coverage of the
termination rule, the final-answer selection, or the agent-index wrap.

These lock that contract with a scripted mock policy and a scripted stub pool
(zero GPU, zero network). Each `run_trajectory` coroutine is driven with
``asyncio.run`` so no async test plugin is needed.
"""
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.orchestration.session import (  # noqa: E402
    _final_answer,
    _transcript_text,
    run_trajectory,
)
from trinity.types import Role, Task, TurnRecord  # noqa: E402

POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]


def _task() -> Task:
    return Task(task_id="t1", benchmark="math500", prompt="2+2?", answer="4")


@dataclass
class _Res:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _StubPool:
    """Returns scripted responses in call order; records the models it was asked."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.models_called: list[str] = []

    async def chat(self, model, messages, **kwargs):
        self.models_called.append(model)
        item = self._responses[min(len(self.models_called) - 1, len(self._responses) - 1)]
        return item if isinstance(item, _Res) else _Res(item)


class _ScriptPolicy:
    """Yields scripted (agent_idx, Role) decisions; repeats the last when exhausted."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.i = 0
        self.transcripts: list[str] = []

    def decide(self, transcript_text, *, sample=False, rng=None):
        self.transcripts.append(transcript_text)
        choice = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return choice


def _run(policy, pool, **kw):
    return asyncio.run(run_trajectory(_task(), policy, pool, POOL, **kw))


# --------------------------------------------------------------------------- #
# Termination rule
# --------------------------------------------------------------------------- #
def test_verifier_accept_after_worker_terminates():
    policy = _ScriptPolicy([(0, Role.WORKER), (1, Role.VERIFIER)])
    pool = _StubPool(["the answer is \\boxed{4}", "correct.\nVERDICT: ACCEPT"])
    traj = _run(policy, pool, max_turns=5)
    assert traj.terminated_by == "accept"
    assert len(traj.turns) == 2
    assert traj.turns[1].verdict == "ACCEPT"
    assert traj.final_answer == "the answer is \\boxed{4}"  # last Worker output


def test_turn1_verifier_accept_is_blocked_without_worker():
    # Verifier ACCEPT on turn 1 must NOT terminate (no Worker output yet).
    policy = _ScriptPolicy([(0, Role.VERIFIER)])
    pool = _StubPool(["VERDICT: ACCEPT"])
    traj = _run(policy, pool, max_turns=3, verifier_requires_prior_worker=True)
    assert traj.terminated_by == "max_turns"
    assert len(traj.turns) == 3


def test_turn1_verifier_accept_allowed_when_guard_disabled():
    policy = _ScriptPolicy([(0, Role.VERIFIER)])
    pool = _StubPool(["VERDICT: ACCEPT"])
    traj = _run(policy, pool, max_turns=3, verifier_requires_prior_worker=False)
    assert traj.terminated_by == "accept"
    assert len(traj.turns) == 1


def test_verifier_revise_does_not_terminate():
    policy = _ScriptPolicy([(0, Role.WORKER), (1, Role.VERIFIER), (0, Role.WORKER)])
    pool = _StubPool(["w1", "needs work\nVERDICT: REVISE", "w2"])
    traj = _run(policy, pool, max_turns=3)
    assert traj.terminated_by == "max_turns"
    assert len(traj.turns) == 3
    assert traj.turns[1].verdict == "REVISE"


def test_runs_to_max_turns_without_accept():
    policy = _ScriptPolicy([(0, Role.WORKER)])
    pool = _StubPool(["answer \\boxed{1}"])
    traj = _run(policy, pool, max_turns=4)
    assert traj.terminated_by == "max_turns"
    assert len(traj.turns) == 4


# --------------------------------------------------------------------------- #
# Agent selection
# --------------------------------------------------------------------------- #
def test_agent_index_wraps_modulo_pool_size():
    # An out-of-range agent index is taken modulo the pool size (7 % 3 == 1).
    policy = _ScriptPolicy([(7, Role.WORKER)])
    pool = _StubPool(["x"])
    traj = _run(policy, pool, max_turns=1)
    assert traj.turns[0].agent_name == POOL[1]
    assert pool.models_called == [POOL[1]]


# --------------------------------------------------------------------------- #
# Final-answer selection (O_τ)
# --------------------------------------------------------------------------- #
def test_final_answer_prefers_last_worker_over_later_nonworker():
    policy = _ScriptPolicy([(0, Role.WORKER), (1, Role.THINKER)])
    pool = _StubPool(["worker-out", "thinker-out"])
    traj = _run(policy, pool, max_turns=2)
    assert traj.final_answer == "worker-out"  # Worker preferred even though earlier


def test_final_answer_falls_back_to_last_nonverifier():
    traj_turns = [
        TurnRecord(1, POOL[0], Role.THINKER, "t1", "t1"),
        TurnRecord(2, POOL[1], Role.THINKER, "t2", "t2"),
    ]

    @dataclass
    class _T:
        turns: list

    assert _final_answer(_T(traj_turns)) == "t2"


def test_final_answer_empty_when_no_turns():
    @dataclass
    class _T:
        turns: list

    assert _final_answer(_T([])) == ""


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_nonverifier_turns_have_no_verdict_and_tokens_recorded():
    policy = _ScriptPolicy([(0, Role.WORKER)])
    pool = _StubPool([_Res("w", prompt_tokens=11, completion_tokens=7)])
    traj = _run(policy, pool, max_turns=1)
    tr = traj.turns[0]
    assert tr.verdict is None
    assert tr.prompt_tokens == 11 and tr.completion_tokens == 7
    assert traj.reward is None  # reward is scored later, not here


def test_reasoning_kwarg_is_filtered_for_a_narrow_stub():
    # A stub whose chat() lacks `reasoning`/`client` must still be called cleanly
    # (session._filter_supported drops unsupported kwargs).
    calls = {}

    class _NarrowPool:
        async def chat(self, model, messages, *, temperature=0.0, top_p=1.0, max_tokens=4096):
            calls["ok"] = True
            return _Res("ans")

    traj = _run(_ScriptPolicy([(0, Role.WORKER)]), _NarrowPool(), max_turns=1, reasoning="minimal")
    assert calls.get("ok") is True
    assert traj.turns[0].processed_output == "ans"


def test_transcript_text_grows_with_prior_outputs():
    task = _task()
    assert _transcript_text(task, []) == "QUERY:\n2+2?"
    turns = [TurnRecord(1, POOL[0], Role.WORKER, "raw", "processed-out")]
    text = _transcript_text(task, turns)
    assert "QUERY:\n2+2?" in text
    assert "processed-out" in text
    assert "worker" in text  # role tag rendered
