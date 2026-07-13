"""Tests that per-role `decoding` config is applied by run_trajectory (issue #189).

The pool loads `decoding` from `configs/models*.yaml` onto `OpenAICompatiblePool.decoding`.
`run_trajectory` must apply the per-role temperature/top_p/max_tokens, with precedence:
explicit caller arg > per-role config > built-in default. Pools with no `decoding`
mapping (test stubs) fall through to the current defaults (no behavior change).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from trinity.orchestration.session import run_trajectory
from trinity.types import Role, Task


class RecordingPool:
    """Async chat stub that records the decoding kwargs of every call."""

    def __init__(self, decoding=None):
        self.decoding = decoding or {}
        self.calls: list[dict] = []

    async def chat(self, model, messages, *, temperature, top_p, max_tokens,
                   reasoning=None, client=None):
        self.calls.append(
            {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        )
        # Verifier turns terminate on ACCEPT; a prior Worker turn satisfies the guard.
        return SimpleNamespace(
            text="ok\nVERDICT: ACCEPT", prompt_tokens=1, completion_tokens=1,
            finish_reason="stop",
        )


class ScriptedPolicy:
    def __init__(self, seq):
        self.seq = seq
        self.i = 0

    def decide(self, transcript_text, *, sample=False, rng=None):
        a, r = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return a, r


_TASK = Task(task_id="t", benchmark="math500", prompt="2+2?", answer="4")
_MODELS = ["m0", "m1", "m2"]
# Thinker -> Worker -> Verifier(ACCEPT): three turns, one per role.
_SEQ = [(0, Role.THINKER), (1, Role.WORKER), (2, Role.VERIFIER)]


def _run(pool, **kw):
    policy = ScriptedPolicy(_SEQ)
    traj = asyncio.run(run_trajectory(_TASK, policy, pool, _MODELS, max_turns=5, **kw))
    return traj, pool.calls


def test_per_role_decoding_is_applied():
    decoding = {
        "thinker": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 111},
        "worker": {"temperature": 0.2, "top_p": 0.90, "max_tokens": 222},
        "verifier": {"temperature": 0.0, "max_tokens": 50},  # top_p omitted -> default
    }
    _, calls = _run(RecordingPool(decoding))
    assert len(calls) == 3
    assert calls[0] == {"temperature": 0.7, "top_p": 0.95, "max_tokens": 111}   # thinker
    assert calls[1] == {"temperature": 0.2, "top_p": 0.90, "max_tokens": 222}   # worker
    # verifier: top_p falls back to the built-in default (1.0)
    assert calls[2] == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 50}


def test_no_decoding_block_uses_builtin_defaults():
    _, calls = _run(RecordingPool(decoding={}))
    for c in calls:
        assert c == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096}


def test_pool_without_decoding_attr_uses_defaults():
    class Bare:
        async def chat(self, model, messages, *, temperature, top_p, max_tokens,
                       reasoning=None, client=None):
            self.last = (temperature, top_p, max_tokens)
            return SimpleNamespace(text="ok\nVERDICT: ACCEPT", prompt_tokens=1,
                                   completion_tokens=1, finish_reason="stop")

    pool = Bare()
    asyncio.run(run_trajectory(_TASK, ScriptedPolicy(_SEQ), pool, _MODELS, max_turns=5))
    assert pool.last == (0.0, 1.0, 4096)


def test_explicit_caller_arg_overrides_config():
    decoding = {"thinker": {"temperature": 0.7}, "worker": {"temperature": 0.2}}
    _, calls = _run(RecordingPool(decoding), temperature=0.33)
    # Explicit temperature wins for every role, regardless of the config block.
    assert all(c["temperature"] == 0.33 for c in calls)
