"""Offline unit tests for the Fugu Conductor's honest eval harness.

``fugu/eval.evaluate`` is the number that would go on a leaderboard for the
Conductor track: it reports **pure binary** ``is_correct`` (never the shaped
training reward), averages ``reps`` samples per task to denoise, emits the
per-query majority binary that feeds ``scripts/oracle_ceiling.py``, and honours a
spend cap. None of that aggregation logic was covered by a test.

These lock it with a stub conductor + stub pool and the REAL parse-gate/executor
and grader (so the accuracy really is end-to-end), zero GPU/network. Each async
`evaluate` is driven with ``asyncio.run``.
"""
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.fugu.cost import CONDUCTOR_KEY  # noqa: E402
from trinity.fugu.eval import evaluate  # noqa: E402
from trinity.types import Task  # noqa: E402

# A minimal valid 1-step workflow (worker 0 solves; reads only the query).
VALID_WF = 'model_id = [0]\nsubtasks = ["solve it"]\naccess_list = [[]]'
INVALID_WF = "Sorry, I cannot produce a workflow here."  # no lists -> parse fail


@dataclass
class _Prop:
    text: str
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class _Res:
    text: str
    prompt_tokens: int = 1000
    completion_tokens: int = 1000


class _StubConductor:
    def __init__(self, text: str = VALID_WF) -> None:
        self.text = text
        self.samples: list[bool] = []

    async def propose(self, task, pool_models, *, sample=False, rng=None, client=None):
        self.samples.append(sample)
        return _Prop(self.text)


class _StubPool:
    """`respond(messages, call_index) -> worker_text`; records token cost per call."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.n = 0

    async def chat(self, model, messages, **kwargs):
        i = self.n
        self.n += 1
        return _Res(self.respond(messages, i))


POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]


def _tasks(specs) -> list[Task]:
    # specs: list of (task_id, answer)
    return [Task(task_id=tid, benchmark="math500", prompt=f"[{tid}] 2+2?", answer=ans)
            for tid, ans in specs]


def _ev(conductor, tasks, pool, **kw):
    return asyncio.run(evaluate(conductor, tasks, pool, POOL, **kw))


def _boxed(x) -> str:
    return f"the answer is \\boxed{{{x}}}"


# --------------------------------------------------------------------------- #
# Accuracy / parse_rate aggregation
# --------------------------------------------------------------------------- #
def test_all_correct_single_rep():
    tasks = _tasks([("q1", "4"), ("q2", "4")])
    pool = _StubPool(lambda m, i: _boxed(4))  # always correct
    res = _ev(_StubConductor(), tasks, pool)
    assert res.accuracy == 1.0
    assert res.parse_rate == 1.0
    assert res.n_tasks == 2
    assert res.per_query_binary == {"q1": 1, "q2": 1}


def test_mixed_tasks_give_half_accuracy():
    # q1 answer 1 -> worker says 1 (correct); q2 answer 1 -> worker says 9 (wrong).
    tasks = _tasks([("q1", "1"), ("q2", "1")])

    def respond(messages, i):
        content = messages[-1]["content"]
        return _boxed(1) if "[q1]" in content else _boxed(9)

    res = _ev(_StubConductor(), tasks, _StubPool(respond))
    assert res.accuracy == 0.5
    assert res.per_query_binary == {"q1": 1, "q2": 0}


def test_parse_gate_failure_scores_zero():
    tasks = _tasks([("q1", "4")])
    pool = _StubPool(lambda m, i: _boxed(4))  # never reached
    res = _ev(_StubConductor(text=INVALID_WF), tasks, pool)
    assert res.accuracy == 0.0
    assert res.parse_rate == 0.0
    assert res.per_query_binary == {"q1": 0}
    assert pool.n == 0  # a rejected proposal runs no worker calls


def test_empty_task_list():
    res = _ev(_StubConductor(), [], _StubPool(lambda m, i: _boxed(4)))
    assert res.accuracy == 0.0 and res.parse_rate == 0.0 and res.n_tasks == 0


# --------------------------------------------------------------------------- #
# reps: per-query majority binary + denoising
# --------------------------------------------------------------------------- #
def test_majority_binary_rounds_up_on_majority_correct():
    # reps=3 -> votes [1, 1, 0]: acc 2/3, majority (2*2 >= 3) -> 1.
    tasks = _tasks([("q1", "4")])
    outs = [_boxed(4), _boxed(4), _boxed(9)]
    res = _ev(_StubConductor(), tasks, _StubPool(lambda m, i: outs[i]), reps=3)
    assert res.per_task["q1"]["reps_correct"] == [1, 1, 0]
    assert abs(res.per_task["q1"]["acc"] - 2 / 3) < 1e-9
    assert res.per_query_binary["q1"] == 1


def test_majority_binary_zero_when_minority_correct():
    tasks = _tasks([("q1", "4")])
    outs = [_boxed(4), _boxed(9), _boxed(9)]  # votes [1,0,0]
    res = _ev(_StubConductor(), tasks, _StubPool(lambda m, i: outs[i]), reps=3)
    assert res.per_query_binary["q1"] == 0


def test_reps_gt_one_samples_reps_one_greedy():
    tasks = _tasks([("q1", "4")])
    c2 = _StubConductor()
    _ev(c2, tasks, _StubPool(lambda m, i: _boxed(4)), reps=2)
    assert c2.samples == [True, True]  # sampled draws
    c1 = _StubConductor()
    _ev(c1, tasks, _StubPool(lambda m, i: _boxed(4)), reps=1)
    assert c1.samples == [False]  # greedy


# --------------------------------------------------------------------------- #
# Spend cap
# --------------------------------------------------------------------------- #
def test_cap_usd_aborts_and_stops_starting_tasks():
    tasks = _tasks([("q1", "4"), ("q2", "4"), ("q3", "4")])
    prices = {"deepseek-v4-pro": (1.0, 1.0), CONDUCTOR_KEY: (0.0, 0.0)}
    # Each run costs (1000+1000)/1e6 * 1.0 = 0.002 > cap 0.001 -> abort after run 1.
    res = _ev(
        _StubConductor(), tasks, _StubPool(lambda m, i: _boxed(4)),
        prices=prices, cap_usd=0.001, concurrency=1,
    )
    assert res.aborted is True
    assert res.n_tasks < 3  # not every task got to run
    assert res.cost["aborted"] is True
