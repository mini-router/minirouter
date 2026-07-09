"""Offline unit tests for dataset loading helpers and minibatch sampling."""
from __future__ import annotations

import json
import random

import pytest

import trinity.orchestration.dataset as D
from trinity.types import Task


def _task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        benchmark="math500",
        prompt=f"prompt-{task_id}",
        answer="1",
        meta={"source": "toy"},
    )


def test_sample_minibatch_without_replacement_is_distinct():
    tasks = [_task(str(i)) for i in range(5)]
    rng = random.Random(0)
    batch = D.sample_minibatch(tasks, 3, rng)
    assert len(batch) == 3
    assert len({t.task_id for t in batch}) == 3
    assert all(t in tasks for t in batch)


def test_sample_minibatch_with_replacement_when_pool_too_small():
    tasks = [_task("a"), _task("b")]
    rng = random.Random(1)
    batch = D.sample_minibatch(tasks, 4, rng)
    assert len(batch) == 4
    assert all(t in tasks for t in batch)


def test_sample_minibatch_is_deterministic_for_same_rng():
    tasks = [_task(str(i)) for i in range(6)]
    a = D.sample_minibatch(tasks, 3, random.Random(42))
    b = D.sample_minibatch(tasks, 3, random.Random(42))
    assert [t.task_id for t in a] == [t.task_id for t in b]


def test_sample_minibatch_rejects_empty_tasks():
    with pytest.raises(ValueError, match="empty task list"):
        D.sample_minibatch([], 1, random.Random(0))


def test_sample_minibatch_rejects_non_positive_m():
    tasks = [_task("a")]
    with pytest.raises(ValueError, match="must be positive"):
        D.sample_minibatch(tasks, 0, random.Random(0))


def test_load_tasks_is_deterministic_and_truncates():
    first = D.load_tasks("math500", "test", max_items=2, seed=7)
    second = D.load_tasks("math500", "test", max_items=2, seed=7)
    assert len(first) == 2
    assert [t.task_id for t in first] == [t.task_id for t in second]


def test_parse_lcb_tests_accepts_json_string_payload():
    payload = json.dumps([{"input": "1\n", "output": "1\n"}])
    assert D._parse_lcb_tests({"public_test_cases": payload}) == [
        {"input": "1\n", "output": "1\n"},
    ]


def test_parse_lcb_tests_normalizes_stdin_stdout_aliases():
    row = {"public_test_cases": [{"stdin": "3\n", "stdout": "9\n"}]}
    assert D._parse_lcb_tests(row) == [{"input": "3\n", "output": "9\n"}]


def test_parse_lcb_tests_returns_empty_for_unparseable_payload():
    assert D._parse_lcb_tests({"public_test_cases": "not-json"}) == []
    assert D._parse_lcb_tests({}) == []
