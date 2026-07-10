"""Offline unit tests for the HumanEval loader in trinity.orchestration.dataset.

Pure stdlib, no network/GPU: HF loading is monkeypatched, and the toy-fallback
path is exercised directly.
"""
from __future__ import annotations

import trinity.orchestration.dataset as D


def test_humaneval_hf_row_wraps_check_harness(monkeypatch):
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "entry_point": "add",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }

    seen = {}

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen.setdefault("calls", []).append({"path": path, "split": split})
        if path != "openai/openai_humaneval":
            return None
        return [row]

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("humaneval", "test", max_items=None, seed=0)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.benchmark == "humaneval"
    assert task.task_id == "HumanEval/0"
    assert task.answer["fn_name"] == "add"
    assert task.answer["tests"] == [
        "def check(candidate):\n    assert candidate(1, 2) == 3\n\ncheck(add)\n"
    ]
    assert "```python" in task.prompt
    assert seen["calls"][0] == {"path": "openai/openai_humaneval", "split": "test"}


def test_humaneval_row_missing_fields_is_skipped(monkeypatch):
    rows = [
        {"task_id": "HumanEval/1", "prompt": "", "entry_point": "f", "test": "def check(c): pass"},
        {
            "task_id": "HumanEval/2",
            "prompt": "def f(): pass\n",
            "entry_point": "f",
            "test": "def check(candidate):\n    assert candidate() is None\n",
        },
    ]

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        return rows if path == "openai/openai_humaneval" else None

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("humaneval", "test", max_items=None, seed=0)
    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/2"


def test_humaneval_falls_back_to_toy_set_offline(monkeypatch):
    monkeypatch.setattr(D, "_try_load_hf", lambda *a, **kw: None)

    tasks = D.load_tasks("humaneval", "test", max_items=None, seed=0)

    assert len(tasks) == 2
    assert all(t.benchmark == "humaneval" for t in tasks)
    assert all("tests" in t.answer for t in tasks)
