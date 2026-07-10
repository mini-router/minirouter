"""Offline unit tests for the GSM8K loader in trinity.orchestration.dataset.

Pure stdlib, no network/GPU: HF loading is monkeypatched, and the toy-fallback
path is exercised directly.
"""
from __future__ import annotations

import trinity.orchestration.dataset as D


def test_gsm8k_hf_row_parses_final_answer(monkeypatch):
    row = {
        "question": "Natalia sold clips to 12 friends, then half as many in May. Total?",
        "answer": "She sold 12 in April.\nShe sold 12 / 2 = <<12/2=6>>6 in May.\n"
        "#### 18",
    }

    seen = {}

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen.setdefault("calls", []).append({"path": path, "name": name, "split": split})
        if path != "openai/gsm8k":
            return None
        return [row]

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("gsm8k", "test", max_items=None, seed=0)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.benchmark == "gsm8k"
    assert task.task_id == "gsm8k-0"
    assert task.answer == "18"
    assert "boxed" in task.prompt.lower()
    assert seen["calls"][0] == {"path": "openai/gsm8k", "name": "main", "split": "test"}


def test_gsm8k_final_answer_strips_thousands_comma():
    assert D._gsm8k_final_answer("blah blah\n#### 1,234") == "1234"


def test_gsm8k_row_without_marker_is_skipped(monkeypatch):
    rows = [
        {"question": "no marker here", "answer": "no hash marks at all"},
        {"question": "valid", "answer": "reasoning\n#### 7"},
    ]

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        return rows if path == "openai/gsm8k" else None

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("gsm8k", "test", max_items=None, seed=0)
    assert len(tasks) == 1
    assert tasks[0].answer == "7"


def test_gsm8k_falls_back_to_toy_set_offline(monkeypatch):
    monkeypatch.setattr(D, "_try_load_hf", lambda *a, **kw: None)

    tasks = D.load_tasks("gsm8k", "test", max_items=None, seed=0)

    assert len(tasks) == 2
    assert {t.meta["source"] for t in tasks} == {"toy"}
    assert all(t.benchmark == "gsm8k" for t in tasks)


def test_gsm8k_toy_set_is_deterministic(monkeypatch):
    monkeypatch.setattr(D, "_try_load_hf", lambda *a, **kw: None)

    first = D.load_tasks("gsm8k", "test", max_items=None, seed=3)
    second = D.load_tasks("gsm8k", "test", max_items=None, seed=3)

    assert [t.task_id for t in first] == [t.task_id for t in second]
