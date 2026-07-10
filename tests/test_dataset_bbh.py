"""Offline unit tests for the BBH loader in trinity.orchestration.dataset.

Pure stdlib, no network/GPU: HF loading is monkeypatched per-subtask, and the
toy-fallback path is exercised directly.
"""
from __future__ import annotations

import trinity.orchestration.dataset as D


def test_bbh_pools_rows_across_subtasks(monkeypatch):
    rows_by_subtask = {
        "sports_understanding": [
            {"input": "Is this plausible? \"X threw a pass.\"\n(A) yes\n(B) no", "target": "(A)"}
        ],
        "web_of_lies": [{"input": "Does Alice tell the truth?", "target": "No"}],
    }

    seen_configs = []

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen_configs.append(name)
        if path != "lukaemon/bbh":
            return None
        return rows_by_subtask.get(name)

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("bbh", "test", max_items=None, seed=0)

    # every one of the 27 canonical subtasks is attempted
    assert set(seen_configs) == set(D.BBH_SUBTASKS)
    assert len(tasks) == 2
    by_subtask = {t.meta["subtask"]: t for t in tasks}
    assert by_subtask["sports_understanding"].answer == "(A)"
    assert by_subtask["web_of_lies"].answer == "No"
    assert all(t.benchmark == "bbh" for t in tasks)
    assert all("Answer:" in t.prompt for t in tasks)


def test_bbh_skips_rows_missing_input_or_target(monkeypatch):
    # "navigate" has two malformed rows (skipped); "word_sorting" has one valid
    # row, so the overall result is non-empty and the toy fallback is NOT
    # triggered -- isolating the row-filtering behavior from the fallback path.
    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        if name == "navigate":
            return [{"input": "", "target": "Yes"}, {"input": "Go 3 steps.", "target": ""}]
        if name == "word_sorting":
            return [{"input": "Sort: b, a.", "target": "a b"}]
        return None

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("bbh", "test", max_items=None, seed=0)
    assert len(tasks) == 1
    assert tasks[0].meta["subtask"] == "word_sorting"


def test_bbh_falls_back_to_toy_set_offline(monkeypatch):
    monkeypatch.setattr(D, "_try_load_hf", lambda *a, **kw: None)

    tasks = D.load_tasks("bbh", "test", max_items=None, seed=0)

    assert len(tasks) == 2
    assert all(t.benchmark == "bbh" for t in tasks)
