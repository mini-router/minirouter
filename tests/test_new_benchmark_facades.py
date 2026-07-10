"""Offline unit tests for the gsm8k/humaneval/bbh benchmark facades.

Mirrors the existing benchmarks.livecodebench facade test: each facade must
delegate to trinity.orchestration.dataset.load_tasks with its own fixed
benchmark name.
"""
from __future__ import annotations

import benchmarks.bbh as BBH
import benchmarks.gsm8k as GSM8K
import benchmarks.humaneval as HUMANEVAL
import trinity.orchestration.dataset as D


def test_gsm8k_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(GSM8K, "load_tasks", fake_load_tasks)

    assert GSM8K.load("test", max_items=5, seed=1) == ["ok"]
    assert seen["args"] == ("gsm8k", "test", 5, 1)


def test_humaneval_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(HUMANEVAL, "load_tasks", fake_load_tasks)

    assert HUMANEVAL.load("test", max_items=2, seed=0) == ["ok"]
    assert seen["args"] == ("humaneval", "test", 2, 0)


def test_bbh_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(BBH, "load_tasks", fake_load_tasks)

    assert BBH.load("test", max_items=None, seed=9) == ["ok"]
    assert seen["args"] == ("bbh", "test", None, 9)


def test_bbh_facade_exposes_all_27_subtasks():
    assert len(BBH.BBH_SUBTASKS) == 27
    assert BBH.BBH_SUBTASKS == D.BBH_SUBTASKS


def test_gsm8k_and_humaneval_and_bbh_are_supported_benchmarks():
    assert {"gsm8k", "humaneval", "bbh"} <= set(D.SUPPORTED_BENCHMARKS)
