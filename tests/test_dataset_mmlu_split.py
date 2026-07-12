"""Offline regression tests for MMLU split handling.

These tests verify that the public loader resolves the logical training split
to the real MMLU training pool instead of falling back to the toy dataset.
They avoid importing private helpers so the test stays stable across refactors.
"""
from __future__ import annotations

import trinity.orchestration.dataset as D


def test_mmlu_train_split_maps_to_auxiliary_train(monkeypatch):
    seen = {}

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen["call"] = (path, name, split, version_tag)
        return [
            {
                "question": "Which planet is closest to the Sun?",
                "choices": ["Venus", "Earth", "Mercury", "Mars"],
                "answer": 2,
                "subject": "astronomy",
            }
        ]

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("mmlu", "train", max_items=None, seed=0)

    assert seen["call"] == ("cais/mmlu", "all", "auxiliary_train", None)
    assert len(tasks) == 1
    assert tasks[0].benchmark == "mmlu"
    assert tasks[0].answer == "C"


def test_mmlu_validation_split_maps_through(monkeypatch):
    seen = {}

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen["call"] = (path, name, split, version_tag)
        return [
            {
                "question": "Which planet is closest to the Sun?",
                "choices": ["Venus", "Earth", "Mercury", "Mars"],
                "answer": 2,
                "subject": "astronomy",
            }
        ]

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("mmlu", "validation", max_items=None, seed=0)

    assert seen["call"] == ("cais/mmlu", "all", "validation", None)
    assert len(tasks) == 1
    assert tasks[0].answer == "C"
