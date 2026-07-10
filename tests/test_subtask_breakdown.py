"""Offline unit tests for scripts/subtask_breakdown.py.

These tests exercise only the pure `group_by_subtask` / `render` / `summarize`
functions (and `load_records`' validation). No network/GPU; no dependency on a
real `trinity.eval --breakdown-out` run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subtask_breakdown.py"
_spec = importlib.util.spec_from_file_location("subtask_breakdown", _SCRIPT)
sb = importlib.util.module_from_spec(_spec)
sys.modules["subtask_breakdown"] = sb
_spec.loader.exec_module(sb)


def test_group_by_subtask_groups_and_ignores_missing_scores():
    records = [
        {"task_id": "a", "benchmark": "bbh", "subtask": "navigate", "score": 1.0},
        {"task_id": "b", "benchmark": "bbh", "subtask": "navigate", "score": 0.0},
        {"task_id": "c", "benchmark": "bbh", "subtask": "word_sorting", "score": 1.0},
        {"task_id": "d", "benchmark": "bbh", "subtask": None, "score": None},  # skipped
        {"task_id": "e", "benchmark": "math500", "subtask": None, "score": 1.0},  # falls back to benchmark
    ]

    groups = sb.group_by_subtask(records)

    assert groups["navigate"] == [1.0, 0.0]
    assert groups["word_sorting"] == [1.0]
    assert groups["math500"] == [1.0]
    assert "None" not in groups


def test_group_by_subtask_rejects_bool_scores():
    # bool is a subclass of int in Python; must not silently masquerade as a score.
    records = [{"task_id": "a", "benchmark": "bbh", "subtask": "navigate", "score": True}]
    groups = sb.group_by_subtask(records)
    assert groups == {}


def test_summarize_computes_mean_and_count():
    groups = {"navigate": [1.0, 0.0, 1.0], "word_sorting": [1.0]}
    summary = sb.summarize(groups)
    assert summary["navigate"]["n"] == 3
    assert summary["navigate"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["word_sorting"] == {"n": 1, "accuracy": 1.0}


def test_render_includes_pooled_overall_row():
    groups = {"navigate": [1.0, 0.0], "word_sorting": [1.0]}
    md = sb.render(groups)
    assert "| navigate | 2 | 0.500 |" in md
    assert "| word_sorting | 1 | 1.000 |" in md
    assert "overall (pooled)" in md
    assert "**3**" in md  # 2 + 1 pooled item count


def test_render_empty_groups_has_no_overall_row():
    md = sb.render({})
    assert "overall (pooled)" not in md


def test_load_records_rejects_non_list_json(tmp_path):
    path = tmp_path / "breakdown.json"
    path.write_text(json.dumps({"not": "a list"}))

    with pytest.raises(ValueError, match="must contain a JSON list"):
        sb.load_records(str(path))


def test_load_records_round_trips_a_real_file(tmp_path):
    records = [{"task_id": "a", "benchmark": "bbh", "subtask": "navigate", "score": 1.0}]
    path = tmp_path / "breakdown.json"
    path.write_text(json.dumps(records))

    loaded = sb.load_records(str(path))

    assert loaded == records
