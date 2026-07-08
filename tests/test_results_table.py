"""Unit tests for scripts/results_table.py (offline, no experiments/ required)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results_table.py"
_spec = importlib.util.spec_from_file_location("results_table", _SCRIPT)
rt = importlib.util.module_from_spec(_spec)
sys.modules["results_table"] = rt
_spec.loader.exec_module(rt)


def _oracle_fixture(benchmark: str = "math500") -> dict:
    return {
        "benchmark": benchmark,
        "point_estimates": {
            "best_single": 0.808,
            "routing_oracle": 0.856,
            "routing_headroom": 0.048,
        },
        "trinity": {
            "accuracy": 0.825,
            "router_gap_closed": 0.354,
        },
        "verdict": {
            "label": "ROUTER_BOUND",
            "message": "headroom exists",
        },
    }


def test_load_oracle_reports_skips_invalid_json(tmp_path):
    bad = tmp_path / "oracle_report_bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert rt.load_oracle_reports(str(tmp_path)) == []


def test_load_oracle_reports_skips_missing_fields(tmp_path):
    incomplete = tmp_path / "oracle_report_x.json"
    incomplete.write_text(json.dumps({"benchmark": "x"}), encoding="utf-8")
    assert rt.load_oracle_reports(str(tmp_path)) == []


def test_load_oracle_reports_finds_nested_reports(tmp_path):
    nested = tmp_path / "final" / "oracle_report_math500.json"
    nested.parent.mkdir(parents=True)
    nested.write_text(json.dumps(_oracle_fixture()), encoding="utf-8")
    reports = rt.load_oracle_reports(str(tmp_path))
    assert len(reports) == 1
    assert reports[0]["benchmark"] == "math500"
    assert reports[0]["file"] == str(nested)


def test_render_oracle_section_empty():
    assert rt.render_oracle_section([]) == ""


def test_render_oracle_section_includes_metrics():
    md = rt.render_oracle_section([_oracle_fixture()])
    assert "## Oracle-ceiling diagnostics" in md
    assert "math500" in md
    assert "0.808" in md
    assert "0.856" in md
    assert "0.354" in md
    assert "ROUTER_BOUND" in md


def test_render_oracle_section_missing_gap_closed():
    rep = _oracle_fixture()
    rep.pop("trinity")
    md = rt.render_oracle_section([rep])
    assert "—" in md


def test_render_appends_oracle_section():
    rows = [{
        "benchmark": "math500",
        "coordinator": "pilot",
        "variant": "eval",
        "trinity": 0.82,
        "random": 0.5,
        "best_single": 0.81,
        "best_model": "glm-5p2",
        "singles": {"glm-5p2": 0.81},
    }]
    md = rt.render(rows, [_oracle_fixture()])
    assert "## Per-coordinator held-out evals" in md
    assert "## Oracle-ceiling diagnostics" in md
