"""Offline unit tests for the shared cost-reporting helpers (``trinity.costing``).

Pure stdlib / filesystem — no network, no GPU, no provider keys. Locks the
ledger-path resolution and the JSONL cost aggregation that ``train``/``eval``
runs rely on for their cost summaries.
"""
from __future__ import annotations

import json

import pytest

from trinity.costing import (
    COST_PRICES,
    default_cost_ledger_path,
    ledger_cost_report,
)


def _write_ledger(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# default_cost_ledger_path
# ---------------------------------------------------------------------------


def test_env_override_wins_over_out_path(monkeypatch, tmp_path):
    forced = tmp_path / "forced.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(forced))
    assert default_cost_ledger_path("some/eval/out.json") == forced


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("TRINITY_COST_LEDGER", "~/ledger.jsonl")
    result = default_cost_ledger_path(None)
    assert "~" not in str(result)
    assert str(result).endswith("ledger.jsonl")


def test_out_path_derives_cost_ledger_suffix(monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    result = default_cost_ledger_path("runs/math500/eval.json")
    assert result.name == "eval.cost_ledger.jsonl"


def test_no_out_path_defaults_to_cwd(monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    result = default_cost_ledger_path(None)
    assert result.name == "cost_ledger.jsonl"


# ---------------------------------------------------------------------------
# ledger_cost_report
# ---------------------------------------------------------------------------


def test_missing_ledger_reports_zero_and_missing(tmp_path):
    missing = tmp_path / "nope.jsonl"
    report = ledger_cost_report(missing)
    assert report["cost_missing"] is True
    assert report["cost_usd"] == 0.0
    assert report["cost_ledger"] == str(missing)
    # No aggregation keys are emitted for a missing ledger.
    assert "cost_calls" not in report


def test_priced_rows_aggregate_totals(tmp_path):
    # glm-5p2 = (1.40, 4.40): 1M prompt + 1M completion tokens -> $5.80.
    assert COST_PRICES["fireworks:glm-5p2"] == (1.40, 4.40)
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"provider": "fireworks", "m": "glm-5p2", "p": 1_000_000, "c": 1_000_000},
            {"provider": "fireworks", "m": "glm-5p2", "p": 500_000, "c": 0},
        ],
    )
    report = ledger_cost_report(ledger)
    assert report["cost_missing"] is False
    assert report["cost_calls"] == 2
    assert report["cost_prompt_tokens"] == 1_500_000
    assert report["cost_completion_tokens"] == 1_000_000
    # 5.80 + (0.5 * 1.40) = 6.50
    assert report["cost_usd"] == pytest.approx(6.50)
    bucket = report["cost_per_model"]["fireworks:glm-5p2"]
    assert bucket["calls"] == 2
    assert bucket["usd"] == pytest.approx(6.50)


def test_unknown_model_priced_at_zero_but_still_counted(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    key = "madeup:not-a-real-model"
    assert key not in COST_PRICES
    _write_ledger(ledger, [{"provider": "madeup", "m": "not-a-real-model", "p": 10, "c": 20}])
    report = ledger_cost_report(ledger)
    assert report["cost_usd"] == 0.0
    assert report["cost_calls"] == 1
    assert report["cost_prompt_tokens"] == 10
    assert report["cost_completion_tokens"] == 20
    assert report["cost_per_model"][key]["usd"] == 0.0


def test_blank_and_malformed_lines_are_skipped(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"provider": "fireworks", "m": "glm-5p2", "p": 1_000_000, "c": 0}),
                "",  # blank line
                "   ",  # whitespace-only line
                "{not valid json",  # malformed -> skipped, not fatal
                json.dumps({"provider": "fireworks", "m": "glm-5p2", "p": 1_000_000, "c": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = ledger_cost_report(ledger)
    # Only the two well-formed rows count.
    assert report["cost_calls"] == 2
    assert report["cost_prompt_tokens"] == 2_000_000
    assert report["cost_usd"] == pytest.approx(2.80)  # 2 * (1.0 * 1.40)


def test_missing_and_null_token_fields_coerce_to_zero(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"provider": "fireworks", "m": "glm-5p2"},  # no p/c at all
            {"provider": "fireworks", "m": "glm-5p2", "p": None, "c": None},  # explicit null
        ],
    )
    report = ledger_cost_report(ledger)
    assert report["cost_calls"] == 2
    assert report["cost_prompt_tokens"] == 0
    assert report["cost_completion_tokens"] == 0
    assert report["cost_usd"] == 0.0


def test_per_model_buckets_are_separate_and_sorted(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"provider": "fireworks", "m": "kimi-k2p6", "p": 1_000_000, "c": 0},
            {"provider": "fireworks", "m": "deepseek-v4-pro", "p": 1_000_000, "c": 0},
            {"provider": "fireworks", "m": "kimi-k2p6", "p": 1_000_000, "c": 0},
        ],
    )
    report = ledger_cost_report(ledger)
    per_model = report["cost_per_model"]
    assert set(per_model) == {"fireworks:kimi-k2p6", "fireworks:deepseek-v4-pro"}
    assert per_model["fireworks:kimi-k2p6"]["calls"] == 2
    assert per_model["fireworks:deepseek-v4-pro"]["calls"] == 1
    # Insertion order in the returned dict is sorted by key.
    assert list(per_model.keys()) == sorted(per_model.keys())
