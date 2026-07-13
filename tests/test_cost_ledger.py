"""Offline tests for the cost-ledger write/read round-trip.

``_ledger_append`` (writer) and ``ledger_cost_report`` (reader) must agree on
the JSON line format. A model id containing JSON metacharacters (a quote or a
backslash) must still produce a valid, parseable line so its usage is counted
and not silently dropped from the cost total.
"""
from __future__ import annotations

from trinity.costing import ledger_cost_report
from trinity.llm.openai_compatible_pool import _ledger_append


def test_ledger_append_writes_valid_json_for_plain_model(tmp_path, monkeypatch):
    ledger = tmp_path / "cost_ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))

    _ledger_append("openrouter", "google/gemma-3-4b-it", 1000, 2000)

    report = ledger_cost_report(ledger)
    assert report["cost_missing"] is False
    assert report["cost_calls"] == 1
    assert report["cost_prompt_tokens"] == 1000
    assert report["cost_completion_tokens"] == 2000


def test_ledger_append_escapes_json_metacharacters(tmp_path, monkeypatch):
    # A quote/backslash in the model id previously produced a malformed JSON
    # line that ledger_cost_report drops, silently undercounting spend.
    ledger = tmp_path / "cost_ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    model = 'weird"model\\name'

    _ledger_append("openrouter", model, 1000, 2000)

    report = ledger_cost_report(ledger)
    assert report["cost_calls"] == 1
    assert report["cost_prompt_tokens"] == 1000
    assert report["cost_completion_tokens"] == 2000
    key = f"openrouter:{model}"
    assert key in report["cost_per_model"]
    assert report["cost_per_model"][key]["calls"] == 1
