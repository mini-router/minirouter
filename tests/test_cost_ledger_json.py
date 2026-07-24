"""Offline tests for cost-ledger JSON escaping (#165).

`_ledger_append` hand-built each ledger line with an f-string, so a provider or
model id containing a quote, a backslash, or a control character emitted a line
that is not valid JSON. `costing.ledger_cost_report` skips unparseable lines
silently, so those calls vanished from `cost_usd` / `cost_calls` / the per-model
breakdown — silent under-reporting of API spend, with no error and no warning.

These tests pin that every such id round-trips as valid JSON and is counted.
Pure stdlib: no network / GPU / torch.
"""
from __future__ import annotations

import json

import pytest

from trinity.costing import ledger_cost_report
from trinity.llm.openai_compatible_pool import _ledger_append

# Model ids that broke the hand-built writer, plus benign controls.
_TRICKY_IDS = [
    'weird"model',       # double quote  -> unescaped, truncated the JSON string
    "back\\slash",       # backslash     -> invalid JSON escape
    "ctrl\tchar",        # control char  -> raw tab is invalid inside a JSON string
    "new\nline",         # newline       -> would also split the JSONL record
    "uniécode/model",
    "normal-model",
]


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(path))
    return path


@pytest.mark.parametrize("model", _TRICKY_IDS)
def test_each_record_is_valid_json(ledger, model):
    _ledger_append("openrouter", model, 11, 22)
    lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])  # would raise before the fix
    assert row["m"] == model
    assert row["provider"] == "openrouter"
    assert row["p"] == 11
    assert row["c"] == 22


@pytest.mark.parametrize("provider", ['prov"ider', "prov\\ider", "openrouter"])
def test_provider_is_escaped_too(ledger, provider):
    _ledger_append(provider, "some-model", 1, 2)
    row = json.loads(ledger.read_text().strip())
    assert row["provider"] == provider


def test_no_call_is_dropped_from_the_cost_report(ledger):
    for i, model in enumerate(_TRICKY_IDS):
        _ledger_append("openrouter", model, 100 + i, 200 + i)

    report = ledger_cost_report(ledger)

    assert report["cost_calls"] == len(_TRICKY_IDS)
    assert report["cost_prompt_tokens"] == sum(100 + i for i in range(len(_TRICKY_IDS)))
    assert report["cost_completion_tokens"] == sum(200 + i for i in range(len(_TRICKY_IDS)))


def test_tricky_model_ids_survive_in_the_per_model_breakdown(ledger):
    for model in _TRICKY_IDS:
        _ledger_append("openrouter", model, 1, 1)

    per_model = ledger_cost_report(ledger)["cost_per_model"]
    for model in _TRICKY_IDS:
        assert any(key.endswith(model) for key in per_model), model


def test_newline_in_id_does_not_split_the_record(ledger):
    # A raw newline would otherwise end the JSONL record mid-string and turn one
    # call into two unparseable lines.
    _ledger_append("openrouter", "new\nline", 5, 6)
    lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["m"] == "new\nline"


def test_appends_accumulate(ledger):
    _ledger_append("openrouter", "a", 1, 2)
    _ledger_append("openrouter", "b", 3, 4)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    assert [r["m"] for r in rows] == ["a", "b"]


def test_no_ledger_env_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    _ledger_append("openrouter", "model", 1, 2)  # must not raise
    assert not list(tmp_path.iterdir())
