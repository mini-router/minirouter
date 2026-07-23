"""Offline unit tests for OpenAI-compatible chat-response parsing (issue #31).

`_parse_completion` must be fail-safe against valid-but-unexpected HTTP 200 payloads
(empty `choices`, `{"error": ...}` envelopes, missing `message`, null `content`)
instead of raising `IndexError`/`KeyError` out of `chat()` and aborting an eval run.
No live API calls / no network.
"""
from __future__ import annotations

from trinity.costing import ledger_cost_report
from trinity.llm.openai_compatible_pool import ChatResult, _ledger_append, _parse_completion


def test_normal_completion_is_parsed():
    data = {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    res = _parse_completion(data, "m")
    assert isinstance(res, ChatResult)
    assert res.text == "hello"
    assert res.prompt_tokens == 7
    assert res.completion_tokens == 3
    assert res.finish_reason == "stop"


def test_empty_choices_is_fail_safe():
    data = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}
    res = _parse_completion(data, "m")
    assert res.text == ""
    assert res.finish_reason == "error"
    assert res.prompt_tokens == 5  # usage still accounted


def test_error_envelope_is_fail_safe():
    data = {"error": {"message": "content blocked", "type": "invalid_request"}}
    res = _parse_completion(data, "m")
    assert res.text == ""
    assert res.finish_reason == "error"
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0


def test_null_content_becomes_empty_string():
    data = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    res = _parse_completion(data, "m")
    assert res.text == ""
    assert res.finish_reason == "stop"


def test_missing_message_is_fail_safe():
    data = {"choices": [{"finish_reason": "length"}]}
    res = _parse_completion(data, "m")
    assert res.text == ""
    assert res.finish_reason == "length"


def test_null_choice_entry_is_fail_safe():
    data = {"choices": [None]}
    res = _parse_completion(data, "m")
    assert res.text == ""


def test_non_string_content_is_stringified():
    data = {"choices": [{"message": {"content": 42}, "finish_reason": "stop"}]}
    res = _parse_completion(data, "m")
    assert res.text == "42"


def test_cost_ledger_escapes_model_ids(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))

    _ledger_append("openrouter", 'weird"model\\id', 1000, 2000)

    report = ledger_cost_report(ledger)
    assert report["cost_calls"] == 1
    assert report["cost_prompt_tokens"] == 1000
    assert report["cost_completion_tokens"] == 2000
    assert "openrouter:weird\"model\\id" in report["cost_per_model"]
