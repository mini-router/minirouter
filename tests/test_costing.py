"""Unit tests for default_cost_ledger_path (src/trinity/costing.py).

Covers the three resolution branches: TRINITY_COST_LEDGER env override,
an explicit out_path, and the bare cwd fallback. This helper is imported by
both trinity.eval and trinity.train but previously had no direct coverage.
"""
from __future__ import annotations

from pathlib import Path

from trinity.costing import default_cost_ledger_path


def test_default_cost_ledger_path_falls_back_to_cwd(monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    assert default_cost_ledger_path(None) == Path.cwd() / "cost_ledger.jsonl"


def test_default_cost_ledger_path_derives_from_out_path(monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    assert default_cost_ledger_path("out/eval.json") == Path("out/eval.cost_ledger.jsonl")


def test_default_cost_ledger_path_env_override_wins(monkeypatch, tmp_path):
    ledger = tmp_path / "custom_ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    # env override wins even when an out_path is also given.
    assert default_cost_ledger_path("out/eval.json") == ledger
    assert default_cost_ledger_path(None) == ledger
