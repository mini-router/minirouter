"""Tests for the toy-set fallback guard in ``load_tasks`` (issue #65).

A real eval/train run must not silently score on the 2-3 item offline toy set when
the real benchmark fails to load. These tests simulate a failed load by monkeypatching
the benchmark's HF loader to return an empty list (no network / datasets needed).
"""
from __future__ import annotations

import pytest

from trinity.orchestration import dataset as D


@pytest.fixture
def _clear_env(monkeypatch):
    monkeypatch.delenv(D._TOY_FALLBACK_ENV, raising=False)


def _force_empty_load(monkeypatch, benchmark="math500"):
    """Make the real loader for ``benchmark`` yield nothing, as if HF were unavailable."""
    monkeypatch.setitem(D._HF_LOADERS, benchmark, lambda split: [])


def test_default_raises_on_empty_load(monkeypatch, _clear_env):
    _force_empty_load(monkeypatch)
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        D.load_tasks("math500", "test", max_items=None)


def test_opt_in_param_returns_toy_and_warns(monkeypatch, _clear_env, capsys):
    _force_empty_load(monkeypatch)
    tasks = D.load_tasks("math500", "test", max_items=None, allow_toy_fallback=True)
    assert tasks
    assert all(t.meta.get("source") == "toy" for t in tasks)
    assert "TOY" in capsys.readouterr().err


def test_opt_in_env_returns_toy(monkeypatch, capsys):
    monkeypatch.setenv(D._TOY_FALLBACK_ENV, "1")
    _force_empty_load(monkeypatch)
    tasks = D.load_tasks("math500", "test", max_items=None)
    assert tasks
    assert "TOY" in capsys.readouterr().err


def test_env_falsey_still_raises(monkeypatch):
    monkeypatch.setenv(D._TOY_FALLBACK_ENV, "0")
    _force_empty_load(monkeypatch)
    with pytest.raises(RuntimeError):
        D.load_tasks("math500", "test", max_items=None)


@pytest.mark.parametrize("env_value", ["FALSE", "no", "off", "NO"])
def test_env_common_falsey_values_still_raise(monkeypatch, env_value):
    monkeypatch.setenv(D._TOY_FALLBACK_ENV, env_value)
    _force_empty_load(monkeypatch)
    with pytest.raises(RuntimeError):
        D.load_tasks("math500", "test", max_items=None)


def test_successful_load_is_unaffected(monkeypatch, _clear_env, capsys):
    real = D._toy_tasks("math500")[:1]
    monkeypatch.setitem(D._HF_LOADERS, "math500", lambda split: list(real))
    tasks = D.load_tasks("math500", "test", max_items=None)
    assert len(tasks) == 1
    assert capsys.readouterr().err == ""


def test_unknown_benchmark_still_valueerror(_clear_env):
    with pytest.raises(ValueError, match="Unknown benchmark"):
        D.load_tasks("not-a-benchmark", "test", max_items=None)
