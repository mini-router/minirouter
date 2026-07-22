"""Offline regression tests for the toy-set fallback policy (issue #65).

The offline toy set (2-3 hand-written Tasks per benchmark) exists only so smoke
tests run with zero network. Before this, ``load_tasks`` silently substituted it
whenever the real HuggingFace load yielded 0 tasks, so a real ``train`` / ``eval``
run could report a headline score computed on 2-3 fake items with no warning.

These tests lock the policy:
  * default is STRICT -> ``RuntimeError`` when the real load returns nothing;
  * the toy set is opt-in, via the ``allow_toy_fallback`` argument or the
    ``TRINITY_ALLOW_TOY_FALLBACK`` env var, and warns loudly on stderr;
  * a successful real load is completely unaffected (no warning, no raise).

Everything here is offline: the real HF loader is monkeypatched to simulate a
failed/empty load, so no network or ``datasets`` install is required.
"""
from __future__ import annotations

import pytest

import benchmarks.livecodebench as LCB
import trinity.orchestration.dataset as D


@pytest.fixture(autouse=True)
def _clear_toy_env(monkeypatch):
    """Never inherit a real TRINITY_ALLOW_TOY_FALLBACK from the dev/CI environment."""
    monkeypatch.delenv(D._TOY_FALLBACK_ENV, raising=False)


def _force_empty_load(monkeypatch, benchmark="math500"):
    """Make the benchmark's real loader return 0 tasks (offline / gated / bad split)."""
    monkeypatch.setitem(D._HF_LOADERS, benchmark, lambda split: [])


# --------------------------------------------------------------------------- #
# Default: strict. A real run must never score on the toy set.
# --------------------------------------------------------------------------- #
def test_empty_real_load_raises_by_default(monkeypatch):
    _force_empty_load(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        D.load_tasks("math500", "test", max_items=120, seed=0)
    msg = str(exc.value)
    assert "math500" in msg and "0 tasks" in msg
    # The error must tell the caller how to opt in, and why we refuse.
    assert "allow_toy_fallback=True" in msg
    assert D._TOY_FALLBACK_ENV in msg


def test_default_raise_does_not_return_toy_tasks(monkeypatch):
    # Regression: the old behaviour returned the 3-item toy set here instead.
    _force_empty_load(monkeypatch)
    with pytest.raises(RuntimeError):
        D.load_tasks("math500", "test", max_items=None, seed=0)


def test_unknown_benchmark_still_raises_value_error():
    # The pre-existing ValueError contract is unchanged (and takes precedence).
    with pytest.raises(ValueError):
        D.load_tasks("not_a_benchmark", "test", max_items=None, seed=0)


# --------------------------------------------------------------------------- #
# Opt-in: explicit argument.
# --------------------------------------------------------------------------- #
def test_explicit_opt_in_returns_toy_tasks(monkeypatch):
    _force_empty_load(monkeypatch)
    tasks = D.load_tasks(
        "math500", "test", max_items=None, seed=0, allow_toy_fallback=True
    )
    assert tasks, "opt-in must yield the offline toy set"
    assert all(t.meta.get("source") == "toy" for t in tasks)


def test_explicit_opt_in_warns_on_stderr(monkeypatch, capsys):
    _force_empty_load(monkeypatch)
    D.load_tasks("math500", "test", max_items=None, seed=0, allow_toy_fallback=True)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "TOY SET" in err
    # The warning must say the number is not a real benchmark result.
    assert "fake data" in err


def test_opt_in_still_respects_max_items(monkeypatch):
    _force_empty_load(monkeypatch)
    tasks = D.load_tasks(
        "math500", "test", max_items=1, seed=0, allow_toy_fallback=True
    )
    assert len(tasks) == 1


# --------------------------------------------------------------------------- #
# Opt-in: environment variable (for callers that cannot pass the flag).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_var_opt_in_truthy(monkeypatch, value):
    _force_empty_load(monkeypatch)
    monkeypatch.setenv(D._TOY_FALLBACK_ENV, value)
    tasks = D.load_tasks("math500", "test", max_items=None, seed=0)
    assert tasks and all(t.meta.get("source") == "toy" for t in tasks)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
def test_env_var_falsy_still_raises(monkeypatch, value):
    _force_empty_load(monkeypatch)
    monkeypatch.setenv(D._TOY_FALLBACK_ENV, value)
    with pytest.raises(RuntimeError):
        D.load_tasks("math500", "test", max_items=None, seed=0)


# --------------------------------------------------------------------------- #
# A successful real load is unaffected: no raise, no warning, real tasks.
# --------------------------------------------------------------------------- #
def test_successful_load_is_unaffected(monkeypatch, capsys):
    sentinel = D.Task(
        task_id="real-0",
        benchmark="math500",
        prompt="1+1?",
        answer="2",
        meta={"source": "real"},
    )
    monkeypatch.setitem(D._HF_LOADERS, "math500", lambda split: [sentinel])

    tasks = D.load_tasks("math500", "test", max_items=None, seed=0)

    assert tasks == [sentinel]
    assert capsys.readouterr().err == ""


def test_successful_load_ignores_opt_in(monkeypatch):
    # Opting in must not *prefer* the toy set — it only permits it as a fallback.
    sentinel = D.Task(
        task_id="real-0",
        benchmark="math500",
        prompt="1+1?",
        answer="2",
        meta={"source": "real"},
    )
    monkeypatch.setitem(D._HF_LOADERS, "math500", lambda split: [sentinel])

    tasks = D.load_tasks(
        "math500", "test", max_items=None, seed=0, allow_toy_fallback=True
    )
    assert tasks == [sentinel]


# --------------------------------------------------------------------------- #
# Remaining callers: the benchmark facade must forward the opt-in, not drop it.
# --------------------------------------------------------------------------- #
def test_livecodebench_facade_forwards_opt_in(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items=None, seed=0, allow_toy_fallback=False):
        seen["allow_toy_fallback"] = allow_toy_fallback
        return ["ok"]

    monkeypatch.setattr(LCB, "_load_tasks", fake_load_tasks)

    LCB.load("test", max_items=2, allow_toy_fallback=True)
    assert seen["allow_toy_fallback"] is True, "facade dropped the opt-in"

    LCB.load_tasks("test", max_items=2, allow_toy_fallback=True)
    assert seen["allow_toy_fallback"] is True


def test_livecodebench_facade_is_strict_by_default(monkeypatch):
    _force_empty_load(monkeypatch, "livecodebench")
    # Reaching the real dataset loader through the facade must raise, not toy-fall-back.
    with pytest.raises(RuntimeError):
        LCB.load("test", max_items=2)
