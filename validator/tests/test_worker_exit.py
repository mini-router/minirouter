"""Exit-code semantics for the one-shot (non --loop) worker invocation.

A single pass must exit 0 on success regardless of whether the queue had work;
process_once's integer return is a "did work" signal for the --loop sleep
decision, not a process exit status.
"""
from __future__ import annotations

import sys

import pytest

from eval_backend import worker


class _FakeSettings:
    @classmethod
    def load(cls) -> "_FakeSettings":
        return cls()

    def ensure_dirs(self) -> None:
        pass


def _stub_worker_bootstrap(monkeypatch) -> None:
    """Neutralize DB/engine setup so main() can run without Postgres."""
    monkeypatch.setattr(worker, "Settings", _FakeSettings)
    monkeypatch.setattr(worker, "build_engine", lambda settings: object())
    monkeypatch.setattr(worker.Base.metadata, "create_all", lambda **kwargs: None)
    monkeypatch.setattr(worker, "ensure_schema", lambda engine: None)
    monkeypatch.setattr(worker, "build_session_factory", lambda engine: (lambda: None))
    monkeypatch.setattr(sys, "argv", ["worker"])


@pytest.mark.parametrize("processed", [1, 0])
def test_worker_once_exits_zero_on_success(monkeypatch, processed: int) -> None:
    _stub_worker_bootstrap(monkeypatch)
    monkeypatch.setattr(worker, "process_once", lambda session_factory, settings: processed)

    with pytest.raises(SystemExit) as excinfo:
        worker.main()

    # Success (work done or empty queue) must exit 0, never 1.
    assert excinfo.value.code == 0


def test_worker_once_propagates_failure(monkeypatch) -> None:
    _stub_worker_bootstrap(monkeypatch)

    def _boom(session_factory, settings):
        raise RuntimeError("evaluation crashed")

    monkeypatch.setattr(worker, "process_once", _boom)

    # A genuine failure still surfaces (nonzero exit), not swallowed as success.
    with pytest.raises(RuntimeError, match="evaluation crashed"):
        worker.main()
