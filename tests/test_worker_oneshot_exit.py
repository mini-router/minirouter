"""One-shot worker must exit 0 after a successful pass (#168)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1] / "validator" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_oneshot_exits_zero_even_when_work_was_done(monkeypatch):
    import eval_backend.worker as worker

    monkeypatch.setattr(
        worker,
        "Settings",
        mock.Mock(load=mock.Mock(return_value=mock.Mock(
            ensure_dirs=mock.Mock(),
        ))),
    )
    monkeypatch.setattr(worker, "build_engine", mock.Mock())
    monkeypatch.setattr(worker, "Base", mock.Mock(metadata=mock.Mock(create_all=mock.Mock())))
    monkeypatch.setattr(worker, "ensure_schema", mock.Mock())
    monkeypatch.setattr(worker, "build_session_factory", mock.Mock(return_value=object()))
    monkeypatch.setattr(worker, "process_once", mock.Mock(return_value=1))
    monkeypatch.setattr(sys, "argv", ["worker"])

    with pytest.raises(SystemExit) as ei:
        worker.main()
    assert ei.value.code == 0
