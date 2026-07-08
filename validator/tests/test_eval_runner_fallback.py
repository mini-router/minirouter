from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from eval_backend.core.config import Settings
from eval_backend.db import Base
from eval_backend.models import Submission
from eval_backend.services import eval_runner


def _build_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()


def _build_settings(tmp_path: Path, *, allow_local_fallback: bool) -> Settings:
    return Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        eval_execution_mode="remote_gpu",
        eval_allow_local_fallback=allow_local_fallback,
        eval_max_items=2,
    )


def _add_submission(session, checkpoint_path: Path) -> Submission:
    submission = Submission(
        id="sub-fallback",
        source="upload",
        artifact_name="bundle.zip",
        artifact_path=str(checkpoint_path),
        artifact_sha256="abc123",
        checkpoint_path=str(checkpoint_path),
        benchmark="math500",
        status="queued",
    )
    session.add(submission)
    session.flush()
    return submission


def test_remote_failure_records_local_fallback_metadata(tmp_path, monkeypatch):
    session = _build_session()
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(*args, **kwargs):
        raise RuntimeError("ssh failed")

    def _fake_local_attempt(settings, checkpoint_path, local_results_path, submission_id, env):
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.9}}}),
            encoding="utf-8",
        )
        return ("local-command", 0, "stdout", "")

    monkeypatch.setattr(eval_runner, "_remote_attempt", _fake_remote_attempt)
    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "completed"
    assert result.metrics["execution_mode"] == "local_fallback"
    assert result.metrics["local_fallback"] is True
    assert "remote gpu attempt failed" in result.metrics["remote_error"]


def test_remote_failure_fails_when_local_fallback_disabled(tmp_path, monkeypatch):
    session = _build_session()
    settings = _build_settings(tmp_path, allow_local_fallback=False)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(*args, **kwargs):
        raise RuntimeError("ssh failed")

    monkeypatch.setattr(eval_runner, "_remote_attempt", _fake_remote_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "failed"
    assert submission.status == "failed"
    assert result.metrics["execution_mode"] == "remote_gpu"
    assert result.metrics["local_fallback"] is False
    assert "remote gpu attempt failed" in result.metrics["remote_error"]


def test_remote_success_marks_remote_execution_mode(tmp_path, monkeypatch):
    session = _build_session()
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(settings, checkpoint_path, local_results_path, submission_id, env, on_line=None):
        local_results_path.parent.mkdir(parents=True, exist_ok=True)
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.6}}}),
            encoding="utf-8",
        )
        return ("remote-command", 0, "stdout", "")

    def _fake_local_attempt(*args, **kwargs):
        raise AssertionError("local fallback should not run on remote success")

    monkeypatch.setattr(eval_runner, "_remote_attempt", _fake_remote_attempt)
    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "completed"
    assert result.metrics["execution_mode"] == "remote_gpu"
    assert result.metrics["local_fallback"] is False
    assert "remote_error" not in result.metrics
