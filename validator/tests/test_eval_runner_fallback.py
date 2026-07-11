from __future__ import annotations

import json
from pathlib import Path

from eval_backend.core.config import Settings
from eval_backend.models import Artifact, Submission
from eval_backend.services import eval_runner


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
    artifact = Artifact(
        id="artifact-fallback",
        storage_backend="local",
        storage_uri=str(checkpoint_path),
        file_names_json=[checkpoint_path.name],
        sha256="abc123",
        size_bytes=checkpoint_path.stat().st_size,
        mime_type="application/octet-stream",
        submission_id="sub-fallback",
        meta_json={"checkpoint_path": str(checkpoint_path)},
    )
    submission = Submission(
        id="sub-fallback",
        source="upload",
        miner_id="miner-a",
        benchmark_names_json=["math500"],
        status="queued",
    )
    session.add(artifact)
    session.add(submission)
    submission.submission_artifact_id = artifact.id
    session.flush()
    return submission


def test_remote_failure_records_local_fallback_metadata(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(*args, **kwargs):
        raise RuntimeError("ssh failed")

    def _fake_local_attempt(
        settings,
        checkpoint_path,
        local_results_path,
        local_ledger_path,
        submission_id,
        env,
    ):
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.9}}}),
            encoding="utf-8",
        )
        local_ledger_path.write_text(
            json.dumps({"provider": "chutes", "m": "google/gemma-4-31B-turbo-TEE", "p": 100, "c": 50})
            + "\n",
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


def test_remote_failure_fails_when_local_fallback_disabled(validator_session, tmp_path, monkeypatch):
    session = validator_session
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


def test_remote_connection_failure_aborts_without_local_fallback(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(*args, **kwargs):
        raise eval_runner.RemoteConnectionError("ssh refused")

    def _fake_local_attempt(*args, **kwargs):
        raise AssertionError("local fallback must not run after remote connection failure")

    monkeypatch.setattr(eval_runner, "_remote_attempt", _fake_remote_attempt)
    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "failed"
    assert submission.status == "failed"
    assert result.metrics["execution_mode"] == "remote_gpu"
    assert result.metrics["local_fallback"] is False
    assert "remote gpu connection failed" in result.metrics["remote_connection_error"]
    assert result.run.error and "remote gpu connection failed" in result.run.error


def test_remote_success_marks_remote_execution_mode(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_remote_attempt(
        settings,
        checkpoint_path,
        local_results_path,
        local_ledger_path,
        submission_id,
        env,
        on_line=None,
    ):
        local_results_path.parent.mkdir(parents=True, exist_ok=True)
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.6}}}),
            encoding="utf-8",
        )
        local_ledger_path.write_text(
            "\n".join(
                [
                    json.dumps({"provider": "chutes", "m": "google/gemma-4-31B-turbo-TEE", "p": 1000, "c": 500}),
                    json.dumps({"provider": "chutes", "m": "Qwen/Qwen3-32B-TEE", "p": 2000, "c": 750}),
                ]
            )
            + "\n",
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
    assert result.metrics["duration_seconds"] >= 0
    assert result.metrics["cost_usd"] > 0
    assert result.metrics["cost_calls"] == 2


def test_remote_command_includes_batch_size(tmp_path):
    settings = _build_settings(tmp_path, allow_local_fallback=True)
    settings.eval_batch_size = 4
    command = eval_runner._build_remote_command(
        settings,
        tmp_path / "theta.npy",
        tmp_path / "results.json",
        tmp_path / "ledger.jsonl",
        tmp_path / "workspace",
    )
    assert "--batch-size 4" in command
