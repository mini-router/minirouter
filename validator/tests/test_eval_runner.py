from __future__ import annotations

import json
from pathlib import Path

from eval_backend.core.config import Settings
from eval_backend.models import Artifact, Submission
from eval_backend.services import eval_runner


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        eval_execution_mode="local_cpu",
        eval_max_items=2,
    )


def _add_submission(session, checkpoint_path: Path) -> Submission:
    artifact = Artifact(
        id="artifact-sub-1",
        storage_backend="local",
        storage_uri=str(checkpoint_path),
        file_names_json=[checkpoint_path.name],
        sha256="abc123",
        size_bytes=checkpoint_path.stat().st_size,
        mime_type="application/octet-stream",
        submission_id="sub-1",
        meta_json={"checkpoint_path": str(checkpoint_path)},
    )
    submission = Submission(
        id="sub-1",
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


def test_missing_results_marks_evaluation_failed(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_local_attempt(*args, **kwargs):
        return ("fake-eval-command", 0, "stdout", "")

    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "failed"
    assert submission.status == "failed"
    assert result.score is None
    assert result.metrics["results_missing"] is True
    assert "did not produce results.json" in (result.run.error or "")


def test_valid_results_stay_completed(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_local_attempt(settings, checkpoint_path, local_results_path, submission_id, env):
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.75}}}),
            encoding="utf-8",
        )
        return ("fake-eval-command", 0, "stdout", "")

    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "completed"
    assert submission.status == "completed"
    assert result.score == 0.75
