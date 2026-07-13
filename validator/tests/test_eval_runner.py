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
        meta_json={"checkpoint_path": str(checkpoint_path)},
    )
    session.add(artifact)
    session.flush()
    submission = Submission(
        id="sub-1",
        source="upload",
        miner_id="miner-a",
        benchmark_names_json=["math500"],
        status="queued",
        submission_artifact_id=artifact.id,
    )
    session.add(submission)
    session.flush()
    artifact.submission_id = submission.id
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

    def _fake_local_attempt(
        settings,
        checkpoint_path,
        local_results_path,
        local_ledger_path,
        submission_id,
        env,
    ):
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.75}}}),
            encoding="utf-8",
        )
        local_ledger_path.write_text(
            json.dumps({"provider": "chutes", "m": "google/gemma-4-31B-turbo-TEE", "p": 100, "c": 50})
            + "\n",
            encoding="utf-8",
        )
        return ("fake-eval-command", 0, "stdout", "")

    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "completed"
    assert submission.status == "completed"
    assert result.score == 0.75


def test_nonzero_exit_with_results_still_completes(validator_session, tmp_path, monkeypatch):
    session = validator_session
    settings = _build_settings(tmp_path)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path)

    def _fake_local_attempt(
        settings,
        checkpoint_path,
        local_results_path,
        local_ledger_path,
        submission_id,
        env,
    ):
        local_results_path.write_text(
            json.dumps({"results": {"TRINITY": {"accuracy": 0.88}}}),
            encoding="utf-8",
        )
        local_ledger_path.write_text(
            json.dumps({"provider": "openrouter", "m": "google/gemma-3-4b-it", "p": 100, "c": 50})
            + "\n",
            encoding="utf-8",
        )
        return ("fake-eval-command", 1, "stdout", "stderr")

    monkeypatch.setattr(eval_runner, "_local_attempt", _fake_local_attempt)

    result = eval_runner.evaluate_submission(session, submission, settings)

    assert result.run.status == "completed"
    assert submission.status == "completed"
    assert result.score == 0.88


def test_ledger_cost_report_prices_current_openrouter_models(tmp_path):
    ledger = tmp_path / "cost_ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"provider": "openrouter", "m": "qwen/qwen3-coder-30b-a3b-instruct", "p": 1000, "c": 500}),
                json.dumps({"provider": "openrouter", "m": "openai/gpt-oss-120b", "p": 1000, "c": 500}),
                json.dumps({"provider": "openrouter", "m": "google/gemma-3-4b-it", "p": 1000, "c": 500}),
                json.dumps({"provider": "openrouter", "m": "nvidia/nemotron-3-ultra-550b-a55b", "p": 1000, "c": 500}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = eval_runner._ledger_cost_report(ledger)

    assert metrics["cost_missing"] is False
    assert metrics["cost_calls"] == 4
    assert metrics["cost_usd"] > 0
    assert metrics["cost_per_model"]["openrouter:google/gemma-3-4b-it"]["usd"] > 0
