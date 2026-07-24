from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from eval_backend import worker
from eval_backend.core.config import Settings
from eval_backend.models import Artifact, EvaluationRun, JobQueue, Submission
from eval_backend.services import eval_runner
from eval_backend.services.eval_runner import EvaluationResult


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        eval_execution_mode="local_cpu",
        eval_max_items=2,
    )


def _add_submission(session, checkpoint_path: Path, submission_id: str) -> Submission:
    artifact = Artifact(
        id=f"artifact-{submission_id}",
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
        id=submission_id,
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


def test_touch_progress_is_committed_not_just_flushed(validator_session, tmp_path, monkeypatch):
    """Progress only helps if another process can read it mid-run."""
    session = validator_session
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path, "sub-progress")
    run = EvaluationRun(
        submission_id=submission.id,
        benchmark_names_json=["math500"],
        status="running",
    )
    session.add(run)
    session.flush()

    committed: list[bool] = []
    monkeypatch.setattr(session, "commit", lambda: committed.append(True))

    eval_runner._touch_progress(
        session,
        run,
        submission,
        phase="evaluation_running",
        message="item 3/20 running",
        current=3,
        total=20,
    )

    assert committed, "progress update was never committed, so the API cannot observe it"
    assert run.phase == "evaluation_running"
    assert run.progress_current == 3
    assert run.progress_total == 20


def test_worker_commits_the_claim_before_the_job_body_runs(validator_session, tmp_path, monkeypatch):
    """The claim must be durable before the (up to 2h) evaluation starts."""
    session = validator_session
    settings = _build_settings(tmp_path)
    checkpoint_path = tmp_path / "theta.npy"
    checkpoint_path.write_bytes(b"theta")
    submission = _add_submission(session, checkpoint_path, "sub-claim")
    session.add(
        JobQueue(
            id="job-claim",
            job_type="evaluation",
            job_id=submission.id,
            submission_id=submission.id,
            queue_name="default",
            status="queued",
            priority=0,
            dedupe_key="evaluation:sub-claim",
            attempts=0,
            max_attempts=3,
            payload_json={"submission_id": submission.id},
        )
    )
    session.flush()

    commits: list[bool] = []
    real_commit = session.commit

    def _tracking_commit() -> None:
        commits.append(True)
        real_commit()

    monkeypatch.setattr(session, "commit", _tracking_commit)
    # process_once() closes the session it is handed; the fixture owns this one.
    monkeypatch.setattr(session, "close", lambda: None)

    observed: dict[str, object] = {}

    def _fake_evaluate_submission(session_, submission_, settings_, **kwargs):
        observed["commits_before_eval"] = len(commits)
        observed["job_status"] = session_.get(JobQueue, "job-claim").status
        now = datetime.now(timezone.utc)
        run = EvaluationRun(
            submission_id=submission_.id,
            benchmark_names_json=["math500"],
            status="completed",
            score=0.5,
            started_at=now,
            finished_at=now,
        )
        session_.add(run)
        session_.flush()
        return EvaluationResult(run=run, score=0.5, metrics={}, stdout="", stderr="")

    monkeypatch.setattr(worker, "evaluate_submission", _fake_evaluate_submission)

    processed = worker.process_once(lambda: session, settings)

    assert processed == 1
    assert observed["job_status"] == "running"
    assert observed["commits_before_eval"] >= 1, (
        "the worker held the claim in an uncommitted transaction for the whole run, "
        "so /api/jobs still reported the job as queued"
    )
