from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from eval_backend.core.config import Settings
from eval_backend.models import JobQueue, Submission
from eval_backend.services.queue import enqueue_submission_job
from eval_backend import worker as worker_module


def _submission(submission_id: str) -> Submission:
    return Submission(
        id=submission_id,
        source="upload",
        miner_id="miner-a",
        benchmark_names_json=["math500"],
        status="queued",
    )


def _build_settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
    )


def _session_factory(validator_engine):
    return sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def test_process_once_marks_job_and_submission_failed_on_eval_exception(
    validator_engine, validator_session, tmp_path, monkeypatch
):
    session = validator_session
    submission = _submission("stuck-sub")
    session.add(submission)
    session.flush()
    job = enqueue_submission_job(session, submission, job_type="evaluation")
    session.flush()
    session.commit()

    session_factory = _session_factory(validator_engine)

    def _boom(*args, **kwargs):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(worker_module, "evaluate_submission", _boom)

    processed = worker_module.process_once(session_factory, _build_settings(tmp_path))

    assert processed == 1
    with session_factory() as verify_session:
        failed_submission = verify_session.get(Submission, "stuck-sub")
        failed_job = verify_session.get(JobQueue, job.id)
        assert failed_submission is not None
        assert failed_submission.status == "failed"
        assert failed_job is not None
        assert failed_job.status == "failed"
        assert "eval exploded" in (failed_job.last_error or "")


def test_process_once_does_not_retry_failed_job(
    validator_engine, validator_session, tmp_path, monkeypatch
):
    session = validator_session
    submission = _submission("bad-sub")
    session.add(submission)
    session.flush()
    enqueue_submission_job(session, submission, job_type="evaluation")
    session.flush()
    session.commit()

    session_factory = _session_factory(validator_engine)
    calls = {"count": 0}

    def _fail_once(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("transient eval crash")

    monkeypatch.setattr(worker_module, "evaluate_submission", _fail_once)

    settings = _build_settings(tmp_path)
    worker_module.process_once(session_factory, settings)
    worker_module.process_once(session_factory, settings)

    assert calls["count"] == 1
    with session_factory() as verify_session:
        failed_submission = verify_session.get(Submission, "bad-sub")
        assert failed_submission is not None
        assert failed_submission.status == "failed"
