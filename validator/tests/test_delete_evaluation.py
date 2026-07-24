from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import sessionmaker

from eval_backend.api.routes import admin_delete_evaluation
from eval_backend.models import EvaluationRun, JobQueue, Submission
from eval_backend.services.queue import enqueue_provider_eval_job


def _build_request(session_factory) -> MagicMock:
    request = MagicMock()
    request.app.state.session_factory = session_factory
    return request


def _standalone_eval(status: str = "queued") -> EvaluationRun:
    now = datetime.now(timezone.utc)
    return EvaluationRun(
        submission_id=None,
        benchmark_names_json=["math500"],
        provider="compatible",
        models_config="configs/models.openrouter-chutes.yaml",
        execution_mode="local_cpu",
        device="cpu",
        dtype="float32",
        batch_size=1,
        max_items=1,
        status=status,
        phase=status,
        created_at=now,
    )


def test_delete_standalone_evaluation_cancels_its_queued_job(validator_engine):
    session_factory = sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    with session_factory() as session:
        run = _standalone_eval(status="queued")
        session.add(run)
        session.flush()
        job = enqueue_provider_eval_job(
            session,
            run,
            payload_json={"evaluation_id": run.id, "job_type": "provider_eval"},
        )
        evaluation_id = run.id
        job_id = job.id
        session.commit()

    response = admin_delete_evaluation(
        _build_request(session_factory),
        evaluation_id,
        user=MagicMock(),
    )

    assert response.deleted_at is not None

    with session_factory() as session:
        run = session.get(EvaluationRun, evaluation_id)
        job = session.get(JobQueue, job_id)
        assert run.deleted_at is not None
        # A soft-deleted standalone eval must not stay in the queue: otherwise the
        # worker would still claim the job, run the provider route, and incur cost.
        assert job.status == "cancelled"
        assert job.last_error == "evaluation deleted"


def test_delete_evaluation_leaves_other_evaluations_jobs_untouched(validator_engine):
    session_factory = sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    with session_factory() as session:
        target = _standalone_eval(status="queued")
        other = _standalone_eval(status="queued")
        session.add_all([target, other])
        session.flush()
        target_job = enqueue_provider_eval_job(
            session,
            target,
            payload_json={"evaluation_id": target.id, "job_type": "provider_eval"},
        )
        other_job = enqueue_provider_eval_job(
            session,
            other,
            payload_json={"evaluation_id": other.id, "job_type": "provider_eval"},
        )
        target_id = target.id
        target_job_id = target_job.id
        other_job_id = other_job.id
        session.commit()

    admin_delete_evaluation(
        _build_request(session_factory),
        target_id,
        user=MagicMock(),
    )

    with session_factory() as session:
        assert session.get(JobQueue, target_job_id).status == "cancelled"
        assert session.get(JobQueue, other_job_id).status == "queued"


def test_delete_submission_backed_evaluation_is_rejected(validator_engine):
    session_factory = sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        submission = Submission(
            id="sub-eval-1",
            source="upload",
            miner_id="miner",
            benchmark_names_json=["math500"],
            status="completed",
            created_at=now,
            updated_at=now,
        )
        run = _standalone_eval(status="completed")
        run.submission_id = submission.id
        session.add_all([submission, run])
        session.flush()
        evaluation_id = run.id
        session.commit()

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        admin_delete_evaluation(
            _build_request(session_factory),
            evaluation_id,
            user=MagicMock(),
        )
    assert excinfo.value.status_code == 400
