from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import PIPELINE_TRAIN_EVAL, Settings
from ..models import JobQueue, Submission, TrainRun


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_submission_job(
    session: Session,
    submission: Submission,
    *,
    job_type: str = "evaluation",
    queue_name: str = "default",
    priority: int = 0,
    payload_json: dict[str, Any] | None = None,
) -> JobQueue:
    dedupe_key = f"{job_type}:{submission.id}"
    existing = session.execute(
        select(JobQueue).where(JobQueue.dedupe_key == dedupe_key)
    ).scalar_one_or_none()
    payload = payload_json or {
        "submission_id": submission.id,
        "job_type": job_type,
        "benchmarks": list(submission.benchmark_names_json or []),
        "source": submission.source,
    }
    if existing is not None:
        existing.job_id = submission.id
        existing.submission_id = submission.id
        existing.queue_name = queue_name
        existing.status = "queued"
        existing.priority = priority
        existing.claimed_by = None
        existing.claimed_at = None
        existing.heartbeat_at = None
        existing.attempts = 0
        existing.max_attempts = max(existing.max_attempts, 3)
        existing.next_run_at = _utcnow()
        existing.last_error = None
        existing.payload_json = payload
        existing.updated_at = _utcnow()
        session.flush()
        return existing

    queue = JobQueue(
        id=str(uuid4()),
        job_type=job_type,
        job_id=submission.id,
        submission_id=submission.id,
        queue_name=queue_name,
        status="queued",
        priority=priority,
        dedupe_key=dedupe_key,
        attempts=0,
        max_attempts=3,
        next_run_at=_utcnow(),
        payload_json=payload,
    )
    session.add(queue)
    session.flush()
    return queue


def enqueue_train_job(
    session: Session,
    train: TrainRun,
    *,
    queue_name: str = "default",
    priority: int = 0,
    payload_json: dict[str, Any] | None = None,
) -> JobQueue:
    dedupe_key = f"train:{train.id}"
    existing = session.execute(
        select(JobQueue).where(JobQueue.dedupe_key == dedupe_key)
    ).scalar_one_or_none()
    payload = payload_json or {
        "train_id": train.id,
        "job_type": "train",
        "submission_id": train.submission_id,
        "benchmarks": list(train.benchmark_names_json or []),
    }
    if existing is not None:
        existing.job_id = str(train.id)
        existing.submission_id = train.submission_id
        existing.queue_name = queue_name
        existing.status = "queued"
        existing.priority = priority
        existing.claimed_by = None
        existing.claimed_at = None
        existing.heartbeat_at = None
        existing.attempts = 0
        existing.max_attempts = max(existing.max_attempts, 3)
        existing.next_run_at = _utcnow()
        existing.last_error = None
        existing.payload_json = payload
        existing.updated_at = _utcnow()
        session.flush()
        return existing

    queue = JobQueue(
        id=str(uuid4()),
        job_type="train",
        job_id=str(train.id),
        submission_id=train.submission_id,
        queue_name=queue_name,
        status="queued",
        priority=priority,
        dedupe_key=dedupe_key,
        attempts=0,
        max_attempts=3,
        next_run_at=_utcnow(),
        payload_json=payload,
    )
    session.add(queue)
    session.flush()
    return queue


def enqueue_submission_pipeline_job(
    session: Session,
    submission: Submission,
    settings: Settings,
    *,
    queue_name: str = "default",
    priority: int = 0,
    payload_json: dict[str, Any] | None = None,
) -> JobQueue | None:
    if settings.pipeline_mode == PIPELINE_TRAIN_EVAL:
        if submission.submission_artifact_id is None:
            return None
        train = TrainRun(
            submission_id=submission.id,
            source=submission.source,
            benchmark_names_json=list(submission.benchmark_names_json or [settings.train_benchmark]),
            warmstart_artifact_id=None,
            status="queued",
            phase="queued",
            message="train job queued",
        )
        session.add(train)
        session.flush()
        submission.latest_train_id = train.id
        submission.latest_eval_id = None
        submission.best_eval_id = None
        submission.latest_score = None
        submission.status = "queued"
        submission.updated_at = _utcnow()
        return enqueue_train_job(
            session,
            train,
            queue_name=queue_name,
            priority=priority,
            payload_json=payload_json
            or {
                "train_id": train.id,
                "job_type": "train",
                "submission_id": submission.id,
                "benchmarks": list(train.benchmark_names_json or []),
                "source": submission.source,
                "submission_artifact_id": submission.submission_artifact_id,
            },
        )

    if submission.submission_artifact_id is None:
        return None
    return enqueue_submission_job(
        session,
        submission,
        queue_name=queue_name,
        priority=priority,
        payload_json=payload_json,
    )
