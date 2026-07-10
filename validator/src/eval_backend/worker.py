from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from .core.config import Settings
from .db import Base, build_engine, build_session_factory, ensure_schema
from .models import JobQueue, Submission, TrainRun
from .services.eval_runner import evaluate_submission
from .services.github import publish_submission_result
from .services.queue import enqueue_submission_job
from .services.train_runner import run_train_job

logger = logging.getLogger("eval_backend.worker")


@contextmanager
def session_scope(session_factory) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def process_once(session_factory, settings: Settings) -> int:
    session = session_factory()
    submission = None
    result = None
    try:
        logger.info("polling for queued jobs")
        job = (
            session.execute(
                select(JobQueue)
                .where(JobQueue.status == "queued")
                .order_by(JobQueue.priority.desc(), JobQueue.created_at.asc())
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .first()
        )
        if job is None:
            logger.info("no queued jobs found")
            return 0
        job.status = "running"
        job.claimed_by = "worker"
        now = datetime.now(timezone.utc)
        job.claimed_at = now
        job.heartbeat_at = now
        session.flush()
        payload = job.payload_json or {}
        if job.job_type == "train":
            train = session.get(TrainRun, int(job.job_id))
            if train is None:
                job.status = "failed"
                job.last_error = f"train {job.job_id} not found"
                session.flush()
                session.commit()
                logger.error("queued job %s references missing train", job.id)
                return 1
            submission = session.get(Submission, train.submission_id) if train.submission_id else None
            logger.info(
                "processing train job id=%s train id=%s submission id=%s benchmark=%s",
                job.id,
                train.id,
                train.submission_id,
                ", ".join(train.benchmark_names_json or []) or "unknown",
            )
            train_result = run_train_job(session, train, settings)
            job.status = "completed" if train_result.train.status == "completed" else "failed"
            job.last_error = train_result.train.error
            job.heartbeat_at = train_result.train.finished_at
            job.updated_at = train_result.train.finished_at or now
            if train_result.train.status == "completed" and train_result.output_artifact is not None:
                checkpoint_path = (
                    (train_result.output_artifact.meta_json or {}).get("checkpoint_path")
                    or train_result.output_artifact.storage_uri
                )
                if submission is not None:
                    enqueue_submission_job(
                        session,
                        submission,
                        job_type="evaluation",
                        payload_json={
                            "submission_id": submission.id,
                            "train_id": train.id,
                            "input_artifact_id": train_result.output_artifact.id,
                            "checkpoint_path": checkpoint_path,
                            "benchmark_names": train.benchmark_names_json,
                            "source": submission.source,
                        },
                    )
            session.commit()
            logger.info(
                "finished train job id=%s train id=%s status=%s",
                job.id,
                train.id,
                train_result.train.status,
            )
            return 1

        submission = session.get(Submission, job.submission_id) if job.submission_id else None
        if submission is None:
            job.status = "failed"
            job.last_error = f"submission {job.submission_id} not found"
            session.flush()
            session.commit()
            logger.error("queued job %s references missing submission", job.id)
            return 1
        submission.status = "running"
        checkpoint_override = None
        if payload.get("checkpoint_path"):
            from pathlib import Path

            checkpoint_override = Path(str(payload["checkpoint_path"]))
        logger.info(
            "processing evaluation job id=%s submission id=%s source=%s benchmark=%s",
            job.id,
            submission.id,
            submission.source,
            submission.benchmark,
        )
        result = evaluate_submission(
            session,
            submission,
            settings,
            checkpoint_path_override=checkpoint_override,
            train_id=int(payload["train_id"]) if payload.get("train_id") is not None else None,
            input_artifact_id=str(payload["input_artifact_id"]) if payload.get("input_artifact_id") else None,
        )
        job.status = "completed" if result.run.status == "completed" else "failed"
        job.last_error = result.run.error
        job.heartbeat_at = result.run.finished_at
        job.updated_at = result.run.finished_at or now
        session.commit()
        logger.info(
            "finished evaluation job id=%s submission id=%s status=%s score=%s",
            job.id,
            submission.id,
            result.run.status,
            result.score,
        )
    except Exception:
        session.rollback()
        logger.exception("worker failed while processing queued submission")
        raise
    finally:
        session.close()

    if submission is not None and result is not None and submission.source == "github_pr":
        try:
            import asyncio

            asyncio.run(publish_submission_result(settings, submission, result))
        except Exception:
            pass
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll and evaluate queued minirouter submissions")
    parser.add_argument("--loop", action="store_true", help="keep polling until interrupted")
    parser.add_argument("--interval", type=int, default=15, help="poll interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    settings = Settings.load()
    settings.ensure_dirs()
    engine = build_engine(settings)
    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    session_factory = build_session_factory(engine)

    if not args.loop:
        raise SystemExit(process_once(session_factory, settings))

    while True:
        processed = process_once(session_factory, settings)
        if processed == 0:
            logger.info("sleeping for %ss", max(1, args.interval))
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
