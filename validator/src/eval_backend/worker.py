from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from .core.config import Settings
from .db import Base, build_engine, build_session_factory, ensure_schema
from .models import Submission
from .services.eval_runner import evaluate_submission
from .services.github import publish_submission_result

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
    try:
        logger.info("polling for queued submissions")
        submission = (
            session.execute(
                select(Submission)
                .where(Submission.status == "queued")
                .order_by(Submission.created_at.asc())
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .first()
        )
        if submission is None:
            logger.info("no queued submissions found")
            return 0
        submission.status = "running"
        session.flush()
        logger.info(
            "processing submission id=%s source=%s benchmark=%s",
            submission.id,
            submission.source,
            submission.benchmark,
        )
        result = evaluate_submission(session, submission, settings)
        session.commit()
        logger.info(
            "finished submission id=%s status=%s score=%s",
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

    if submission.source == "github_pr":
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
