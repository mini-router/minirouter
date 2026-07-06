from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from .core.config import Settings
from .db import Base, build_engine, build_session_factory
from .models import Submission
from .services.eval_runner import evaluate_submission
from .services.github import publish_submission_result


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
        submission = (
            session.execute(
                select(Submission).where(Submission.status == "queued").order_by(Submission.created_at.asc())
            )
            .scalars()
            .first()
        )
        if submission is None:
            return 0
        result = evaluate_submission(session, submission, settings)
        session.commit()
    except Exception:
        session.rollback()
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

    settings = Settings.load()
    settings.ensure_dirs()
    engine = build_engine(settings)
    Base.metadata.create_all(bind=engine)
    session_factory = build_session_factory(engine)

    if not args.loop:
        raise SystemExit(process_once(session_factory, settings))

    while True:
        processed = process_once(session_factory, settings)
        if processed == 0:
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
