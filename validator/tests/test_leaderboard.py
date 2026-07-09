from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from eval_backend.api.routes import leaderboard
from eval_backend.db import Base
from eval_backend.models import Submission


def _submission(
    submission_id: str,
    *,
    source: str,
    status: str,
    latest_score: float | None,
    team_name: str,
) -> Submission:
    now = datetime.now(timezone.utc)
    return Submission(
        id=submission_id,
        source=source,
        team_name=team_name,
        artifact_name="bundle.tar.gz",
        artifact_path=f"/tmp/{submission_id}.tar.gz",
        artifact_sha256="0" * 64,
        benchmark="math500",
        status=status,
        latest_score=latest_score,
        created_at=now,
        updated_at=now,
    )


def _build_request(session_factory):
    request = MagicMock()
    request.app.state.session_factory = session_factory
    return request


def test_leaderboard_includes_completed_github_pr_and_upload_submissions():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with session_factory() as session:
        session.add_all(
            [
                _submission(
                    "seed-1",
                    source="seed",
                    status="completed",
                    latest_score=0.5,
                    team_name="seed-team",
                ),
                _submission(
                    "pr-1",
                    source="github_pr",
                    status="completed",
                    latest_score=0.9,
                    team_name="miner-a",
                ),
                _submission(
                    "upload-1",
                    source="upload",
                    status="completed",
                    latest_score=0.7,
                    team_name="miner-b",
                ),
                _submission(
                    "queued-1",
                    source="upload",
                    status="queued",
                    latest_score=None,
                    team_name="pending",
                ),
            ]
        )
        session.commit()

    response = leaderboard(_build_request(session_factory), limit=100)

    assert [entry.submission_id for entry in response.items] == ["pr-1", "upload-1", "seed-1"]
    assert [entry.team for entry in response.items] == ["miner-a", "miner-b", "seed-team"]


def test_leaderboard_excludes_incomplete_submissions():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with session_factory() as session:
        session.add_all(
            [
                _submission(
                    "failed-1",
                    source="github_pr",
                    status="failed",
                    latest_score=None,
                    team_name="failed-team",
                ),
                _submission(
                    "running-1",
                    source="github_pr",
                    status="running",
                    latest_score=0.8,
                    team_name="running-team",
                ),
            ]
        )
        session.commit()

    response = leaderboard(_build_request(session_factory), limit=100)

    assert response.items == []
