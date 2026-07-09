from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from eval_backend.api.routes import leaderboard
from eval_backend.db import Base
from eval_backend.models import EvaluationRun, Submission


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _build_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()


def _add_submission(
    session,
    *,
    submission_id: str,
    source: str,
    status: str,
    latest_score: float | None,
    team_name: str | None = None,
    created_at: datetime | None = None,
) -> Submission:
    created = created_at or _utc(2026, 7, 1)
    submission = Submission(
        id=submission_id,
        source=source,
        team_name=team_name,
        artifact_name="bundle.tar.gz",
        artifact_path=f"/tmp/{submission_id}.tar.gz",
        artifact_sha256="abc123",
        benchmark="math500",
        status=status,
        latest_score=latest_score,
        created_at=created,
        updated_at=created,
    )
    session.add(submission)
    session.flush()
    return submission


def _request_with_session(session) -> MagicMock:
    request = MagicMock()
    request.app.state.session_factory = lambda: session
    return request


def test_leaderboard_includes_completed_github_pr_submissions() -> None:
    session = _build_session()
    _add_submission(
        session,
        submission_id="pr-sub-1",
        source="github_pr",
        status="completed",
        latest_score=0.82,
        team_name="team-alpha",
    )
    session.commit()

    response = leaderboard(_request_with_session(session))
    assert len(response.items) == 1
    assert response.items[0].submission_id == "pr-sub-1"
    assert response.items[0].team == "team-alpha"
    assert response.items[0].accuracy == 0.82


def test_leaderboard_excludes_incomplete_submissions() -> None:
    session = _build_session()
    _add_submission(
        session,
        submission_id="queued-sub",
        source="github_pr",
        status="queued",
        latest_score=None,
    )
    _add_submission(
        session,
        submission_id="failed-sub",
        source="github_pr",
        status="failed",
        latest_score=None,
    )
    session.commit()

    response = leaderboard(_request_with_session(session))
    assert response.items == []


@pytest.mark.parametrize("source", ["github_pr", "upload", "seed"])
def test_leaderboard_includes_eligible_sources(source: str) -> None:
    session = _build_session()
    _add_submission(
        session,
        submission_id=f"{source}-sub",
        source=source,
        status="completed",
        latest_score=0.5,
        team_name=source,
    )
    session.commit()

    response = leaderboard(_request_with_session(session))
    assert [item.submission_id for item in response.items] == [f"{source}-sub"]


def test_leaderboard_orders_by_score_then_submission_time() -> None:
    session = _build_session()
    _add_submission(
        session,
        submission_id="lower-score",
        source="github_pr",
        status="completed",
        latest_score=0.7,
        team_name="lower",
        created_at=_utc(2026, 7, 1),
    )
    _add_submission(
        session,
        submission_id="higher-score",
        source="github_pr",
        status="completed",
        latest_score=0.9,
        team_name="higher",
        created_at=_utc(2026, 7, 2),
    )
    _add_submission(
        session,
        submission_id="tie-earlier",
        source="upload",
        status="completed",
        latest_score=0.7,
        team_name="earlier",
        created_at=_utc(2026, 6, 30),
    )
    session.commit()

    response = leaderboard(_request_with_session(session))
    assert [item.submission_id for item in response.items] == [
        "higher-score",
        "tie-earlier",
        "lower-score",
    ]
    assert [item.rank for item in response.items] == [1, 2, 3]


def test_leaderboard_loads_metrics_from_best_run() -> None:
    session = _build_session()
    submission = _add_submission(
        session,
        submission_id="metrics-sub",
        source="github_pr",
        status="completed",
        latest_score=0.88,
        team_name="metrics-team",
    )
    run = EvaluationRun(
        submission_id=submission.id,
        status="completed",
        score=0.88,
        metrics_json=json.dumps({"mmlu": 0.81, "math": 0.95, "params": 10240}),
    )
    session.add(run)
    session.flush()
    submission.best_run_id = run.id
    session.commit()

    response = leaderboard(_request_with_session(session))
    entry = response.items[0]
    assert entry.mmlu == 0.81
    assert entry.math == 0.95
    assert entry.params == 10240
