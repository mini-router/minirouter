from __future__ import annotations

import hashlib
import hmac

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from eval_backend.api.routes import router
from eval_backend.models import JobQueue, Submission
from eval_backend.core.config import Settings
from eval_backend.services.github import create_pr_submission
from eval_backend.services.github import should_promote_submission
from eval_backend.services.runtime_config import seed_runtime_config


def _signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_github_submission_starts_awaiting_ci(validator_session) -> None:
    settings = Settings()

    submission = create_pr_submission(
        validator_session,
        settings,
        repo_full_name="mini-router/minirouter",
        pr_number=123,
        head_sha="abc123",
        team_name="tmimmanuel",
    )

    assert submission.status == "awaiting_ci"
    assert submission.repo_full_name == "mini-router/minirouter"
    assert submission.pr_number == 123


def test_github_submission_keeps_terminal_status_on_metadata_update(validator_session) -> None:
    settings = Settings()

    submission = create_pr_submission(
        validator_session,
        settings,
        repo_full_name="mini-router/minirouter",
        pr_number=124,
        head_sha="abc123",
        team_name="tmimmanuel",
    )
    submission.status = "completed"
    validator_session.flush()

    updated = create_pr_submission(
        validator_session,
        settings,
        repo_full_name="mini-router/minirouter",
        pr_number=124,
        head_sha="def456",
        team_name="tmimmanuel-2",
    )

    assert updated.status == "completed"
    assert updated.head_sha == "def456"
    assert updated.miner_id == "tmimmanuel-2"


def test_github_webhook_enqueues_submission_job(validator_engine) -> None:
    settings = Settings(
        github_webhook_secret="super-secret",
        allowed_repo="mini-router/minirouter",
        github_access_token="",
        public_site_url="https://example.com",
    )
    Session = sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    with Session() as session:
        seed_runtime_config(session, settings)
        session.commit()

    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.session_factory = Session

    body = (
        b'{"repository":{"full_name":"mini-router/minirouter"},'
        b'"labels":[{"name":"submission"},{"name":"miner"}],'
        b'"pull_request":{"number":5,"head":{"sha":"abc123"}},'
        b'"sender":{"login":"tmimmanuel"}}'
    )
    headers = {
        "x-hub-signature-256": _signature(settings.github_webhook_secret, body),
        "content-type": "application/json",
    }

    with TestClient(app) as client:
        response = client.post("/webhooks/github", content=body, headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission"]["status"] == "queued"
    assert payload["submission"]["pr_number"] == 5

    with Session() as session:
        submission = session.query(Submission).filter_by(pr_number=5).one()
        job = session.query(JobQueue).filter_by(submission_id=submission.id).one()

    assert submission.status == "queued"
    assert job.status == "queued"
    assert job.job_type == "evaluation"


def test_should_promote_submission_requires_threshold_and_king_score() -> None:
    assert should_promote_submission(0.81, 0.8, 0.8) is True
    assert should_promote_submission(0.8, 0.8, 0.8) is False
    assert should_promote_submission(0.95, 0.8, 0.96) is False
    assert should_promote_submission(None, 0.8, 0.8) is False

def test_github_webhook_ignores_non_submission_pr(validator_engine) -> None:
    settings = Settings(
        github_webhook_secret="super-secret",
        allowed_repo="mini-router/minirouter",
        github_access_token="",
        public_site_url="https://example.com",
    )
    Session = sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    with Session() as session:
        seed_runtime_config(session, settings)
        session.commit()

    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.session_factory = Session

    body = (
        b'{"repository":{"full_name":"mini-router/minirouter"},'
        b'"pull_request":{"number":6,"head":{"sha":"abc123"},"labels":[]},'
        b'"sender":{"login":"tmimmanuel"}}'
    )
    headers = {
        "x-hub-signature-256": _signature(settings.github_webhook_secret, body),
        "content-type": "application/json",
    }

    with TestClient(app) as client:
        response = client.post("/webhooks/github", content=body, headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission"]["status"] == "ignored"

    with Session() as session:
        submission = session.query(Submission).filter_by(pr_number=6).first()
        job_count = session.query(JobQueue).count()

    assert submission is None
    assert job_count == 0
