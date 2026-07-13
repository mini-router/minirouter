from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from eval_backend.api.routes import router
from eval_backend.core.config import Settings
from eval_backend.models import EvaluationRun, JobQueue, Submission
from eval_backend.services import github as github_module
from eval_backend.services.eval_runner import EvaluationResult


def _signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _session_factory(validator_engine):
    return sessionmaker(
        bind=validator_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def _build_settings(tmp_path) -> Settings:
    return Settings(
        github_webhook_secret="webhook-secret",
        allowed_repo="mini-router/minirouter",
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        local_repo_dir=tmp_path,
        public_site_url="https://example.test",
    )


def _build_app(validator_engine, settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    app.state.session_factory = _session_factory(validator_engine)
    app.include_router(router)
    return app


def test_github_webhook_without_artifact_stays_awaiting_ci(
    validator_engine, tmp_path, monkeypatch
):
    settings = _build_settings(tmp_path)
    app = _build_app(validator_engine, settings)
    monkeypatch.setattr("eval_backend.api.routes.set_commit_status", AsyncMock())

    payload = {
        "action": "opened",
        "repository": {"full_name": "mini-router/minirouter"},
        "pull_request": {"number": 501, "head": {"sha": "abc123"}},
        "sender": {"login": "RealDiligent"},
    }
    body = json.dumps(payload).encode("utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": _signature(settings.github_webhook_secret, body),
            },
        )

    assert response.status_code == 200, response.text
    session_factory = _session_factory(validator_engine)
    with session_factory() as session:
        submission = session.execute(
            select(Submission).where(
                Submission.source == "github_pr",
                Submission.pr_number == 501,
            )
        ).scalar_one()
        jobs = session.execute(
            select(JobQueue).where(JobQueue.submission_id == submission.id)
        ).scalars().all()
        assert submission.status == "awaiting_ci"
        assert jobs == []


def test_publish_submission_result_skips_pr_close_without_artifact(monkeypatch):
    close_mock = AsyncMock()
    merge_mock = AsyncMock()
    monkeypatch.setattr(github_module, "close_pull_request", close_mock)
    monkeypatch.setattr(github_module, "merge_pull_request", merge_mock)
    monkeypatch.setattr(github_module, "set_commit_status", AsyncMock())
    monkeypatch.setattr(github_module, "post_pr_comment", AsyncMock())

    submission = Submission(
        id="infra-pr",
        source="github_pr",
        repo_full_name="mini-router/minirouter",
        pr_number=137,
        head_sha="deadbeef",
        benchmark_names_json=["math500"],
        status="failed",
    )
    run = EvaluationRun(
        id=1,
        submission_id=submission.id,
        status="failed",
        error="does not have a checkpoint to evaluate",
    )
    settings = Settings(github_access_token="token")

    asyncio.run(
        github_module.publish_submission_result(
            settings,
            submission,
            EvaluationResult(
                run=run,
                score=None,
                metrics={"missing_checkpoint": True},
                stdout="",
                stderr="",
            ),
        )
    )

    close_mock.assert_not_called()
    merge_mock.assert_not_called()
