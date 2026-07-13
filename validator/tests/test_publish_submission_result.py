"""Unit tests for the PR merge/close decision in publish_submission_result.

Regression coverage for issue #190: an evaluation that *fails* (the validator
could not run it — e.g. the GPU host was unreachable) must NOT close the PR. Only
an eval that *completes* and scores at/below threshold is a rejection.
"""
from __future__ import annotations

import asyncio

import pytest

from eval_backend.core.config import Settings
from eval_backend.models import EvaluationRun, Submission
from eval_backend.services import github
from eval_backend.services.eval_runner import EvaluationResult


def _settings() -> Settings:
    return Settings(
        github_access_token="test-token",
        github_post_comment_on_eval=True,
        github_review_score_threshold=0.8,
        public_site_url="",
    )


def _submission() -> Submission:
    sub = Submission()
    sub.id = "sub-1"
    sub.source = "github_pr"
    sub.pr_number = 42
    sub.repo_full_name = "mini-router/minirouter"
    sub.head_sha = "deadbeef"
    sub.latest_eval_id = 1
    return sub


def _result(*, status: str, score: float | None, metrics: dict | None = None) -> EvaluationResult:
    run = EvaluationRun()
    run.id = 1
    run.status = status
    run.score = score
    run.error = None
    return EvaluationResult(run=run, score=score, metrics=metrics or {}, stdout="", stderr="")


@pytest.fixture()
def calls(monkeypatch):
    """Record which GitHub side effects publish_submission_result triggers."""
    recorded: dict[str, object] = {"merged": False, "closed": False, "status": None}

    async def _fake_comment(settings, submission, body):
        return None

    async def _fake_status(settings, submission, *, state, description, context="", target_url=None):
        recorded["status"] = state

    async def _fake_merge(settings, submission):
        recorded["merged"] = True

    async def _fake_close(settings, submission):
        recorded["closed"] = True

    monkeypatch.setattr(github, "post_pr_comment", _fake_comment)
    monkeypatch.setattr(github, "set_commit_status", _fake_status)
    monkeypatch.setattr(github, "merge_pull_request", _fake_merge)
    monkeypatch.setattr(github, "close_pull_request", _fake_close)
    # Keep the summary builder from needing a fully-populated ORM graph.
    monkeypatch.setattr(github, "build_submission_summary_markdown", lambda *a, **k: "summary")
    return recorded


def test_infra_failure_does_not_close_pr(calls):
    # SSH/connection failure: the eval never produced a score.
    result = _result(
        status="failed",
        score=None,
        metrics={"remote_connection_error": "remote ssh setup failed: ... exit status 255"},
    )
    asyncio.run(github.publish_submission_result(_settings(), _submission(), result))

    assert calls["closed"] is False  # <-- the fix: infra failure must not close the PR
    assert calls["merged"] is False
    assert calls["status"] == "error"  # retryable, not a hard "failure"


def test_completed_low_score_closes_pr(calls):
    result = _result(status="completed", score=0.5)
    asyncio.run(github.publish_submission_result(_settings(), _submission(), result))

    assert calls["closed"] is True  # genuine rejection: eval ran, scored below threshold
    assert calls["merged"] is False
    assert calls["status"] == "failure"


def test_completed_high_score_merges_pr(calls):
    result = _result(status="completed", score=0.95)
    asyncio.run(github.publish_submission_result(_settings(), _submission(), result))

    assert calls["merged"] is True
    assert calls["closed"] is False
    assert calls["status"] == "success"
