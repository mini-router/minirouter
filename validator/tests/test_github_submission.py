from __future__ import annotations

from eval_backend.core.config import Settings
from eval_backend.services.github import create_pr_submission


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
