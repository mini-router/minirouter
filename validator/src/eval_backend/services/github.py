from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import Submission
from .storage import StoredArtifact


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_pr_submission(
    session: Session,
    settings: Settings,
    *,
    repo_full_name: str | None,
    pr_number: int | None,
    head_sha: str | None,
    team_name: str | None = None,
    artifact: StoredArtifact | None = None,
    extra: dict[str, Any] | None = None,
) -> Submission:
    submission = Submission(
        id=str(uuid4()),
        source="github_pr",
        team_name=team_name,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
        artifact_name=(artifact.name if artifact else "github-pr"),
        artifact_path=(str(artifact.path) if artifact else ""),
        artifact_sha256=(artifact.sha256 if artifact else ""),
        checkpoint_path=(str(artifact.checkpoint_path) if artifact and artifact.checkpoint_path else None),
        benchmark=settings.eval_benchmark,
        status="queued",
        created_at=_utcnow(),
        updated_at=_utcnow(),
    )
    session.add(submission)
    session.flush()
    return submission
