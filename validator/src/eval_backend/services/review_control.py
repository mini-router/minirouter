from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ReviewControl


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def seed_review_control(session: Session) -> ReviewControl:
    row = session.execute(select(ReviewControl).where(ReviewControl.id == 1)).scalar_one_or_none()
    if row is None:
        row = ReviewControl(id=1, enabled=False)
        session.add(row)
        session.flush()
        return row
    return row


def get_review_control(session: Session) -> ReviewControl:
    row = session.execute(select(ReviewControl).where(ReviewControl.id == 1)).scalar_one_or_none()
    if row is None:
        row = seed_review_control(session)
    return row


def start_review(session: Session, *, started_by: str | None = None) -> ReviewControl:
    row = get_review_control(session)
    row.enabled = True
    row.started_by = started_by.strip() if started_by and started_by.strip() else row.started_by
    row.started_at = row.started_at or _utcnow()
    row.updated_at = _utcnow()
    session.flush()
    return row


def pause_review(session: Session) -> ReviewControl:
    row = get_review_control(session)
    row.enabled = False
    row.updated_at = _utcnow()
    session.flush()
    return row
