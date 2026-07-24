from __future__ import annotations

from datetime import datetime, timedelta, timezone

from eval_backend.api.routes import _submission_to_schema
from eval_backend.models import EvaluationRun, Submission, TrainRun


def _submission() -> Submission:
    now = datetime.now(timezone.utc)
    return Submission(
        id="sub-order",
        source="github_pr",
        miner_id="miner-a",
        benchmark_names_json=["math500"],
        status="completed",
        created_at=now,
        updated_at=now,
    )


def _evaluation(eval_id: int, created_at: datetime) -> EvaluationRun:
    return EvaluationRun(id=eval_id, status="completed", created_at=created_at)


def _train(train_id: int, created_at: datetime) -> TrainRun:
    return TrainRun(id=train_id, status="completed", created_at=created_at)


def test_submission_schema_orders_evaluations_chronologically() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _evaluation(1, base)
    newer = _evaluation(2, base + timedelta(hours=1))

    submission = _submission()
    # Relationship contents can arrive in any order (e.g. DB row order); the
    # serialized detail response must still be chronological, like ``trains``.
    submission.evaluations = [newer, older]
    submission.trains = []

    out = _submission_to_schema(submission)

    assert [run.id for run in out.evaluations] == [1, 2]


def test_submission_schema_orders_trains_chronologically() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _train(1, base)
    newer = _train(2, base + timedelta(hours=1))

    submission = _submission()
    submission.evaluations = []
    submission.trains = [newer, older]

    out = _submission_to_schema(submission)

    assert [run.id for run in out.trains] == [1, 2]
