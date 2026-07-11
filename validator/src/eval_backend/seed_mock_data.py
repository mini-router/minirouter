from __future__ import annotations

import json

from sqlalchemy import delete

from .core.config import Settings
from .db import Base, build_engine, build_session_factory, ensure_schema
from .mock_leaderboard import MOCK_LEADERBOARD_ENTRIES
from .models import EvaluationRun, Submission


def seed_mock_leaderboard(session) -> int:
    seed_ids = [entry.submission_id for entry in MOCK_LEADERBOARD_ENTRIES]
    session.execute(delete(EvaluationRun).where(EvaluationRun.submission_id.in_(seed_ids)))
    session.execute(delete(Submission).where(Submission.id.in_(seed_ids)))
    session.flush()

    created = 0
    for entry in MOCK_LEADERBOARD_ENTRIES:
        submission = Submission(
            id=entry.submission_id,
            source="seed",
            miner_id=entry.miner_id or entry.team,
            repo_full_name=None,
            pr_number=None,
            head_sha=None,
            benchmark_names_json=["combined"],
            status="completed",
            latest_score=entry.accuracy,
            latest_eval_id=None,
            best_eval_id=None,
            finished_at=entry.submitted,
            duration_seconds=None,
            cost_usd=None,
            created_at=entry.submitted,
            updated_at=entry.submitted,
        )
        session.add(submission)
        session.flush()

        metrics = {
            "accuracy": entry.accuracy,
            "mmlu": entry.mmlu,
            "math": entry.math,
            "params": entry.params,
        }
        run = EvaluationRun(
            submission_id=submission.id,
            status="completed",
            score=entry.accuracy,
            phase="completed",
            message="seeded from README results table",
            progress_current=1,
            progress_total=1,
            benchmark_names_json=["combined"],
            metrics_json=json.dumps(metrics),
            command="seed://readme-results",
            stdout="seeded from README results table\n",
            stderr=None,
            results_path=f"seed://leaderboard/{entry.submission_id}.json",
            error=None,
            started_at=entry.submitted,
            finished_at=entry.submitted,
            created_at=entry.submitted,
        )
        session.add(run)
        session.flush()
        submission.latest_eval_id = run.id
        submission.best_eval_id = run.id
        created += 1

    session.commit()
    return created


def main() -> None:
    settings = Settings.load()
    engine = build_engine(settings)
    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        count = seed_mock_leaderboard(session)
    print(f"seeded {count} mock leaderboard submissions")


if __name__ == "__main__":
    main()
