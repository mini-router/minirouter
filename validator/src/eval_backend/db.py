from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .core.config import DEFAULT_GITHUB_REVIEW_SCORE_THRESHOLD, Settings


Base = declarative_base()


def _ensure_postgres(database_url: str) -> None:
    if not make_url(database_url).drivername.startswith("postgresql"):
        raise ValueError(
            "Postgres is the only supported database backend for minirouter validator"
        )


def build_engine(settings: Settings):
    _ensure_postgres(settings.database_url)
    return create_engine(
        settings.database_url,
        future=True,
        pool_pre_ping=True,
    )


def ensure_schema(engine) -> None:
    # Minimal, Postgres-only schema upgrades for the running validator.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS miner_id VARCHAR(255)"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS benchmark_names_json JSON"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS submission_artifact_id VARCHAR(36)"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS latest_train_id INTEGER"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS latest_eval_id INTEGER"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS best_eval_id INTEGER"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ"
        )
        conn.exec_driver_sql(
            "ALTER TABLE submissions ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE PRECISION"
        )
        conn.exec_driver_sql("ALTER TABLE submissions ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION")
        conn.exec_driver_sql(
            "ALTER TABLE competition_runtime_config ADD COLUMN IF NOT EXISTS default_eval_execution_mode VARCHAR(32)"
        )
        conn.exec_driver_sql(
            "ALTER TABLE competition_runtime_config ADD COLUMN IF NOT EXISTS default_eval_batch_size INTEGER"
        )
        conn.exec_driver_sql(
            "ALTER TABLE competition_runtime_config ADD COLUMN IF NOT EXISTS king_score DOUBLE PRECISION"
        )

        # Legacy columns below only exist on DBs created before those fields became
        # ORM @property accessors. Postgres has no IF EXISTS for ALTER COLUMN, so
        # gate each statement on information_schema — otherwise fresh-DB startup
        # crashes with UndefinedColumn (issue #120).
        legacy_columns = {
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'submissions' AND table_schema = current_schema()"
            )
        }
        for column in ("artifact_name", "artifact_path", "artifact_sha256", "checkpoint_path", "benchmark"):
            if column in legacy_columns:
                conn.exec_driver_sql(f"ALTER TABLE submissions ALTER COLUMN {column} DROP NOT NULL")

        if "team_name" in legacy_columns:
            conn.exec_driver_sql(
                "UPDATE submissions SET miner_id = COALESCE(miner_id, team_name) "
                "WHERE miner_id IS NULL AND team_name IS NOT NULL"
            )
        if "benchmark" in legacy_columns:
            conn.exec_driver_sql(
                "UPDATE submissions SET benchmark_names_json = COALESCE(benchmark_names_json, json_build_array(benchmark)) "
                "WHERE benchmark_names_json IS NULL AND benchmark IS NOT NULL"
            )
        conn.exec_driver_sql(
            "UPDATE competition_runtime_config "
            "SET default_eval_execution_mode = COALESCE(NULLIF(default_eval_execution_mode, ''), 'remote_gpu')"
        )
        conn.exec_driver_sql(
            "UPDATE competition_runtime_config "
            "SET default_eval_batch_size = COALESCE(NULLIF(default_eval_batch_size, 0), 1)"
        )
        conn.exec_driver_sql(
            "UPDATE competition_runtime_config "
            f"SET king_score = COALESCE(king_score, {DEFAULT_GITHUB_REVIEW_SCORE_THRESHOLD})"
        )


def build_session_factory(engine):
    # Keep loaded ORM attributes available after commit so short-lived
    # worker/webhook flows can safely inspect objects after the session ends.
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@contextmanager
def session_scope(session_factory) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
