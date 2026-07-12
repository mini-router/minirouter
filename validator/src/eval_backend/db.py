from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .core.config import Settings


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

        # The statements below relax / back-fill *legacy* columns that only exist
        # on databases created before those fields became ORM properties. Postgres
        # has no IF EXISTS form for ALTER COLUMN, and a fresh DB built from the
        # current models has none of these columns, so gate each on the column
        # actually being present — otherwise first-run startup crashes with
        # UndefinedColumn (issue #120).
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
