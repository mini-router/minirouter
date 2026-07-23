from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
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


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    ).first()
    return row is not None


def _column_nullable(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    ).first()
    return bool(row) and row[0] == "YES"


def _add_column_if_missing(conn, table: str, column: str, ddl_type: str) -> None:
    # Only take the exclusive ALTER TABLE lock when the column is actually
    # missing. In steady state (the common case, after the first migration)
    # this is a plain catalog SELECT that only needs a non-blocking
    # AccessShareLock, so starting a new process no longer has to fight a
    # busy job transaction for an exclusive lock it doesn't really need.
    if not _column_exists(conn, table, column):
        conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl_type}")


def _drop_not_null_if_needed(conn, table: str, column: str) -> None:
    if not _column_nullable(conn, table, column):
        conn.exec_driver_sql(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL")


def ensure_schema(engine) -> None:
    # Minimal, Postgres-only schema upgrades for the running validator.
    # Every process (API and worker) runs this at startup. ALTER TABLE takes
    # an exclusive lock and Postgres queues later requests (even plain
    # SELECTs) behind a pending exclusive lock request, so a long-running job
    # transaction here can otherwise block not just this migration but every
    # already-running process's unrelated queries. The lock_timeout bounds
    # the wait for the rare case a migration is genuinely needed while
    # something's busy; the _if_missing/_if_needed checks avoid requesting
    # the exclusive lock at all once the schema is already up to date.
    with engine.begin() as conn:
        conn.exec_driver_sql("SET LOCAL lock_timeout = '5s'")
        _add_column_if_missing(conn, "submissions", "miner_id", "VARCHAR(255)")
        _add_column_if_missing(conn, "submissions", "benchmark_names_json", "JSON")
        _add_column_if_missing(conn, "submissions", "submission_artifact_id", "VARCHAR(36)")
        _add_column_if_missing(conn, "submissions", "latest_train_id", "INTEGER")
        _add_column_if_missing(conn, "submissions", "latest_eval_id", "INTEGER")
        _add_column_if_missing(conn, "submissions", "best_eval_id", "INTEGER")
        _add_column_if_missing(conn, "submissions", "finished_at", "TIMESTAMPTZ")
        _add_column_if_missing(conn, "submissions", "duration_seconds", "DOUBLE PRECISION")
        _add_column_if_missing(conn, "submissions", "cost_usd", "DOUBLE PRECISION")
        _add_column_if_missing(conn, "submissions", "deleted_at", "TIMESTAMPTZ")
        _add_column_if_missing(conn, "evaluations", "deleted_at", "TIMESTAMPTZ")
        _add_column_if_missing(
            conn, "competition_runtime_config", "default_eval_execution_mode", "VARCHAR(32)"
        )
        _add_column_if_missing(
            conn, "competition_runtime_config", "default_eval_batch_size", "INTEGER"
        )
        _add_column_if_missing(conn, "competition_runtime_config", "king_score", "DOUBLE PRECISION")
        _drop_not_null_if_needed(conn, "submissions", "artifact_name")
        _drop_not_null_if_needed(conn, "submissions", "artifact_path")
        _drop_not_null_if_needed(conn, "submissions", "artifact_sha256")
        _drop_not_null_if_needed(conn, "submissions", "checkpoint_path")
        _drop_not_null_if_needed(conn, "submissions", "benchmark")
        conn.exec_driver_sql(
            "UPDATE submissions SET miner_id = COALESCE(miner_id, team_name) "
            "WHERE miner_id IS NULL AND team_name IS NOT NULL"
        )
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
