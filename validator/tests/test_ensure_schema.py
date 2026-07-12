from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from eval_backend.db import Base, ensure_schema
import eval_backend.models  # noqa: F401  (registers ORM tables on Base.metadata)

DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://minirouter:minirouter@127.0.0.1:5432/minirouter_test"
)
_SCHEMA = "ensure_schema_test"

_LEGACY_COLUMNS = ("artifact_name", "artifact_path", "artifact_sha256", "checkpoint_path", "benchmark")


def _database_url() -> str:
    return (
        os.environ.get("VALIDATOR_TEST_DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
        or DEFAULT_TEST_DATABASE_URL
    )


@pytest.fixture()
def isolated_engine():
    """A Postgres engine pinned to a throwaway schema, dropped after each test.

    ``ensure_schema`` targets unqualified table names, so we point ``search_path``
    at a dedicated schema; every connection from this engine (including the one
    ``ensure_schema`` opens internally) resolves there. That keeps these DDL-heavy
    migration tests fully isolated from the rest of the suite's ``public`` tables.

    Skips when no Postgres is reachable, matching the rest of the validator DB
    tests (``conftest.validator_engine``) so the suite stays runnable wherever a
    database isn't provisioned.
    """
    url = _database_url()
    if not make_url(url).drivername.startswith("postgresql"):
        pytest.skip(f"ensure_schema tests require a Postgres database URL, got {url!r}")

    engine = create_engine(
        url,
        future=True,
        connect_args={"options": f"-csearch_path={_SCHEMA}"},
    )
    try:
        engine.connect().close()
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"Postgres test database unavailable at {url!r}: {exc}")

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        conn.exec_driver_sql(f"CREATE SCHEMA {_SCHEMA}")
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        engine.dispose()


def _columns(engine) -> dict[str, str]:
    with engine.connect() as conn:
        return {
            row[0]: row[1]
            for row in conn.exec_driver_sql(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'submissions' AND table_schema = current_schema()"
            )
        }


def test_ensure_schema_succeeds_on_fresh_database(isolated_engine):
    # A fresh DB built from the current models has none of the legacy columns.
    # ensure_schema must not crash on it (regression for issue #120) and must be
    # idempotent.
    Base.metadata.create_all(isolated_engine)
    ensure_schema(isolated_engine)
    ensure_schema(isolated_engine)

    columns = _columns(isolated_engine)
    assert {"miner_id", "benchmark_names_json", "submission_artifact_id"} <= set(columns)


def test_ensure_schema_relaxes_and_backfills_legacy_database(isolated_engine):
    # Simulate a pre-migration database that still carries the old NOT NULL columns.
    with isolated_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE submissions ("
            " id VARCHAR PRIMARY KEY,"
            " team_name VARCHAR,"
            " artifact_name VARCHAR NOT NULL,"
            " artifact_path VARCHAR NOT NULL,"
            " artifact_sha256 VARCHAR NOT NULL,"
            " checkpoint_path VARCHAR NOT NULL,"
            " benchmark VARCHAR NOT NULL"
            ")"
        )
        conn.exec_driver_sql(
            "INSERT INTO submissions"
            " (id, team_name, artifact_name, artifact_path, artifact_sha256, checkpoint_path, benchmark)"
            " VALUES ('s1', 'team-a', 'bundle.zip', '/p', 'abc', '/c', 'math500')"
        )

    ensure_schema(isolated_engine)

    columns = _columns(isolated_engine)
    # Legacy NOT NULL constraints relaxed.
    for column in _LEGACY_COLUMNS:
        assert columns[column] == "YES", f"{column} should have been made nullable"
    # New columns added and back-filled from the legacy ones.
    assert "miner_id" in columns and "benchmark_names_json" in columns
    with isolated_engine.connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT miner_id, benchmark_names_json FROM submissions WHERE id = 's1'"
        ).one()
    assert row[0] == "team-a"
    assert row[1] == ["math500"]
