from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from eval_backend.db import Base

# Imported for its side effect of registering every ORM model on ``Base.metadata``
# so ``create_all`` below builds the full validator schema regardless of which
# test module triggered collection.
import eval_backend.models  # noqa: F401

# Mirrors DEFAULT_DATABASE_URL in eval_backend.core.config but points at a
# throwaway database so a developer's local Postgres is never clobbered.
DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://minirouter:minirouter@127.0.0.1:5432/minirouter_test"
)

TRUNCATE_VALIDATOR_TABLES = text(
    "TRUNCATE TABLE job_queues, evaluations, trains, artifacts, submissions "
    "RESTART IDENTITY CASCADE"
)


def _test_database_url() -> str:
    return (
        os.environ.get("VALIDATOR_TEST_DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
        or DEFAULT_TEST_DATABASE_URL
    )


def _require_postgres_in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


@pytest.fixture(scope="session")
def validator_engine():
    """Session-scoped engine bound to the validator's real (Postgres) backend.

    The validator only supports Postgres in production (see
    ``eval_backend.db._ensure_postgres``), so the test suite exercises the same
    backend instead of an in-memory SQLite stand-in. When no Postgres instance is
    reachable (e.g. a laptop without a local DB), the dependent tests are skipped
    rather than failing, so the suite stays runnable while still covering the
    production path wherever a database is available. In CI, a missing Postgres
    backend is treated as a hard failure so coverage cannot silently no-op.
    """
    url = _test_database_url()
    if not make_url(url).drivername.startswith("postgresql"):
        message = f"validator tests require a Postgres database URL, got {url!r}"
        if _require_postgres_in_ci():
            pytest.fail(message)
        pytest.skip(message)

    engine = create_engine(url, future=True, pool_pre_ping=True)
    try:
        engine.connect().close()
    except SQLAlchemyError as exc:
        engine.dispose()
        message = f"Postgres test database unavailable at {url!r}: {exc}"
        if _require_postgres_in_ci():
            pytest.fail(message)
        pytest.skip(message)

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.execute(TRUNCATE_VALIDATOR_TABLES)
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_validator_db(request):
    uses_validator_db = {"validator_engine", "validator_session"} & set(request.fixturenames)
    if not uses_validator_db:
        yield
        return
    engine = request.getfixturevalue("validator_engine")
    yield
    with engine.begin() as conn:
        conn.execute(TRUNCATE_VALIDATOR_TABLES)


@pytest.fixture()
def validator_session(validator_engine):
    """A validator DB session isolated per test via an outer transaction rollback.

    Session settings mirror ``eval_backend.db.build_session_factory`` so the
    session behaves exactly like the one the running service uses.
    """
    connection = validator_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(
        bind=connection,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
