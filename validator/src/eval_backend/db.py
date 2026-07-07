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
            "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS phase VARCHAR(64)"
        )
        conn.exec_driver_sql(
            "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS message TEXT"
        )
        conn.exec_driver_sql(
            "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS progress_current INTEGER"
        )
        conn.exec_driver_sql(
            "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS progress_total INTEGER"
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
