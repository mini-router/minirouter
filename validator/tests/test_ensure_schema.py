from __future__ import annotations

from sqlalchemy import text

from eval_backend.db import Base, ensure_schema
import eval_backend.models  # noqa: F401  — register ORM metadata

_LEGACY_COLUMNS = (
    "artifact_name",
    "artifact_path",
    "artifact_sha256",
    "checkpoint_path",
    "benchmark",
    "team_name",
)


def test_ensure_schema_succeeds_on_fresh_create_all(validator_engine) -> None:
    """Fresh DBs have no legacy submission columns; ensure_schema must not crash."""
    Base.metadata.create_all(validator_engine)
    ensure_schema(validator_engine)

    with validator_engine.connect() as conn:
        columns = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'submissions' AND table_schema = current_schema()"
                )
            )
        }
    assert "miner_id" in columns
    assert "benchmark_names_json" in columns
    # These are @property accessors on the current model — not real columns.
    assert "artifact_name" not in columns
    assert "team_name" not in columns


def test_ensure_schema_relaxes_and_backfills_legacy_columns(validator_engine) -> None:
    """When legacy columns exist, ensure_schema should relax NOT NULL and back-fill."""
    Base.metadata.create_all(validator_engine)

    try:
        with validator_engine.begin() as conn:
            for column, coltype in (
                ("artifact_name", "VARCHAR(255)"),
                ("artifact_path", "VARCHAR(1024)"),
                ("artifact_sha256", "VARCHAR(64)"),
                ("checkpoint_path", "VARCHAR(1024)"),
                ("benchmark", "VARCHAR(64)"),
                ("team_name", "VARCHAR(255)"),
            ):
                conn.execute(
                    text(
                        f"ALTER TABLE submissions ADD COLUMN IF NOT EXISTS {column} {coltype}"
                    )
                )
                conn.execute(text(f"UPDATE submissions SET {column} = '' WHERE {column} IS NULL"))
                conn.execute(text(f"ALTER TABLE submissions ALTER COLUMN {column} SET NOT NULL"))

            conn.execute(text("DELETE FROM submissions WHERE id = 'legacy-1'"))
            conn.execute(
                text(
                    """
                    INSERT INTO submissions (
                        id, source, team_name, artifact_name, artifact_path,
                        artifact_sha256, checkpoint_path, benchmark, status,
                        benchmark_names_json, created_at, updated_at
                    ) VALUES (
                        'legacy-1', 'upload', 'old-team', 'a.npy', '/tmp/a.npy',
                        'deadbeef', '/tmp/ckpt.npy', 'math500', 'queued',
                        '[]'::json, NOW(), NOW()
                    )
                    """
                )
            )
            # Make the back-fill path fire: NULL benchmark_names_json + non-null benchmark.
            conn.execute(
                text("ALTER TABLE submissions ALTER COLUMN benchmark_names_json DROP NOT NULL")
            )
            conn.execute(
                text("UPDATE submissions SET benchmark_names_json = NULL WHERE id = 'legacy-1'")
            )
            conn.execute(text("UPDATE submissions SET miner_id = NULL WHERE id = 'legacy-1'"))

        ensure_schema(validator_engine)

        with validator_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT miner_id, benchmark_names_json::text, "
                    "(SELECT is_nullable FROM information_schema.columns "
                    " WHERE table_name = 'submissions' AND column_name = 'artifact_name' "
                    "   AND table_schema = current_schema()) "
                    "FROM submissions WHERE id = 'legacy-1'"
                )
            ).one()
        assert row[0] == "old-team"
        assert "math500" in (row[1] or "")
        assert row[2] == "YES"
    finally:
        # Restore a clean modern schema for later tests sharing this engine.
        with validator_engine.begin() as conn:
            for column in _LEGACY_COLUMNS:
                conn.execute(text(f"ALTER TABLE submissions DROP COLUMN IF EXISTS {column}"))
            conn.execute(
                text("ALTER TABLE submissions ALTER COLUMN benchmark_names_json SET NOT NULL")
            )
            conn.execute(text("DELETE FROM submissions WHERE id = 'legacy-1'"))
