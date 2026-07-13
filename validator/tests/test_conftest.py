"""CI guardrails for validator Postgres-backed tests."""
from __future__ import annotations

import pytest


def test_validator_engine_fixture_runs_in_ci(validator_engine):
    assert validator_engine is not None
