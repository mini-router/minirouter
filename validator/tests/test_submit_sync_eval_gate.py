"""Offline tests for the synchronous-eval gate on ``POST /submit`` (issue #143).

A submission with no ``submission_artifact_id`` has no uploaded model bundle
(an infra/bugfix PR that never ships ``best_theta.npy``). Running
``evaluate_submission`` on it only yields a spurious ``missing_checkpoint``
failure that marks the PR's submission ``failed`` — the behaviour that closes
infra PRs. The synchronous path must skip such submissions, exactly as the async
enqueue paths already do.

These tests exercise the pure ``_should_sync_eval`` decision, so they need no
database and run everywhere (the DB-backed validator tests skip without Postgres).
"""
from __future__ import annotations

from types import SimpleNamespace

from eval_backend.api.routes import _should_sync_eval
from eval_backend.core.config import PIPELINE_TRAIN_EVAL, Settings


def _submission(artifact_id):
    return SimpleNamespace(submission_artifact_id=artifact_id)


def test_no_artifact_is_never_sync_evaluated():
    """The #143 regression: a checkpoint-less submission must not be evaluated."""
    settings = Settings(sync_eval_on_submit=True)
    assert settings.uses_train_pipeline is False
    assert _should_sync_eval(settings, _submission(None)) is False


def test_artifact_present_is_sync_evaluated():
    settings = Settings(sync_eval_on_submit=True)
    assert _should_sync_eval(settings, _submission("artifact-123")) is True


def test_sync_eval_disabled_never_runs_inline():
    settings = Settings(sync_eval_on_submit=False)
    assert _should_sync_eval(settings, _submission("artifact-123")) is False


def test_train_pipeline_defers_to_async_even_with_artifact():
    settings = Settings(sync_eval_on_submit=True, pipeline_mode=PIPELINE_TRAIN_EVAL)
    assert settings.uses_train_pipeline is True
    assert _should_sync_eval(settings, _submission("artifact-123")) is False
