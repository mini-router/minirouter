from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..core.config import Settings
from ..models import EvaluationRun, Submission
from ..schemas import (
    EvaluationOut,
    HealthResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    SubmissionCreateResponse,
    SubmissionOut,
)
from ..services.eval_runner import evaluate_submission
from ..services.github import create_pr_submission
from ..services.storage import store_upload

router = APIRouter()


def _submission_to_schema(submission: Submission) -> SubmissionOut:
    ordered_evaluations = sorted(
        submission.evaluations,
        key=lambda run: (run.created_at or _utcnow(), run.id),
    )
    latest = ordered_evaluations[-1] if ordered_evaluations else None
    evaluations = [
        EvaluationOut(
            id=run.id,
            submission_id=run.submission_id,
            status=run.status,
            score=run.score,
            phase=run.phase,
            message=run.message,
            progress_current=run.progress_current,
            progress_total=run.progress_total,
            metrics=json.loads(run.metrics_json) if run.metrics_json else {},
            command=run.command,
            stdout=run.stdout,
            stderr=run.stderr,
            results_path=run.results_path,
            error=run.error,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
        )
        for run in submission.evaluations
    ]
    return SubmissionOut(
        id=submission.id,
        source=submission.source,
        team_name=submission.team_name,
        repo_full_name=submission.repo_full_name,
        pr_number=submission.pr_number,
        head_sha=submission.head_sha,
        artifact_name=submission.artifact_name,
        artifact_path=submission.artifact_path,
        artifact_sha256=submission.artifact_sha256,
        checkpoint_path=submission.checkpoint_path,
        benchmark=submission.benchmark,
        status=submission.status,
        latest_score=submission.latest_score,
        best_run_id=submission.best_run_id,
        current_phase=latest.phase if latest else None,
        current_message=latest.message if latest else None,
        current_progress_current=latest.progress_current if latest else None,
        current_progress_total=latest.progress_total if latest else None,
        created_at=submission.created_at,
        updated_at=submission.updated_at,
        evaluations=evaluations,
    )


def _evaluation_to_schema(run: EvaluationRun) -> EvaluationOut:
    return EvaluationOut(
        id=run.id,
        submission_id=run.submission_id,
        status=run.status,
        score=run.score,
        phase=run.phase,
        message=run.message,
        progress_current=run.progress_current,
        progress_total=run.progress_total,
        metrics=json.loads(run.metrics_json) if run.metrics_json else {},
        command=run.command,
        stdout=run.stdout,
        stderr=run.stderr,
        results_path=run.results_path,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
    )


def _safe_team_name(request: Request, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return request.headers.get("x-team-name") or request.query_params.get("team_name")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Session:
    return request.app.state.session_factory()


def _ensure_webhook_secret_configured(secret: str) -> str:
    configured = (secret or "").strip()
    if not configured or configured == "replace-me":
        raise HTTPException(
            status_code=500,
            detail="webhook secret is not configured; set GITHUB_WEBHOOK_SECRET",
        )
    return configured


def _verify_github_signature(raw_body: bytes, signature: str | None, secret: str) -> None:
    configured_secret = _ensure_webhook_secret_configured(secret)
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="missing github signature")
    expected = hmac.new(configured_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="invalid github signature")


def _verify_shared_secret(provided: str | None, secret: str) -> None:
    configured_secret = _ensure_webhook_secret_configured(secret)
    if not provided:
        raise HTTPException(status_code=401, detail="missing webhook secret")
    if not hmac.compare_digest(provided, configured_secret):
        raise HTTPException(status_code=401, detail="invalid webhook secret")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/submit", response_model=SubmissionCreateResponse)
async def submit(
    request: Request,
    file: UploadFile = File(...),
    team_name: str | None = None,
    repo_full_name: str | None = Form(None),
    pr_number: int | None = Form(None),
    head_sha: str | None = Form(None),
    settings: Settings = Depends(get_settings),
) -> SubmissionCreateResponse:
    session = get_session(request)
    try:
        submission_id = str(uuid4())
        artifact = store_upload(file, settings, submission_id)
        if repo_full_name and pr_number is not None:
            submission = create_pr_submission(
                session,
                settings,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                head_sha=head_sha,
                team_name=_safe_team_name(request, team_name),
                artifact=artifact,
            )
            submission.benchmark = settings.eval_benchmark
            submission.status = "queued"
        else:
            submission = Submission(
                id=submission_id,
                source="upload",
                team_name=_safe_team_name(request, team_name),
                artifact_name=artifact.name,
                artifact_path=str(artifact.path),
                artifact_sha256=artifact.sha256,
                checkpoint_path=str(artifact.checkpoint_path) if artifact.checkpoint_path else None,
                benchmark=settings.eval_benchmark,
                status="queued",
            )
            session.add(submission)
        session.flush()
        session.commit()

        if settings.sync_eval_on_submit:
            session.refresh(submission)
            evaluate_submission(session, submission, settings)
            session.commit()

        session.refresh(submission)
        evaluation = None
        if submission.best_run_id:
            run = session.get(EvaluationRun, submission.best_run_id)
            if run is not None:
                evaluation = _evaluation_to_schema(run)
        return SubmissionCreateResponse(
            submission=_submission_to_schema(submission),
            evaluation=evaluation,
        )
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@router.post("/webhooks/github", response_model=SubmissionCreateResponse)
async def github_webhook(request: Request, settings: Settings = Depends(get_settings)) -> SubmissionCreateResponse:
    raw_body = await request.body()
    _verify_github_signature(
        raw_body,
        request.headers.get("x-hub-signature-256"),
        settings.github_webhook_secret,
    )
    payload: dict[str, Any] = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    session = get_session(request)
    try:
        repository = payload.get("repository") or {}
        repo_full_name = repository.get("full_name")
        if repo_full_name and repo_full_name != settings.allowed_repo:
            raise HTTPException(status_code=403, detail="repository not allowed")

        pull_request = payload.get("pull_request") or {}
        pr_number = pull_request.get("number")
        head_sha = (pull_request.get("head") or {}).get("sha")
        team_name = payload.get("sender", {}).get("login")
        submission = create_pr_submission(
            session,
            settings,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            team_name=team_name,
        )
        session.commit()
        return SubmissionCreateResponse(submission=_submission_to_schema(submission), evaluation=None)
    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@router.post("/webhooks/github/submission", response_model=SubmissionCreateResponse)
async def github_submission_upload(
    request: Request,
    file: UploadFile = File(...),
    repo_full_name: str = Form(...),
    pr_number: int = Form(...),
    head_sha: str | None = Form(None),
    team_name: str | None = Form(None),
    settings: Settings = Depends(get_settings),
) -> SubmissionCreateResponse:
    _verify_shared_secret(request.headers.get("x-minirouter-webhook-secret"), settings.github_webhook_secret)
    session = get_session(request)
    try:
        submission = create_pr_submission(
            session,
            settings,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            team_name=team_name,
        )
        artifact = store_upload(file, settings, submission.id)
        submission.artifact_name = artifact.name
        submission.artifact_path = str(artifact.path)
        submission.artifact_sha256 = artifact.sha256
        submission.checkpoint_path = str(artifact.checkpoint_path) if artifact.checkpoint_path else None
        submission.benchmark = settings.eval_benchmark
        submission.status = "queued"
        submission.latest_score = None
        submission.best_run_id = None
        submission.updated_at = _utcnow()
        session.commit()
        return SubmissionCreateResponse(submission=_submission_to_schema(submission), evaluation=None)
    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/api/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(request: Request, submission_id: str) -> SubmissionOut:
    session = get_session(request)
    try:
        submission = (
            session.execute(
                select(Submission)
                .where(Submission.id == submission_id)
                .options(selectinload(Submission.evaluations))
            )
            .scalars()
            .one()
        )
        return _submission_to_schema(submission)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="submission not found") from exc
    finally:
        session.close()


@router.get("/api/evaluations/{evaluation_id}", response_model=EvaluationOut)
def get_evaluation(request: Request, evaluation_id: int) -> EvaluationOut:
    session = get_session(request)
    try:
        run = session.get(EvaluationRun, evaluation_id)
        if run is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return _evaluation_to_schema(run)
    finally:
        session.close()


@router.get("/api/leaderboard", response_model=LeaderboardResponse)
def leaderboard(request: Request, limit: int = 100) -> LeaderboardResponse:
    session = get_session(request)
    try:
        stmt = (
            select(Submission)
            .where(
                Submission.status == "completed",
                Submission.latest_score.isnot(None),
            )
            .order_by(Submission.latest_score.desc(), Submission.created_at.asc())
            .limit(max(1, min(limit, 500)))
        )
        items = session.execute(stmt).scalars().all()
        board: list[LeaderboardEntry] = []
        for idx, submission in enumerate(items, start=1):
            metrics: dict[str, Any] = {}
            if submission.best_run_id:
                run = session.get(EvaluationRun, submission.best_run_id)
                if run and run.metrics_json:
                    metrics = json.loads(run.metrics_json)
            board.append(
                LeaderboardEntry(
                    rank=idx,
                    submission_id=submission.id,
                    team=submission.team_name or submission.repo_full_name or submission.id[:8],
                    accuracy=submission.latest_score,
                    gsm8k=_metric_value(metrics, "gsm8k"),
                    mmlu=_metric_value(metrics, "mmlu"),
                    math=_metric_value(metrics, "math"),
                    humaneval=_metric_value(metrics, "humaneval"),
                    bbh=_metric_value(metrics, "bbh"),
                    params=_metric_int(metrics, "params"),
                    submitted=submission.created_at,
                    report=f"/api/submissions/{submission.id}",
                    status=submission.status,
                )
            )
        return LeaderboardResponse(items=board)
    finally:
        session.close()


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    return int(value) if isinstance(value, (int, float)) else None
