from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..core.config import Settings
from ..models import AdminUser, CompetitionRuntimeConfig, EvaluationRun, JobQueue, Submission, TrainRun
from ..schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    AdminRuntimeConfigOut,
    AdminRuntimeConfigUpdate,
    AdminLogoutResponse,
    AdminMeResponse,
    EvaluationOut,
    HealthResponse,
    JobQueueOut,
    JobQueueResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    ReviewControlOut,
    TrainCreateRequest,
    TrainCreateResponse,
    TrainOut,
    SubmissionCreateResponse,
    SubmissionOut,
)
from ..services.admin_auth import (
    authenticate_admin_token,
    create_admin_session,
    revoke_admin_token,
    verify_password,
)
from ..services.runtime_config import (
    apply_runtime_defaults,
    get_runtime_config,
    update_runtime_config,
)
from ..services.review_control import get_review_control, pause_review, start_review
from ..services.eval_runner import evaluate_submission
from ..services.github import create_pr_submission
from ..services.github import set_commit_status
from ..services.artifacts import persist_stored_artifact
from ..services.queue import cancel_submission_jobs
from ..services.queue import enqueue_submission_pipeline_job
from ..services.queue import enqueue_train_job
from ..services.storage import store_upload

router = APIRouter()
admin_router = APIRouter(prefix="/api/admin")


def _latest_run(submission: Submission) -> tuple[datetime | None, str | None, str | None, int | None, int | None]:
    newest = None
    for candidate in list(submission.evaluations) + list(submission.trains):
        if newest is None:
            newest = candidate
            continue
        left = candidate.created_at or _utcnow()
        right = newest.created_at or _utcnow()
        if left > right or (left == right and getattr(candidate, "id", 0) > getattr(newest, "id", 0)):
            newest = candidate
    if newest is None:
        return None, None, None, None, None
    return (
        getattr(newest, "created_at", None),
        getattr(newest, "phase", None),
        getattr(newest, "message", None),
        getattr(newest, "progress_current", None),
        getattr(newest, "progress_total", None),
    )


def _submission_to_schema(submission: Submission) -> SubmissionOut:
    ordered_evaluations = sorted(
        submission.evaluations,
        key=lambda run: (run.created_at or _utcnow(), run.id),
    )
    ordered_trains = sorted(
        submission.trains,
        key=lambda run: (run.created_at or _utcnow(), run.id),
    )
    latest = ordered_evaluations[-1] if ordered_evaluations else None
    evaluations = [
        EvaluationOut(
            id=run.id,
            submission_id=run.submission_id,
            train_id=run.train_id,
            input_artifact_id=run.input_artifact_id,
            status=run.status,
            score=run.score,
            phase=run.phase,
            message=run.message,
            progress_current=run.progress_current,
            progress_total=run.progress_total,
            benchmark_names=list(run.benchmark_names_json or []),
            provider=run.provider,
            models_config=run.models_config,
            execution_mode=run.execution_mode,
            device=run.device,
            dtype=run.dtype,
            batch_size=run.batch_size,
            max_items=run.max_items,
            max_turns=run.max_turns,
            max_tokens=run.max_tokens,
            reasoning=run.reasoning,
            seed=run.seed,
            cost_usd=run.cost_usd,
            duration_seconds=run.duration_seconds,
            metrics=json.loads(run.metrics_json) if run.metrics_json else {},
            command=run.command,
            stdout=run.stdout,
            stderr=run.stderr,
            results_path=run.results_path,
            results_artifact_id=run.results_artifact_id,
            error=run.error,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
        )
        for run in submission.evaluations
    ]
    trains = [
        TrainOut(
            id=run.id,
            submission_id=run.submission_id,
            status=run.status,
            phase=run.phase,
            message=run.message,
            progress_current=run.progress_current,
            progress_total=run.progress_total,
            benchmark_names=list(run.benchmark_names_json or []),
            warmstart_artifact_id=run.warmstart_artifact_id,
            output_artifact_id=run.output_artifact_id,
            cost_usd=run.cost_usd,
            duration_seconds=run.duration_seconds,
            metrics=json.loads(run.metrics_json) if run.metrics_json else {},
            command=run.command,
            stdout=run.stdout,
            stderr=run.stderr,
            error=run.error,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
        )
        for run in ordered_trains
    ]
    _, current_phase, current_message, current_progress_current, current_progress_total = _latest_run(submission)
    return SubmissionOut(
        id=submission.id,
        source=submission.source,
        miner_id=submission.miner_id,
        team_name=submission.team_name,
        repo_full_name=submission.repo_full_name,
        pr_number=submission.pr_number,
        head_sha=submission.head_sha,
        benchmark=submission.benchmark,
        benchmarks=submission.benchmarks,
        status=submission.status,
        latest_score=submission.latest_score,
        latest_train_id=submission.latest_train_id,
        latest_eval_id=submission.latest_eval_id,
        best_eval_id=submission.best_eval_id,
        current_phase=current_phase,
        current_message=current_message,
        current_progress_current=current_progress_current,
        current_progress_total=current_progress_total,
        finished_at=submission.finished_at,
        duration_seconds=submission.duration_seconds,
        cost_usd=submission.cost_usd,
        submission_artifact_id=submission.submission_artifact_id,
        created_at=submission.created_at,
        updated_at=submission.updated_at,
        evaluations=evaluations,
        trains=trains,
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


def _train_to_schema(run: TrainRun) -> TrainOut:
    return TrainOut(
        id=run.id,
        submission_id=run.submission_id,
        status=run.status,
        phase=run.phase,
        message=run.message,
        progress_current=run.progress_current,
        progress_total=run.progress_total,
        benchmark_names=list(run.benchmark_names_json or []),
        warmstart_artifact_id=run.warmstart_artifact_id,
        output_artifact_id=run.output_artifact_id,
        cost_usd=run.cost_usd,
        duration_seconds=run.duration_seconds,
        metrics=json.loads(run.metrics_json) if run.metrics_json else {},
        command=run.command,
        stdout=run.stdout,
        stderr=run.stderr,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
    )


def _job_kind(job: JobQueue) -> str:
    payload = job.payload_json or {}
    if job.job_type == "train":
        return "train"
    if payload.get("train_id") is not None:
        return "evaluation"
    return "submission"


def _job_to_schema(job: JobQueue) -> JobQueueOut:
    payload = job.payload_json if isinstance(job.payload_json, dict) else {}
    train_id = payload.get("train_id")
    if isinstance(train_id, str) and train_id.isdigit():
        train_id = int(train_id)
    elif not isinstance(train_id, int):
        train_id = None
    return JobQueueOut(
        id=job.id,
        job_type=job.job_type,
        kind=_job_kind(job),
        job_id=job.job_id,
        submission_id=job.submission_id,
        train_id=train_id,
        queue_name=job.queue_name,
        status=job.status,
        priority=job.priority,
        dedupe_key=job.dedupe_key,
        claimed_by=job.claimed_by,
        claimed_at=job.claimed_at,
        heartbeat_at=job.heartbeat_at,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        next_run_at=job.next_run_at,
        last_error=job.last_error,
        payload=payload,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing admin token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="missing admin token")
    return token.strip()


def _require_admin_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AdminUser:
    session = get_session(request)
    try:
        token = _extract_bearer_token(authorization)
        user = authenticate_admin_token(session, token)
        session.commit()
        return user
    except HTTPException:
        session.rollback()
        raise
    finally:
        session.close()


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


def _runtime_settings(session: Session, settings: Settings) -> Settings:
    runtime = get_runtime_config(session, settings)
    return apply_runtime_defaults(settings, runtime)


def _runtime_config_to_schema(session: Session, settings: Settings) -> AdminRuntimeConfigOut:
    runtime = get_runtime_config(session, settings)
    row = session.get(CompetitionRuntimeConfig, 1)
    return AdminRuntimeConfigOut(
        benchmark_names=list(runtime.benchmark_names),
        eval_max_items=runtime.eval_max_items,
        eval_provider=runtime.eval_provider,
        eval_models_config=runtime.eval_models_config,
        eval_execution_mode=runtime.eval_execution_mode,
        updated_at=row.updated_at if row is not None else None,
    )


def _review_control_to_schema(session: Session) -> ReviewControlOut:
    row = get_review_control(session)
    return ReviewControlOut(
        enabled=row.enabled,
        started_by=row.started_by,
        started_at=row.started_at,
        updated_at=row.updated_at,
    )


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
    if repo_full_name and pr_number is not None:
        _verify_shared_secret(
            request.headers.get("x-minirouter-webhook-secret"),
            settings.github_webhook_secret,
        )
    session = get_session(request)
    try:
        runtime = get_runtime_config(session, settings)
        runtime_settings = apply_runtime_defaults(settings, runtime)
        submission_id = str(uuid4())
        artifact = store_upload(file, settings, submission_id)
        if repo_full_name and pr_number is not None:
            submission = create_pr_submission(
                session,
                runtime_settings,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                head_sha=head_sha,
                team_name=_safe_team_name(request, team_name),
                artifact=artifact,
            )
        else:
            submission = Submission(
                id=submission_id,
                source="upload",
                miner_id=_safe_team_name(request, team_name),
                benchmark_names_json=list(runtime.benchmark_names),
                status="queued",
            )
            session.add(submission)
            artifact_row = persist_stored_artifact(
                session,
                artifact,
                storage_backend=settings.artifact_storage_backend,
                submission_id=submission.id,
                meta_json={
                    "checkpoint_path": str(artifact.checkpoint_path) if artifact.checkpoint_path else None,
                    "extracted_root": str(artifact.extracted_root) if artifact.extracted_root else None,
                },
            )
            submission.submission_artifact_id = artifact_row.id
        if repo_full_name and pr_number is not None and submission.submission_artifact_id is not None:
            submission.status = "queued"
            submission.latest_score = None
            submission.latest_eval_id = None
            submission.best_eval_id = None
        if submission.submission_artifact_id is not None and (
            not settings.sync_eval_on_submit or settings.uses_train_pipeline
        ):
            enqueue_submission_pipeline_job(
                session,
                submission,
                runtime_settings,
                payload_json={
                    "submission_id": submission.id,
                    "benchmark_names": submission.benchmark_names_json,
                    "source": submission.source,
                },
            )
        session.flush()
        session.commit()

        if settings.sync_eval_on_submit and not settings.uses_train_pipeline:
            session.refresh(submission)
            evaluate_submission(session, submission, runtime_settings)
            session.commit()

        session.refresh(submission)
        evaluation = None
        if submission.best_eval_id:
            run = session.get(EvaluationRun, submission.best_eval_id)
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


@router.post("/api/trains", response_model=TrainCreateResponse)
def create_train_job(
    request: Request,
    payload: TrainCreateRequest,
    settings: Settings = Depends(get_settings),
) -> TrainCreateResponse:
    session = get_session(request)
    try:
        runtime = get_runtime_config(session, settings)
        submission = session.get(Submission, payload.submission_id) if payload.submission_id else None
        train = TrainRun(
            submission_id=payload.submission_id,
            source="manual",
            benchmark_names_json=payload.benchmark_names or list(runtime.benchmark_names),
            warmstart_artifact_id=payload.warmstart_artifact_id,
            status="queued",
            phase="queued",
            message="train job queued",
        )
        session.add(train)
        session.flush()
        enqueue_train_job(session, train, payload_json=payload.model_dump())
        if submission is not None:
            submission.latest_train_id = train.id
            submission.updated_at = _utcnow()
        session.commit()
        return TrainCreateResponse(train=_train_to_schema(train), job_id=str(train.id))
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
        runtime = get_runtime_config(session, settings)
        runtime_settings = apply_runtime_defaults(settings, runtime)
        repository = payload.get("repository") or {}
        repo_full_name = repository.get("full_name")
        if repo_full_name and repo_full_name != settings.allowed_repo:
            raise HTTPException(status_code=403, detail="repository not allowed")

        action = str(payload.get("action") or "").strip().lower()
        pull_request = payload.get("pull_request") or {}
        pr_number = pull_request.get("number")
        head_sha = (pull_request.get("head") or {}).get("sha")
        team_name = payload.get("sender", {}).get("login")
        submission = create_pr_submission(
            session,
            runtime_settings,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            team_name=team_name,
        )
        if action == "closed":
            if submission.status not in {"completed", "failed"}:
                submission.status = "closed"
                submission.updated_at = _utcnow()
            cancel_submission_jobs(session, submission.id, reason="pull request closed")
        else:
            if submission.submission_artifact_id is None and submission.status not in {
                "queued",
                "running",
                "completed",
                "failed",
                "closed",
                "cancelled",
            }:
                submission.status = "awaiting_ci"
            submission.updated_at = _utcnow()
        session.commit()
        try:
            commit_state: str | None = "pending"
            commit_description = "Awaiting submission upload"
            if action == "closed":
                if submission.status in {"completed", "failed"}:
                    commit_state = None
                else:
                    commit_state = "failure"
                    commit_description = "Pull request closed"
            if commit_state is not None:
                await set_commit_status(
                    settings,
                    submission,
                    state=commit_state,
                    description=commit_description,
                    target_url=f"{settings.public_site_url.rstrip('/')}/submission/{submission.id}",
                )
        except Exception:
            pass
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
        runtime = get_runtime_config(session, settings)
        runtime_settings = apply_runtime_defaults(settings, runtime)
        submission = create_pr_submission(
            session,
            runtime_settings,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            team_name=team_name,
        )
        artifact = store_upload(file, settings, submission.id)
        submission.status = "queued"
        submission.latest_score = None
        submission.latest_eval_id = None
        submission.best_eval_id = None
        artifact_row = persist_stored_artifact(
            session,
            artifact,
            storage_backend=settings.artifact_storage_backend,
            submission_id=submission.id,
            meta_json={
                "checkpoint_path": str(artifact.checkpoint_path) if artifact.checkpoint_path else None,
                "extracted_root": str(artifact.extracted_root) if artifact.extracted_root else None,
            },
        )
        submission.submission_artifact_id = artifact_row.id
        if not settings.sync_eval_on_submit or settings.uses_train_pipeline:
            enqueue_submission_pipeline_job(
                session,
                submission,
                settings,
                payload_json={
                    "submission_id": submission.id,
                    "benchmark_names": submission.benchmark_names_json,
                    "source": submission.source,
                },
            )
        session.commit()
        try:
            await set_commit_status(
                settings,
                submission,
                state="pending",
                description="Awaiting review start",
                target_url=f"{settings.public_site_url.rstrip('/')}/submission/{submission.id}",
            )
        except Exception:
            pass
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
                .options(
                    selectinload(Submission.evaluations),
                    selectinload(Submission.trains),
                    selectinload(Submission.submission_artifact),
                )
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
            if submission.best_eval_id:
                run = session.get(EvaluationRun, submission.best_eval_id)
                if run and run.metrics_json:
                    metrics = json.loads(run.metrics_json)
            board.append(
                LeaderboardEntry(
                    rank=idx,
                    submission_id=submission.id,
                    team=submission.miner_id or submission.repo_full_name or submission.id[:8],
                    miner_id=submission.miner_id,
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


@router.get("/api/jobs", response_model=JobQueueResponse)
def list_jobs(
    request: Request,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 100,
) -> JobQueueResponse:
    session = get_session(request)
    try:
        stmt = select(JobQueue).order_by(JobQueue.created_at.desc(), JobQueue.id.desc())
        if status:
            status_values = [part.strip() for part in status.split(",") if part.strip()]
            if status_values:
                stmt = stmt.where(JobQueue.status.in_(status_values))
        else:
            stmt = stmt.where(JobQueue.status.in_(("queued", "running")))
        if job_type:
            stmt = stmt.where(JobQueue.job_type == job_type.strip())
        stmt = stmt.limit(max(1, min(limit, 500)))
        items = session.execute(stmt).scalars().all()
        return JobQueueResponse(items=[_job_to_schema(job) for job in items])
    finally:
        session.close()


@admin_router.post("/login", response_model=AdminLoginResponse)
def admin_login(
    request: Request,
    payload: AdminLoginRequest,
    settings: Settings = Depends(get_settings),
) -> AdminLoginResponse:
    session = get_session(request)
    try:
        username = payload.username.strip()
        user = session.execute(select(AdminUser).where(AdminUser.username == username)).scalar_one_or_none()
        if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid username or password")
        admin_session, token = create_admin_session(session, user, settings)
        session.commit()
        return AdminLoginResponse(
            access_token=token,
            token_type="bearer",
            username=user.username,
            expires_at=admin_session.expires_at,
        )
    except HTTPException:
        session.rollback()
        raise
    finally:
        session.close()


@admin_router.post("/logout", response_model=AdminLogoutResponse)
def admin_logout(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AdminLogoutResponse:
    session = get_session(request)
    try:
        token = _extract_bearer_token(authorization)
        revoke_admin_token(session, token)
        session.commit()
        return AdminLogoutResponse(ok=True)
    except HTTPException:
        session.rollback()
        raise
    finally:
        session.close()


@admin_router.get("/me", response_model=AdminMeResponse)
def admin_me(
    request: Request,
    user: AdminUser = Depends(_require_admin_user),
) -> AdminMeResponse:
    return AdminMeResponse(username=user.username)


@admin_router.get("/config", response_model=AdminRuntimeConfigOut)
def admin_get_config(
    request: Request,
    settings: Settings = Depends(get_settings),
    user: AdminUser = Depends(_require_admin_user),
) -> AdminRuntimeConfigOut:
    session = get_session(request)
    try:
        return _runtime_config_to_schema(session, settings)
    finally:
        session.close()


@admin_router.put("/config", response_model=AdminRuntimeConfigOut)
def admin_update_config(
    request: Request,
    payload: AdminRuntimeConfigUpdate,
    settings: Settings = Depends(get_settings),
    user: AdminUser = Depends(_require_admin_user),
) -> AdminRuntimeConfigOut:
    session = get_session(request)
    try:
        update_runtime_config(
            session,
            settings,
            benchmark_names=payload.benchmark_names,
            eval_max_items=payload.eval_max_items,
            eval_provider=payload.eval_provider,
            eval_models_config=payload.eval_models_config,
            eval_execution_mode=payload.eval_execution_mode,
        )
        session.commit()
        return _runtime_config_to_schema(session, settings)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@admin_router.get("/review", response_model=ReviewControlOut)
def admin_get_review_control(
    request: Request,
    user: AdminUser = Depends(_require_admin_user),
) -> ReviewControlOut:
    session = get_session(request)
    try:
        return _review_control_to_schema(session)
    finally:
        session.close()


@admin_router.post("/review/start", response_model=ReviewControlOut)
def admin_start_review(
    request: Request,
    user: AdminUser = Depends(_require_admin_user),
) -> ReviewControlOut:
    session = get_session(request)
    try:
        start_review(session, started_by=user.username)
        session.commit()
        return _review_control_to_schema(session)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@admin_router.post("/review/pause", response_model=ReviewControlOut)
def admin_pause_review(
    request: Request,
    user: AdminUser = Depends(_require_admin_user),
) -> ReviewControlOut:
    session = get_session(request)
    try:
        pause_review(session)
        session.commit()
        return _review_control_to_schema(session)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@admin_router.get("/leaderboard", response_model=LeaderboardResponse)
def admin_leaderboard(
    request: Request,
    limit: int = 100,
    user: AdminUser = Depends(_require_admin_user),
) -> LeaderboardResponse:
    return leaderboard(request, limit=limit)


@admin_router.get("/jobs", response_model=JobQueueResponse)
def admin_list_jobs(
    request: Request,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 100,
    user: AdminUser = Depends(_require_admin_user),
) -> JobQueueResponse:
    return list_jobs(request, status=status, job_type=job_type, limit=limit)


@admin_router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def admin_get_submission(
    request: Request,
    submission_id: str,
    user: AdminUser = Depends(_require_admin_user),
) -> SubmissionOut:
    return get_submission(request, submission_id)


@admin_router.get("/evaluations/{evaluation_id}", response_model=EvaluationOut)
def admin_get_evaluation(
    request: Request,
    evaluation_id: int,
    user: AdminUser = Depends(_require_admin_user),
) -> EvaluationOut:
    return get_evaluation(request, evaluation_id)


@admin_router.post("/trains", response_model=TrainCreateResponse)
def admin_create_train_job(
    request: Request,
    payload: TrainCreateRequest,
    settings: Settings = Depends(get_settings),
    user: AdminUser = Depends(_require_admin_user),
) -> TrainCreateResponse:
    return create_train_job(request, payload, settings)


@admin_router.post("/submit", response_model=SubmissionCreateResponse)
async def admin_submit(
    request: Request,
    file: UploadFile = File(...),
    team_name: str | None = None,
    repo_full_name: str | None = Form(None),
    pr_number: int | None = Form(None),
    head_sha: str | None = Form(None),
    settings: Settings = Depends(get_settings),
    user: AdminUser = Depends(_require_admin_user),
) -> SubmissionCreateResponse:
    return await submit(
        request,
        file=file,
        team_name=team_name,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
        settings=settings,
    )


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    return int(value) if isinstance(value, (int, float)) else None
