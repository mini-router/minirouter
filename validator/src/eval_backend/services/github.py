from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import EvaluationRun, Submission
from .eval_runner import EvaluationResult
from .artifacts import persist_stored_artifact
from .storage import StoredArtifact

GITHUB_API_BASE = "https://api.github.com"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _repo_owner(repo_full_name: str | None) -> str | None:
    if not repo_full_name or "/" not in repo_full_name:
        return None
    return repo_full_name.split("/", 1)[0]


def _repo_name(repo_full_name: str | None) -> str | None:
    if not repo_full_name or "/" not in repo_full_name:
        return None
    return repo_full_name.split("/", 1)[1]


def _submission_query(session: Session, repo_full_name: str | None, pr_number: int | None) -> Submission | None:
    if not repo_full_name or pr_number is None:
        return None
    return session.execute(
        select(Submission).where(
            Submission.source == "github_pr",
            Submission.repo_full_name == repo_full_name,
            Submission.pr_number == pr_number,
        )
    ).scalar_one_or_none()


def create_pr_submission(
    session: Session,
    settings: Settings,
    *,
    repo_full_name: str | None,
    pr_number: int | None,
    head_sha: str | None,
    team_name: str | None = None,
    artifact: StoredArtifact | None = None,
    extra: dict[str, Any] | None = None,
) -> Submission:
    existing = _submission_query(session, repo_full_name, pr_number)
    if existing is not None:
        changed = False
        if team_name and team_name != existing.miner_id:
            existing.miner_id = team_name
            changed = True
        if head_sha and head_sha != existing.head_sha:
            existing.head_sha = head_sha
            changed = True
        if artifact is not None:
            artifact_row = persist_stored_artifact(
                session,
                artifact,
                storage_backend=settings.artifact_storage_backend,
                submission_id=existing.id,
                meta_json={
                    "checkpoint_path": str(artifact.checkpoint_path)
                    if artifact.checkpoint_path
                    else None,
                    "extracted_root": str(artifact.extracted_root) if artifact.extracted_root else None,
                },
            )
            existing.submission_artifact_id = artifact_row.id
            changed = True
        if changed:
            if artifact is not None:
                existing.status = "queued"
            elif existing.status not in {"completed", "failed", "running"}:
                existing.status = "awaiting_ci"
            existing.latest_score = None
            existing.latest_eval_id = None
            existing.best_eval_id = None
            existing.updated_at = _utcnow()
        return existing

    submission = Submission(
        id=str(uuid4()),
        source="github_pr",
        miner_id=team_name,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
        benchmark_names_json=[settings.eval_benchmark],
        status="queued" if artifact is not None else "awaiting_ci",
        created_at=_utcnow(),
        updated_at=_utcnow(),
    )
    session.add(submission)
    session.flush()
    if artifact is not None:
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
    return submission


def _format_metrics_table(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    priority_keys = [
        "accuracy",
        "score",
        "overall",
        "macro_avg",
        "gsm8k",
        "mmlu",
        "math",
        "humaneval",
        "bbh",
        "params",
        "duration_seconds",
        "cost_usd",
        "cost_calls",
        "cost_prompt_tokens",
        "cost_completion_tokens",
        "cost_missing",
    ]
    seen: set[str] = set()

    def add_row(key: str, value: Any) -> None:
        if key in seen:
            return
        if isinstance(value, (int, float)):
            if key in {"accuracy", "score", "overall", "macro_avg", "gsm8k", "mmlu", "math", "humaneval", "bbh"}:
                display = f"{value:.4f} ({value * 100:.2f}%)"
            elif key == "duration_seconds":
                display = f"{value:.2f}s"
            elif key == "cost_usd":
                display = f"${value:.4f}"
            elif key == "params":
                display = f"{int(value):,}"
            else:
                display = f"{value}"
        else:
            display = str(value)
        rows.append(f"| {key} | {display} |")
        seen.add(key)

    for key in priority_keys:
        if key in metrics:
            add_row(key, metrics[key])

    for key, value in metrics.items():
        if key not in seen and isinstance(value, (str, int, float, bool)):
            add_row(key, value)

    if not rows:
        return "| metric | value |\n| --- | --- |\n| status | no scalar metrics were returned |"
    return "| metric | value |\n| --- | --- |\n" + "\n".join(rows)


def build_submission_summary_markdown(
    submission: Submission,
    evaluation: EvaluationRun,
    *,
    metrics: dict[str, Any] | None = None,
) -> str:
    metrics = metrics or {}
    score_text = "pending"
    if evaluation.score is not None:
        score_text = f"{evaluation.score:.4f} ({evaluation.score * 100:.2f}%)"

    summary_line = (
        f"Submission for **{submission.miner_id or submission.repo_full_name or submission.id}** "
        f"on **{submission.benchmark}** completed with status **{evaluation.status}**."
    )
    if evaluation.status == "failed" and evaluation.error:
        summary_line = (
            f"Submission for **{submission.miner_id or submission.repo_full_name or submission.id}** "
            f"on **{submission.benchmark}** failed during evaluation."
        )

    progress_current = getattr(evaluation, "progress_current", None)
    progress_total = getattr(evaluation, "progress_total", None)
    table = [
        "| field | value |",
        "| --- | --- |",
        f"| submission | `{submission.id}` |",
        f"| PR | #{submission.pr_number if submission.pr_number is not None else 'n/a'} |",
        f"| repo | `{submission.repo_full_name or 'n/a'}` |",
        f"| benchmark | `{submission.benchmark}` |",
        f"| status | `{evaluation.status}` |",
        f"| phase | `{getattr(evaluation, 'phase', None) or 'n/a'}` |",
        f"| progress | {progress_current if progress_current is not None else 'n/a'}/{progress_total if progress_total is not None else 'n/a'} |",
        f"| score | {score_text} |",
        f"| started | {evaluation.started_at.isoformat() if evaluation.started_at else 'n/a'} |",
        f"| finished | {evaluation.finished_at.isoformat() if evaluation.finished_at else 'n/a'} |",
    ]

    parts = [
        "### MiniRouter evaluation result",
        "",
        summary_line,
        "",
        "\n".join(table),
    ]

    if metrics:
        parts.extend(["", "### Metrics", "", _format_metrics_table(metrics)])

    if evaluation.error:
        parts.extend(["", "### Error", "", f"`{evaluation.error}`"])

    return "\n".join(parts)


async def _github_request(
    settings: Settings,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    if not settings.github_access_token:
        raise RuntimeError("GITHUB_ACCESS_TOKEN is not configured")

    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE,
        headers={
            "Authorization": f"Bearer {settings.github_access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    ) as client:
        response = await client.request(method, path, json=json_body)
        response.raise_for_status()
        return response


async def post_pr_comment(settings: Settings, submission: Submission, body: str) -> None:
    owner = _repo_owner(submission.repo_full_name)
    repo = _repo_name(submission.repo_full_name)
    if owner is None or repo is None or submission.pr_number is None:
        return
    await _github_request(
        settings,
        "POST",
        f"/repos/{owner}/{repo}/issues/{submission.pr_number}/comments",
        json_body={"body": body},
    )


async def set_commit_status(
    settings: Settings,
    submission: Submission,
    *,
    state: str,
    description: str,
    context: str = "MiniRouter / submission",
    target_url: str | None = None,
) -> None:
    owner = _repo_owner(submission.repo_full_name)
    repo = _repo_name(submission.repo_full_name)
    if owner is None or repo is None or not submission.head_sha:
        return
    payload: dict[str, Any] = {
        "state": state,
        "description": description[:140],
        "context": context,
    }
    if target_url:
        payload["target_url"] = target_url
    await _github_request(
        settings,
        "POST",
        f"/repos/{owner}/{repo}/statuses/{submission.head_sha}",
        json_body=payload,
    )


async def merge_pull_request(settings: Settings, submission: Submission) -> None:
    owner = _repo_owner(submission.repo_full_name)
    repo = _repo_name(submission.repo_full_name)
    if owner is None or repo is None or submission.pr_number is None:
        return
    payload: dict[str, Any] = {
        "merge_method": settings.github_merge_method,
    }
    if submission.head_sha:
        payload["sha"] = submission.head_sha
    await _github_request(
        settings,
        "PUT",
        f"/repos/{owner}/{repo}/pulls/{submission.pr_number}/merge",
        json_body=payload,
    )


async def publish_submission_result(
    settings: Settings,
    submission: Submission,
    evaluation: EvaluationResult | EvaluationRun,
) -> None:
    if submission.source != "github_pr" or not settings.github_access_token:
        return

    run = evaluation.run if isinstance(evaluation, EvaluationResult) else evaluation
    metrics: dict[str, Any]
    if isinstance(evaluation, EvaluationResult):
        metrics = evaluation.metrics
    else:
        metrics = {}
        if run.metrics_json:
            try:
                import json

                metrics = json.loads(run.metrics_json)
            except Exception:
                metrics = {}

    body = build_submission_summary_markdown(submission, run, metrics=metrics)
    target_url = None
    if settings.public_site_url:
        target_url = f"{settings.public_site_url.rstrip('/')}/submission/{submission.id}"

    try:
        if settings.github_post_comment_on_eval:
            await post_pr_comment(settings, submission, body)
    except Exception:
        # Comment failures should not break the evaluation pipeline.
        pass

    commit_state = "pending"
    commit_description = "Evaluation queued"
    if run.status == "completed":
        commit_state = "success"
        commit_description = "Evaluation completed"
    elif run.status == "failed":
        commit_state = "failure"
        commit_description = "Evaluation failed"

    try:
        await set_commit_status(
            settings,
            submission,
            state=commit_state,
            description=commit_description,
            target_url=target_url,
        )
    except Exception:
        pass

    if settings.github_auto_merge_submissions and run.status == "completed" and run.score is not None:
        try:
            await merge_pull_request(settings, submission)
        except Exception:
            pass
