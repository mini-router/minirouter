from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import EvaluationRun, Submission


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pointer_lookup(payload: Any, pointer: str) -> Any:
    current = payload
    for piece in pointer.split("."):
        if not piece:
            continue
        if isinstance(current, dict):
            current = current[piece]
        elif isinstance(current, list):
            current = current[int(piece)]
        else:
            raise KeyError(pointer)
    return current


def _flatten_metrics(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        flat: dict[str, Any] = {}
        for key in (
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
        ):
            if key in payload:
                flat[key] = payload[key]
        for key in ("results", "metrics", "TRINITY"):
            value = payload.get(key)
            if isinstance(value, dict):
                flat.update({k: v for k, v in value.items() if k not in flat})
        return flat
    return {}


def _extract_score(metrics: dict[str, Any]) -> float | None:
    for key in ("accuracy", "score", "overall", "macro_avg"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for nested_key in ("accuracy", "score", "mean"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, (int, float)):
                    return float(nested_value)
    return None


def _format_command(
    template: str,
    *,
    repo_dir: Path,
    checkpoint_path: Path,
    results_path: Path,
    workspace: Path,
    benchmark: str,
    provider: str,
    models_config: str,
    max_items: int,
) -> str:
    return template.format(
        repo_dir=str(repo_dir),
        checkpoint_path=str(checkpoint_path),
        artifact_path=str(checkpoint_path),
        results_path=str(results_path),
        workspace=str(workspace),
        benchmark=benchmark,
        provider=provider,
        models_config=models_config,
        max_items=max_items,
    )


def _run_bash(command: str, cwd: Path, timeout: int, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _remote_path(path: str | Path) -> str:
    return str(path)


def _remote_workspace(settings: Settings, submission_id: str) -> Path:
    return Path(settings.trinity_remote_workspace_root) / "submissions" / submission_id


def _local_workspace(settings: Settings, submission_id: str) -> Path:
    return settings.workspace_root.expanduser() / "submissions" / submission_id


def _prepare_results(
    results_path: Path,
    *,
    settings: Settings,
) -> tuple[dict[str, Any], float | None]:
    if not results_path.exists():
        return {"results_missing": True}, None

    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        selected = _pointer_lookup(payload, settings.eval_result_pointer)
    except Exception:
        selected = payload

    metrics = _flatten_metrics(selected if isinstance(selected, dict) else payload)
    if isinstance(selected, dict):
        metrics.update({k: v for k, v in selected.items() if k not in metrics})
    elif isinstance(selected, (int, float)):
        score = float(selected)
        return metrics or {"raw": payload}, score

    score = _extract_score(metrics)
    if not metrics:
        metrics = {"raw": payload}
    return metrics, score


def _build_remote_command(
    settings: Settings,
    checkpoint_path: Path,
    results_path: Path,
    workspace: Path,
) -> str:
    repo_dir = Path(settings.trinity_remote_dir).expanduser()
    formatted = _format_command(
        settings.remote_eval_command_template,
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        results_path=results_path,
        workspace=workspace,
        benchmark=settings.eval_benchmark,
        provider=settings.eval_provider,
        models_config=settings.eval_models_config,
        max_items=settings.eval_max_items,
    )
    return (
        f"export TRINITY_REMOTE_DIR={shlex.quote(str(repo_dir))}; "
        f"export TRINITY_GPU_INDEX={shlex.quote(str(getattr(settings, 'trinity_gpu_index', 5)))}; "
        f"cd {shlex.quote(str(repo_dir))} && "
        "source .venv/bin/activate && "
        "source scripts/remote_env.sh && "
        f"{formatted}"
    )


def _build_local_command(
    settings: Settings,
    checkpoint_path: Path,
    results_path: Path,
    workspace: Path,
) -> str:
    repo_dir = Path(settings.local_repo_dir).expanduser().resolve()
    formatted = _format_command(
        settings.local_eval_command_template,
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        results_path=results_path,
        workspace=workspace,
        benchmark=settings.eval_benchmark,
        provider=settings.eval_provider,
        models_config=settings.eval_models_config,
        max_items=settings.eval_max_items,
    )
    return f"cd {shlex.quote(str(repo_dir))} && source .venv/bin/activate && {formatted}"


def _remote_attempt(
    settings: Settings,
    checkpoint_path: Path,
    local_results_path: Path,
    submission_id: str,
    env: dict[str, str],
) -> tuple[str, subprocess.CompletedProcess[str], str, str]:
    host = settings.trinity_gpu_host
    remote_workspace = _remote_workspace(settings, submission_id)
    remote_checkpoint = remote_workspace / checkpoint_path.name
    remote_results = remote_workspace / local_results_path.name

    remote_command = _build_remote_command(settings, remote_checkpoint, remote_results, remote_workspace)
    subprocess.run(["ssh", host, "mkdir", "-p", _remote_path(remote_workspace)], check=True)
    subprocess.run(["rsync", "-az", str(checkpoint_path), f"{host}:{_remote_path(remote_checkpoint)}"], check=True)
    completed = subprocess.run(
        ["ssh", host, "bash", "-lc", remote_command],
        capture_output=True,
        text=True,
        timeout=settings.eval_timeout_seconds,
        env=env,
        check=False,
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            f"{host}:{_remote_path(remote_workspace)}/",
            f"{_local_workspace(settings, submission_id)}/",
        ],
        check=True,
    )
    return remote_command, completed, completed.stdout or "", completed.stderr or ""


def _local_attempt(
    settings: Settings,
    checkpoint_path: Path,
    local_results_path: Path,
    submission_id: str,
    env: dict[str, str],
) -> tuple[str, subprocess.CompletedProcess[str], str, str]:
    local_workspace = _local_workspace(settings, submission_id)
    local_workspace.mkdir(parents=True, exist_ok=True)
    local_command = _build_local_command(settings, checkpoint_path, local_results_path, local_workspace)
    completed = _run_bash(
        local_command,
        cwd=Path(settings.local_repo_dir).expanduser().resolve(),
        timeout=settings.eval_timeout_seconds,
        env=env,
    )
    return local_command, completed, completed.stdout or "", completed.stderr or ""


@dataclass(slots=True)
class EvaluationResult:
    run: EvaluationRun
    score: float | None
    metrics: dict[str, Any]
    stdout: str
    stderr: str


def evaluate_submission(session: Session, submission: Submission, settings: Settings) -> EvaluationResult:
    if not submission.checkpoint_path:
        raise ValueError(f"submission {submission.id} does not have a checkpoint to evaluate")

    local_workspace = _local_workspace(settings, submission.id)
    local_workspace.mkdir(parents=True, exist_ok=True)
    local_results_path = local_workspace / "results.json"
    checkpoint_path = Path(submission.checkpoint_path).expanduser().resolve()

    run = EvaluationRun(
        submission_id=submission.id,
        status="running",
        started_at=_utcnow(),
        command="",
        results_path=str(local_results_path),
    )
    session.add(run)
    session.flush()

    env = os.environ.copy()
    env["TRINITY_SECRETS_FILE"] = settings.trinity_secrets_file
    env["EVAL_BENCHMARK"] = settings.eval_benchmark
    env["EVAL_MAX_ITEMS"] = str(settings.eval_max_items)
    env["CHECKPOINT_PATH"] = str(checkpoint_path)
    env["RESULTS_PATH"] = str(local_results_path.resolve())
    env["WORKSPACE_ROOT"] = str(settings.workspace_root.expanduser().resolve())
    env["ARTIFACT_ROOT"] = str(settings.artifact_root.expanduser().resolve())

    metrics: dict[str, Any] = {}
    score: float | None = None
    stdout = ""
    stderr = ""
    remote_error: str | None = None

    attempts: list[str] = []
    if settings.eval_execution_mode != "local_cpu":
        try:
            command, completed, out, err = _remote_attempt(
                settings, checkpoint_path, local_results_path, submission.id, env
            )
            attempts.append(command)
            stdout = out
            stderr = err
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode, command, output=out, stderr=err
                )
        except Exception as exc:
            remote_error = f"remote gpu attempt failed: {exc}"

    if remote_error or settings.eval_execution_mode == "local_cpu":
        try:
            command, completed, out, err = _local_attempt(
                settings, checkpoint_path, local_results_path, submission.id, env
            )
            attempts.append(command)
            stdout = out
            stderr = err
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode, command, output=out, stderr=err
                )
        except Exception as exc:
            run.status = "failed"
            run.error = "; ".join(part for part in [remote_error, str(exc)] if part)
            run.stdout = stdout
            run.stderr = stderr
            run.finished_at = _utcnow()
            run.metrics_json = json.dumps({"results_missing": True}, ensure_ascii=False, sort_keys=True)
            submission.status = "failed"
            session.flush()
            return EvaluationResult(run=run, score=None, metrics={"results_missing": True}, stdout=stdout, stderr=stderr)

    metrics, score = _prepare_results(local_results_path, settings=settings)

    run.status = "completed"
    run.score = score
    run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
    run.stdout = stdout
    run.stderr = stderr
    run.finished_at = _utcnow()
    run.command = " || ".join(attempts) if attempts else ""
    submission.status = "completed"
    submission.latest_score = score
    submission.best_run_id = run.id

    session.flush()
    return EvaluationResult(run=run, score=score, metrics=metrics, stdout=stdout, stderr=stderr)
