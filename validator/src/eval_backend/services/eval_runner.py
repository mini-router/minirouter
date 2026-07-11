from __future__ import annotations

import hashlib
import json
import os
import errno
import re
import shlex
import subprocess
import select
import time
import logging
import sys
import pty
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import EvaluationRun, Submission
from .artifacts import persist_stored_artifact
from .storage import StoredArtifact

logger = logging.getLogger("eval_backend.eval_runner")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


_PROGRESS_START_RE = re.compile(r"^\[submission\] item (\d+)/(\d+) start(?: id=(.+))?$")
_PROGRESS_DONE_RE = re.compile(
    r"^\[submission\] item (\d+)/(\d+) done (pass|fail) score=([0-9.]+)"
)
_PROGRESS_INIT_RE = re.compile(r"^\[submission\] model initiated(?: (.+))?$")
_PROGRESS_COMPLETE_RE = re.compile(r"^\[submission\] completed score=([0-9.]+)")

_COST_PRICES: dict[str, tuple[float, float]] = {
    "fireworks:accounts/fireworks/models/deepseek-v4-pro": (1.74, 3.48),
    "fireworks:accounts/fireworks/models/glm-5p2": (1.40, 4.40),
    "fireworks:accounts/fireworks/models/kimi-k2p6": (0.95, 4.00),
    "openrouter:nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    "openrouter:google/gemma-4-31b-it:free": (0.0, 0.0),
    "openrouter:openai/gpt-oss-120b:free": (0.0, 0.0),
    "openrouter:qwen/qwen3-coder:free": (0.0, 0.0),
    "chutes:deepseek-ai/DeepSeek-V3.2-TEE": (1.00, 1.00),
    "chutes:zai-org/GLM-5-TEE": (1.40, 4.40),
    "chutes:moonshotai/Kimi-K2.5-TEE": (0.66, 3.50),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": (0.15, 1.20),
    "chutes:google/gemma-4-31B-turbo-TEE": (0.12, 0.37),
    "chutes:Qwen/Qwen3-32B-TEE": (0.10, 0.42),
}


def _cost_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _ledger_cost_report(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.exists():
        return {
            "cost_usd": 0.0,
            "cost_missing": True,
            "cost_ledger": str(ledger_path),
        }

    per_model: dict[str, dict[str, float | int]] = {}
    total = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            provider = str(row.get("provider", "")).strip()
            model = str(row.get("m", "")).strip()
            pt = int(row.get("p", 0) or 0)
            ct = int(row.get("c", 0) or 0)
            pin, pout = _COST_PRICES.get(_cost_key(provider, model), (0.0, 0.0))
            usd = pt / 1e6 * pin + ct / 1e6 * pout
            total += usd
            prompt_tokens += pt
            completion_tokens += ct
            calls += 1
            bucket = per_model.setdefault(
                _cost_key(provider, model),
                {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usd": 0.0},
            )
            bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + pt
            bucket["completion_tokens"] = int(bucket["completion_tokens"]) + ct
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["usd"] = float(bucket["usd"]) + usd

    return {
        "cost_usd": round(total, 4),
        "cost_missing": False,
        "cost_ledger": str(ledger_path),
        "cost_calls": calls,
        "cost_prompt_tokens": prompt_tokens,
        "cost_completion_tokens": completion_tokens,
        "cost_per_model": {
            key: {
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "calls": int(row["calls"]),
                "usd": round(float(row["usd"]), 4),
            }
            for key, row in sorted(per_model.items())
        },
    }


def _attach_runtime_metrics(metrics: dict[str, Any], *, run: EvaluationRun, ledger_path: Path) -> dict[str, Any]:
    out = dict(metrics)
    if run.started_at and run.finished_at:
        out["duration_seconds"] = round(
            max(0.0, (run.finished_at - run.started_at).total_seconds()),
            2,
        )
    out.update(_ledger_cost_report(ledger_path))
    return out


def _touch_progress(
    session: Session,
    run: EvaluationRun,
    submission: Submission,
    *,
    phase: str | None = None,
    message: str | None = None,
    current: int | None = None,
    total: int | None = None,
    status: str | None = None,
) -> None:
    if phase is not None:
        run.phase = phase
    if message is not None:
        run.message = message
    if current is not None:
        run.progress_current = current
    if total is not None:
        run.progress_total = total
    if status is not None:
        run.status = status
        submission.status = status
    submission.updated_at = _utcnow()
    session.flush()


def _consume_progress_line(
    line: str,
    session: Session,
    run: EvaluationRun,
    submission: Submission,
) -> None:
    line = line.strip()
    if not line:
        return
    logger.info("%s", line)

    match = _PROGRESS_INIT_RE.match(line)
    if match:
        _touch_progress(
            session,
            run,
            submission,
            phase="model_initiated",
            message=match.group(1) or "model initiated",
        )
        return

    match = _PROGRESS_START_RE.match(line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        task_id = match.group(3) or ""
        detail = f"item {current}/{total} running"
        if task_id:
            detail = f"{detail} ({task_id})"
        _touch_progress(
            session,
            run,
            submission,
            phase="evaluation_running",
            message=detail,
            current=current,
            total=total,
        )
        return

    match = _PROGRESS_DONE_RE.match(line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        verdict = match.group(3)
        score = match.group(4)
        detail = f"item {current}/{total} {verdict} score={score}"
        _touch_progress(
            session,
            run,
            submission,
            phase="evaluation_running",
            message=detail,
            current=current,
            total=total,
        )
        return

    match = _PROGRESS_COMPLETE_RE.match(line)
    if match:
        _touch_progress(
            session,
            run,
            submission,
            phase="completed",
            message=f"completed score={match.group(1)}",
        )


def _run_bash_stream(
    command: str,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    *,
    on_line=None,
) -> tuple[int, str, str]:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(cwd),
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    chunks: list[str] = []
    line_buf = ""
    started = time.monotonic()
    try:
        while True:
            if timeout and time.monotonic() - started > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout, output="".join(chunks))
            ready, _, _ = select.select([master_fd], [], [], 1.0)
            if ready:
                try:
                    raw = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        raw = b""
                    else:
                        raise
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    if on_line:
                        line_buf += text
                        while "\n" in line_buf:
                            line, line_buf = line_buf.split("\n", 1)
                            on_line(line)
                elif proc.poll() is not None:
                    break
            if proc.poll() is not None and not ready:
                break
        try:
            leftover = os.read(master_fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                leftover = b""
            else:
                raise
        if leftover:
            text = leftover.decode("utf-8", errors="replace")
            chunks.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
            if on_line:
                line_buf += text
        if on_line and line_buf.strip():
            on_line(line_buf)
        rc = proc.wait()
        return rc, "".join(chunks), ""
    finally:
        os.close(master_fd)


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
    eval_batch_size: int,
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
        eval_batch_size=eval_batch_size,
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


def _is_missing_results_payload(metrics: dict[str, Any]) -> bool:
    """Return True when parsed metrics indicate missing result artifacts."""
    value = metrics.get("results_missing")
    return bool(value) if isinstance(value, bool) else False


def _build_remote_command(
    settings: Settings,
    checkpoint_path: Path,
    results_path: Path,
    ledger_path: Path,
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
        eval_batch_size=settings.eval_batch_size,
    )
    return (
        f"mkdir -p {shlex.quote(str(ledger_path.parent))} && : > {shlex.quote(str(ledger_path))} && "
        f"export TRINITY_COST_LEDGER={shlex.quote(str(ledger_path))}; "
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
    ledger_path: Path,
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
        eval_batch_size=settings.eval_batch_size,
    )
    return (
        f"mkdir -p {shlex.quote(str(ledger_path.parent))} && : > {shlex.quote(str(ledger_path))} && "
        f"export TRINITY_COST_LEDGER={shlex.quote(str(ledger_path))}; "
        f"cd {shlex.quote(str(repo_dir))} && source .venv/bin/activate && {formatted}"
    )


class RemoteConnectionError(RuntimeError):
    """Raised when the remote GPU host cannot be reached or prepared."""


def _remote_attempt(
    settings: Settings,
    checkpoint_path: Path,
    local_results_path: Path,
    local_ledger_path: Path,
    submission_id: str,
    env: dict[str, str],
    *,
    on_line=None,
) -> tuple[str, int, str, str]:
    host = settings.trinity_remote_host
    remote_workspace = _remote_workspace(settings, submission_id)
    remote_checkpoint = remote_workspace / checkpoint_path.name
    remote_results = remote_workspace / local_results_path.name
    remote_ledger = remote_workspace / local_ledger_path.name

    remote_command = _build_remote_command(
        settings,
        remote_checkpoint,
        remote_results,
        remote_ledger,
        remote_workspace,
    )
    try:
        subprocess.run(["ssh", host, "mkdir", "-p", _remote_path(remote_workspace)], check=True)
        subprocess.run(["rsync", "-az", str(checkpoint_path), f"{host}:{_remote_path(remote_checkpoint)}"], check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RemoteConnectionError(f"remote ssh setup failed: {exc}") from exc
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["ssh", "-tt", host, "bash", "-lc", remote_command],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        env=env,
    )
    os.close(slave_fd)
    lines: list[str] = []
    line_buf = ""
    started = time.monotonic()
    try:
        while True:
            if settings.eval_timeout_seconds and time.monotonic() - started > settings.eval_timeout_seconds:
                proc.kill()
                raise subprocess.TimeoutExpired(
                    cmd=["ssh", "-tt", host, "bash", "-lc", remote_command],
                    timeout=settings.eval_timeout_seconds,
                    output="".join(lines),
                )
            ready, _, _ = select.select([master_fd], [], [], 1.0)
            if ready:
                try:
                    raw = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        raw = b""
                    else:
                        raise
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    lines.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    if on_line:
                        line_buf += text
                        while "\n" in line_buf:
                            line, line_buf = line_buf.split("\n", 1)
                            on_line(line)
                elif proc.poll() is not None:
                    break
            if proc.poll() is not None and not ready:
                break
        if on_line and line_buf.strip():
            on_line(line_buf)
        rc = proc.wait()
        subprocess.run(
            [
                "rsync",
                "-az",
                f"{host}:{_remote_path(remote_workspace)}/",
                f"{_local_workspace(settings, submission_id)}/",
            ],
            check=True,
        )
        return remote_command, rc, "".join(lines), ""
    finally:
        os.close(master_fd)


def _local_attempt(
    settings: Settings,
    checkpoint_path: Path,
    local_results_path: Path,
    local_ledger_path: Path,
    submission_id: str,
    env: dict[str, str],
) -> tuple[str, int, str, str]:
    local_workspace = _local_workspace(settings, submission_id)
    local_workspace.mkdir(parents=True, exist_ok=True)
    local_command = _build_local_command(
        settings,
        checkpoint_path,
        local_results_path,
        local_ledger_path,
        local_workspace,
    )
    rc, out, err = _run_bash_stream(
        local_command,
        cwd=Path(settings.local_repo_dir).expanduser().resolve(),
        timeout=settings.eval_timeout_seconds,
        env=env,
    )
    return local_command, rc, out, err


@dataclass(slots=True)
class EvaluationResult:
    run: EvaluationRun
    score: float | None
    metrics: dict[str, Any]
    stdout: str
    stderr: str


def evaluate_submission(
    session: Session,
    submission: Submission,
    settings: Settings,
    *,
    checkpoint_path_override: Path | None = None,
    train_id: int | None = None,
    input_artifact_id: str | None = None,
) -> EvaluationResult:
    local_workspace = _local_workspace(settings, submission.id)
    local_workspace.mkdir(parents=True, exist_ok=True)
    local_results_path = local_workspace / "results.json"
    local_cost_ledger_path = local_workspace / "cost_ledger.jsonl"

    run = EvaluationRun(
        submission_id=submission.id,
        train_id=train_id,
        input_artifact_id=input_artifact_id,
        benchmark_names_json=list(submission.benchmark_names_json or []),
        provider=settings.eval_provider,
        models_config=settings.eval_models_config,
        execution_mode=settings.eval_execution_mode,
        device="cpu" if settings.eval_execution_mode == "local_cpu" else "cuda:0",
        dtype="float32" if settings.eval_execution_mode == "local_cpu" else "bfloat16",
        batch_size=settings.eval_batch_size,
        max_items=settings.eval_max_items,
        status="running",
        phase="processing",
        message="worker claimed submission",
        progress_current=0,
        progress_total=settings.eval_max_items,
        started_at=_utcnow(),
        command="",
        results_path=str(local_results_path),
    )
    session.add(run)
    session.flush()
    submission.status = "running"
    submission.updated_at = _utcnow()
    session.flush()

    checkpoint_source = checkpoint_path_override or (
        Path(submission.checkpoint_path).expanduser().resolve() if submission.checkpoint_path else None
    )
    if not checkpoint_source:
        error = f"submission {submission.id} does not have a checkpoint to evaluate"
        run.status = "failed"
        run.phase = "failed"
        run.message = error
        run.error = error
        run.finished_at = _utcnow()
        metrics = _attach_runtime_metrics({"missing_checkpoint": True}, run=run, ledger_path=local_cost_ledger_path)
        run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
        run.duration_seconds = metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
        run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        submission.status = "failed"
        submission.latest_eval_id = run.id
        submission.latest_score = None
        submission.finished_at = run.finished_at
        submission.duration_seconds = run.duration_seconds
        submission.cost_usd = run.cost_usd
        session.flush()
        return EvaluationResult(
            run=run,
            score=None,
            metrics=json.loads(run.metrics_json),
            stdout="",
            stderr=error,
        )

    checkpoint_path = Path(checkpoint_source).expanduser().resolve()

    env = os.environ.copy()
    env["TRINITY_SECRETS_FILE"] = settings.trinity_secrets_file
    env["EVAL_BENCHMARK"] = settings.eval_benchmark
    env["EVAL_MAX_ITEMS"] = str(settings.eval_max_items)
    env["EVAL_BATCH_SIZE"] = str(settings.eval_batch_size)
    env["TRINITY_COST_LEDGER"] = str(local_cost_ledger_path.resolve())
    env["CHECKPOINT_PATH"] = str(checkpoint_path)
    env["RESULTS_PATH"] = str(local_results_path.resolve())
    env["WORKSPACE_ROOT"] = str(settings.workspace_root.expanduser().resolve())
    env["ARTIFACT_ROOT"] = str(settings.artifact_root.expanduser().resolve())

    metrics: dict[str, Any] = {}
    score: float | None = None
    stdout = ""
    stderr = ""
    remote_error: str | None = None
    remote_connection_error: str | None = None
    execution_mode = settings.eval_execution_mode if settings.eval_execution_mode == "local_cpu" else "remote_gpu"

    attempts: list[str] = []
    if settings.eval_execution_mode != "local_cpu":
        try:
            _touch_progress(
                session,
                run,
                submission,
                phase="remote_gpu",
                message=f"launching remote gpu on {settings.trinity_remote_host}",
                current=0,
                total=settings.eval_max_items,
            )
            command, completed, out, err = _remote_attempt(
                settings,
                checkpoint_path,
                local_results_path,
                local_cost_ledger_path,
                submission.id,
                env,
                on_line=lambda line: _consume_progress_line(line, session, run, submission),
            )
            attempts.append(command)
            stdout = out
            stderr = err
            if completed != 0:
                raise subprocess.CalledProcessError(
                    completed, command, output=out, stderr=err
                )
        except RemoteConnectionError as exc:
            remote_connection_error = f"remote gpu connection failed: {exc}"
            remote_error = remote_connection_error
        except Exception as exc:
            remote_error = f"remote gpu attempt failed: {exc}"

    if remote_connection_error:
        run.status = "failed"
        run.phase = "failed"
        run.message = remote_connection_error
        run.error = remote_connection_error
        run.stdout = stdout
        run.stderr = stderr
        run.finished_at = _utcnow()
        run.command = " || ".join(attempts) if attempts else ""
        metrics = _attach_runtime_metrics(
            {
                "results_missing": True,
                "execution_mode": "remote_gpu",
                "local_fallback": False,
                "remote_error": remote_error,
                "remote_connection_error": remote_connection_error,
            },
            run=run,
            ledger_path=local_cost_ledger_path,
        )
        run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
        run.duration_seconds = metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
        run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        submission.status = "failed"
        submission.latest_eval_id = run.id
        submission.latest_score = None
        submission.finished_at = run.finished_at
        submission.duration_seconds = run.duration_seconds
        submission.cost_usd = run.cost_usd
        session.flush()
        return EvaluationResult(
            run=run,
            score=None,
            metrics=json.loads(run.metrics_json),
            stdout=stdout,
            stderr=stderr,
        )

    if remote_error and not settings.eval_allow_local_fallback:
        run.status = "failed"
        run.phase = "failed"
        run.message = "remote gpu evaluation failed and local fallback is disabled"
        run.error = remote_error
        run.stdout = stdout
        run.stderr = stderr
        run.finished_at = _utcnow()
        run.command = " || ".join(attempts) if attempts else ""
        metrics = _attach_runtime_metrics(
            {
                "results_missing": True,
                "execution_mode": "remote_gpu",
                "local_fallback": False,
                "remote_error": remote_error,
            },
            run=run,
            ledger_path=local_cost_ledger_path,
        )
        run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
        run.duration_seconds = metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
        run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        submission.status = "failed"
        submission.latest_eval_id = run.id
        submission.latest_score = None
        submission.finished_at = run.finished_at
        submission.duration_seconds = run.duration_seconds
        submission.cost_usd = run.cost_usd
        session.flush()
        return EvaluationResult(
            run=run,
            score=None,
            metrics=json.loads(run.metrics_json),
            stdout=stdout,
            stderr=stderr,
        )

    if remote_error or settings.eval_execution_mode == "local_cpu":
        if remote_error:
            execution_mode = "local_fallback"
        else:
            execution_mode = "local_cpu"
        try:
            _touch_progress(
                session,
                run,
                submission,
                phase="local_cpu",
                message="launching local cpu fallback",
                current=0,
                total=settings.eval_max_items,
            )
            command, completed, out, err = _local_attempt(
                settings,
                checkpoint_path,
                local_results_path,
                local_cost_ledger_path,
                submission.id,
                env,
            )
            attempts.append(command)
            stdout = out
            stderr = err
            if completed != 0:
                raise subprocess.CalledProcessError(
                    completed, command, output=out, stderr=err
                )
        except Exception as exc:
            run.status = "failed"
            run.phase = "failed"
            run.message = "; ".join(part for part in [remote_error, str(exc)] if part) or "evaluation failed"
            run.error = "; ".join(part for part in [remote_error, str(exc)] if part)
            run.stdout = stdout
            run.stderr = stderr
            run.finished_at = _utcnow()
            metrics = _attach_runtime_metrics(
                {
                    "results_missing": True,
                    "execution_mode": execution_mode,
                    "local_fallback": bool(remote_error),
                    "remote_error": remote_error,
                },
                run=run,
                ledger_path=local_cost_ledger_path,
            )
            run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
            run.duration_seconds = (
                metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
            )
            run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
            submission.status = "failed"
            submission.latest_eval_id = run.id
            submission.latest_score = None
            submission.finished_at = run.finished_at
            submission.duration_seconds = run.duration_seconds
            submission.cost_usd = run.cost_usd
            session.flush()
            return EvaluationResult(
                run=run,
                score=None,
                metrics=json.loads(run.metrics_json),
                stdout=stdout,
                stderr=stderr,
            )

    run.status = "completed"
    run.phase = "completed"
    run.finished_at = _utcnow()
    run.command = " || ".join(attempts) if attempts else ""

    metrics, score = _prepare_results(local_results_path, settings=settings)
    metrics["execution_mode"] = execution_mode
    metrics["local_fallback"] = bool(remote_error)
    if remote_error:
        metrics["remote_error"] = remote_error
    metrics = _attach_runtime_metrics(metrics, run=run, ledger_path=local_cost_ledger_path)
    if _is_missing_results_payload(metrics):
        error = "evaluation did not produce results.json"
        run.status = "failed"
        run.phase = "failed"
        run.message = error
        run.error = error
        run.score = None
        run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
        run.duration_seconds = (
            metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
        )
        run.stdout = stdout
        run.stderr = stderr
        run.finished_at = _utcnow()
        run.command = " || ".join(attempts) if attempts else ""
        if run.progress_total is None:
            run.progress_total = settings.eval_max_items
        run.progress_current = run.progress_total
        submission.status = "failed"
        submission.latest_score = None
        submission.latest_eval_id = run.id
        submission.finished_at = run.finished_at
        submission.duration_seconds = run.duration_seconds
        submission.cost_usd = run.cost_usd
        session.flush()
        return EvaluationResult(run=run, score=None, metrics=metrics, stdout=stdout, stderr=stderr)

    if score is not None and remote_error:
        run.message = f"completed with local fallback score={score:.4f}"
    elif score is not None:
        run.message = f"completed score={score:.4f}"
    elif remote_error:
        run.message = "completed with local fallback"
    else:
        run.message = "completed"
    run.score = score
    run.stdout = stdout
    run.stderr = stderr
    run.metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
    run.cost_usd = metrics.get("cost_usd") if isinstance(metrics.get("cost_usd"), (int, float)) else None
    run.duration_seconds = (
        metrics.get("duration_seconds") if isinstance(metrics.get("duration_seconds"), (int, float)) else None
    )

    submission.status = "completed"
    submission.latest_score = score
    submission.latest_eval_id = run.id
    submission.best_eval_id = run.id
    submission.finished_at = run.finished_at
    submission.duration_seconds = run.duration_seconds
    submission.cost_usd = run.cost_usd
    if run.progress_total is None:
        run.progress_total = settings.eval_max_items
    run.progress_current = run.progress_total

    if local_results_path.exists():
        result_artifact = persist_stored_artifact(
            session,
            StoredArtifact(
                name=local_results_path.name,
                path=local_results_path,
                sha256=_sha256_file(local_results_path),
            ),
            storage_backend=settings.artifact_storage_backend,
            evaluation_id=run.id,
            meta_json={"results_path": str(local_results_path)},
        )
        run.results_artifact_id = result_artifact.id

    session.flush()
    return EvaluationResult(run=run, score=score, metrics=metrics, stdout=stdout, stderr=stderr)
