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
from ..models import Artifact, Submission, TrainRun
from .artifacts import persist_stored_artifact
from .eval_runner import _ledger_cost_report
from .storage import StoredArtifact

logger = logging.getLogger("eval_backend.train_runner")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_workspace(settings: Settings, train_id: int) -> Path:
    return settings.workspace_root.expanduser() / "trains" / str(train_id)


def _run_name(settings: Settings, train: TrainRun) -> str:
    return f"{settings.train_run_name_prefix}-{train.id}"


def _train_run_dir(settings: Settings, train: TrainRun) -> Path:
    benchmark = train.benchmark_names_json[0] if train.benchmark_names_json else settings.train_benchmark
    return Path(settings.local_repo_dir).expanduser().resolve() / "experiments" / benchmark / _run_name(
        settings, train
    )


def _format_command(
    template: str,
    *,
    repo_dir: Path,
    benchmark: str,
    provider: str,
    models_config: str,
    config: str,
    device: str,
    dtype: str,
    max_items: int,
    generations: int,
    popsize: int,
    m_cma: int,
    run_name: str,
    warmstart_theta: str,
) -> str:
    return template.format(
        repo_dir=str(repo_dir),
        benchmark=benchmark,
        provider=provider,
        models_config=models_config,
        config=config,
        device=device,
        dtype=dtype,
        max_items=max_items,
        generations=generations,
        popsize=popsize,
        m_cma=m_cma,
        run_name=run_name,
        warmstart_theta=warmstart_theta,
    )


def _build_train_command(settings: Settings, train: TrainRun, *, workspace: Path) -> str:
    repo_dir = Path(settings.local_repo_dir).expanduser().resolve()
    benchmark = train.benchmark_names_json[0] if train.benchmark_names_json else settings.train_benchmark
    warmstart_theta = ""
    if train.warmstart_artifact_id and train.warmstart_artifact:
        warm_meta = train.warmstart_artifact.meta_json or {}
        warmstart_theta = warm_meta.get("checkpoint_path") or train.warmstart_artifact.storage_uri
    warmstart_clause = f" --warmstart-theta {shlex.quote(warmstart_theta)}" if warmstart_theta else ""
    formatted = (
        "PYTHONPATH=src PYTHONUNBUFFERED=1 python -u -m trinity.train "
        f"--benchmark {shlex.quote(benchmark)} "
        f"--config {shlex.quote(settings.train_config)} "
        f"--models {shlex.quote(settings.train_models_config)} "
        f"--provider {shlex.quote(settings.train_provider)} "
        f"--device {shlex.quote(settings.train_device)} "
        f"--dtype {shlex.quote(settings.train_dtype)} "
        f"--max-items {settings.train_max_items} "
        f"--generations {settings.train_generations} "
        f"--popsize {settings.train_popsize} "
        f"--m-cma {settings.train_m_cma} "
        f"--run-name {shlex.quote(_run_name(settings, train))}"
        f"{warmstart_clause}"
    )
    return (
        f"mkdir -p {shlex.quote(str(workspace))} && "
        f"cd {shlex.quote(str(repo_dir))} && source .venv/bin/activate && "
        f"{formatted}"
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
        if on_line and line_buf.strip():
            on_line(line_buf)
        rc = proc.wait()
        return rc, "".join(chunks), ""
    finally:
        os.close(master_fd)


def _artifact_from_run_dir(session: Session, settings: Settings, train: TrainRun, run_dir: Path) -> Artifact:
    best_theta = run_dir / "best_theta.npy"
    summary = run_dir / "summary.json"
    history = run_dir / "history.json"
    files = [name for name in [best_theta.name, summary.name] if (run_dir / name).exists()]
    if history.exists():
        files.append(history.name)
    digest = hashlib.sha256()
    for item in sorted(run_dir.glob("*")):
        if item.is_file():
            digest.update(item.name.encode("utf-8"))
            digest.update(_sha256_file(item).encode("utf-8"))
    artifact = Artifact(
        id=f"train-artifact-{train.id}",
        storage_backend=settings.artifact_storage_backend,
        storage_uri=str(run_dir),
        file_names_json=files,
        sha256=digest.hexdigest(),
        size_bytes=sum(item.stat().st_size for item in run_dir.glob("*") if item.is_file()),
        mime_type=None,
        train_id=train.id,
        meta_json={
            "checkpoint_path": str(best_theta),
            "summary_path": str(summary) if summary.exists() else None,
            "history_path": str(history) if history.exists() else None,
        },
    )
    session.add(artifact)
    session.flush()
    return artifact


@dataclass(slots=True)
class TrainResult:
    train: TrainRun
    output_artifact: Artifact | None
    stdout: str
    stderr: str


def run_train_job(session: Session, train: TrainRun, settings: Settings) -> TrainResult:
    workspace = _train_workspace(settings, train.id)
    workspace.mkdir(parents=True, exist_ok=True)
    cost_ledger_path = workspace / "cost_ledger.jsonl"
    train.started_at = _utcnow()
    train.status = "running"
    train.phase = "training"
    train.message = "worker claimed train job"
    train.progress_current = 0
    train.progress_total = settings.train_generations
    session.flush()

    env = os.environ.copy()
    env["TRINITY_SECRETS_FILE"] = settings.trinity_secrets_file
    env["TRINITY_COST_LEDGER"] = str(cost_ledger_path.resolve())
    env["EVAL_BENCHMARK"] = train.benchmark_names_json[0] if train.benchmark_names_json else settings.train_benchmark
    env["EVAL_MAX_ITEMS"] = str(settings.train_max_items)
    env["TRAIN_GENERATIONS"] = str(settings.train_generations)
    env["TRAIN_POPSIZE"] = str(settings.train_popsize)
    env["TRAIN_M_CMA"] = str(settings.train_m_cma)

    command = _build_train_command(settings, train, workspace=workspace)
    rc, stdout, stderr = _run_bash_stream(
        command,
        cwd=Path(settings.local_repo_dir).expanduser().resolve(),
        timeout=settings.eval_timeout_seconds,
        env=env,
    )
    train.command = command
    train.stdout = stdout
    train.stderr = stderr
    train.finished_at = _utcnow()
    train.duration_seconds = round((train.finished_at - train.started_at).total_seconds(), 2) if train.started_at else None
    cost_metrics = _ledger_cost_report(cost_ledger_path)

    run_dir = _train_run_dir(settings, train)
    output_artifact = None
    if rc == 0 and run_dir.exists():
        output_artifact = _artifact_from_run_dir(session, settings, train, run_dir)
        train.output_artifact_id = output_artifact.id
        train.status = "completed"
        train.phase = "completed"
        train.message = f"completed run_name={_run_name(settings, train)}"
    else:
        train.status = "failed"
        train.phase = "failed"
        train.message = f"train exited with code {rc}"
        train.error = f"train exited with code {rc}"

    train.metrics_json = json.dumps(
        {
            "run_dir": str(run_dir),
            "return_code": rc,
            "duration_seconds": train.duration_seconds,
            **cost_metrics,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    train.cost_usd = cost_metrics.get("cost_usd") if isinstance(cost_metrics.get("cost_usd"), (int, float)) else None
    if train.submission_id:
        submission = session.get(Submission, train.submission_id)
        if submission is not None:
            submission.latest_train_id = train.id
            submission.updated_at = _utcnow()
            submission.cost_usd = train.cost_usd
            if train.status == "failed":
                submission.status = "failed"
    session.flush()
    return TrainResult(train=train, output_artifact=output_artifact, stdout=stdout, stderr=stderr)
