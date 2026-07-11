from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")
    miner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    repo_full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    head_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    benchmark_names_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    submission_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    latest_train_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_eval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_eval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    submission_artifact = relationship("Artifact", foreign_keys=[submission_artifact_id])
    trains = relationship("TrainRun", back_populates="submission", cascade="all, delete-orphan")
    evaluations = relationship("EvaluationRun", back_populates="submission", cascade="all, delete-orphan")

    @property
    def team_name(self) -> str | None:
        return self.miner_id

    @team_name.setter
    def team_name(self, value: str | None) -> None:
        self.miner_id = value

    @property
    def benchmark(self) -> str:
        if not self.benchmark_names_json:
            return "unknown"
        return ", ".join(self.benchmark_names_json)

    @benchmark.setter
    def benchmark(self, value: str | list[str]) -> None:
        if isinstance(value, list):
            self.benchmark_names_json = [str(item) for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            self.benchmark_names_json = [text] if text else []

    @property
    def benchmarks(self) -> list[str]:
        return list(self.benchmark_names_json or [])

    @benchmarks.setter
    def benchmarks(self, value: list[str]) -> None:
        self.benchmark_names_json = [str(item) for item in value if str(item).strip()]

    @property
    def best_run_id(self) -> int | None:
        return self.best_eval_id

    @best_run_id.setter
    def best_run_id(self, value: int | None) -> None:
        self.best_eval_id = value

    @property
    def artifact_name(self) -> str | None:
        artifact = self.submission_artifact
        if artifact is None:
            return None
        if artifact.file_names:
            return artifact.file_names[0]
        return artifact.storage_uri.rsplit("/", 1)[-1] if artifact.storage_uri else None

    @property
    def artifact_path(self) -> str | None:
        artifact = self.submission_artifact
        return artifact.storage_uri if artifact is not None else None

    @property
    def artifact_sha256(self) -> str | None:
        artifact = self.submission_artifact
        return artifact.sha256 if artifact is not None else None

    @property
    def checkpoint_path(self) -> str | None:
        artifact = self.submission_artifact
        if artifact is None:
            return None
        meta = artifact.meta_json or {}
        checkpoint_path = meta.get("checkpoint_path")
        if isinstance(checkpoint_path, str) and checkpoint_path.strip():
            return checkpoint_path
        storage_uri = artifact.storage_uri or ""
        if Path(storage_uri).suffix.lower() in {".npy", ".pt", ".pth"}:
            return storage_uri
        return None


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="huggingface")
    storage_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    file_names_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    submission_id: Mapped[str | None] = mapped_column(ForeignKey("submissions.id"), nullable=True)
    train_id: Mapped[int | None] = mapped_column(ForeignKey("trains.id"), nullable=True)
    evaluation_id: Mapped[int | None] = mapped_column(ForeignKey("evaluations.id"), nullable=True)
    meta_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    submission = relationship("Submission", foreign_keys=[submission_id])
    train = relationship("TrainRun", foreign_keys=[train_id])
    evaluation = relationship("EvaluationRun", foreign_keys=[evaluation_id])

    @property
    def file_names(self) -> list[str]:
        return list(self.file_names_json or [])

    @file_names.setter
    def file_names(self, value: list[str]) -> None:
        self.file_names_json = [str(item) for item in value if str(item).strip()]


class TrainRun(Base):
    __tablename__ = "trains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str | None] = mapped_column(ForeignKey("submissions.id"), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    benchmark_names_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    warmstart_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    output_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_current: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    stdout: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    submission = relationship("Submission", back_populates="trains", foreign_keys=[submission_id])
    warmstart_artifact = relationship("Artifact", foreign_keys=[warmstart_artifact_id])
    output_artifact = relationship("Artifact", foreign_keys=[output_artifact_id])

    @property
    def benchmarks(self) -> list[str]:
        return list(self.benchmark_names_json or [])

    @benchmarks.setter
    def benchmarks(self, value: list[str]) -> None:
        self.benchmark_names_json = [str(item) for item in value if str(item).strip()]


class EvaluationRun(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str | None] = mapped_column(ForeignKey("submissions.id"), nullable=True, index=True)
    train_id: Mapped[int | None] = mapped_column(ForeignKey("trains.id"), nullable=True, index=True)
    input_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    benchmark_names_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="fireworks")
    models_config: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="remote_gpu")
    device: Mapped[str] = mapped_column(String(32), nullable=False, default="cuda:0")
    dtype: Mapped[str] = mapped_column(String(32), nullable=False, default="bfloat16")
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_items: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(String(32), nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_current: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    stdout: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    results_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    submission = relationship("Submission", back_populates="evaluations")
    train = relationship("TrainRun", foreign_keys=[train_id])
    input_artifact = relationship("Artifact", foreign_keys=[input_artifact_id])
    results_artifact = relationship("Artifact", foreign_keys=[results_artifact_id])

    @property
    def benchmark(self) -> str:
        if not self.benchmark_names_json:
            return "unknown"
        return ", ".join(self.benchmark_names_json)

    @benchmark.setter
    def benchmark(self, value: str | list[str]) -> None:
        if isinstance(value, list):
            self.benchmark_names_json = [str(item) for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            self.benchmark_names_json = [text] if text else []

    @property
    def benchmarks(self) -> list[str]:
        return list(self.benchmark_names_json or [])

    @benchmarks.setter
    def benchmarks(self, value: list[str]) -> None:
        self.benchmark_names_json = [str(item) for item in value if str(item).strip()]


class JobQueue(Base):
    __tablename__ = "job_queues"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    submission_id: Mapped[str | None] = mapped_column(ForeignKey("submissions.id"), nullable=True, index=True)
    queue_name: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dedupe_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    submission = relationship("Submission", foreign_keys=[submission_id])
