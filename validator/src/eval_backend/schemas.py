from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    submission_id: str | None = None
    train_id: int | None = None
    input_artifact_id: str | None = None
    status: str
    score: float | None = None
    phase: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    benchmark_names: list[str] = Field(default_factory=list)
    provider: str | None = None
    models_config: str | None = None
    execution_mode: str | None = None
    device: str | None = None
    dtype: str | None = None
    batch_size: int | None = None
    max_items: int | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    seed: int | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    command: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    results_path: str | None = None
    results_artifact_id: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class TrainOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    submission_id: str | None = None
    status: str
    phase: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    benchmark_names: list[str] = Field(default_factory=list)
    warmstart_artifact_id: str | None = None
    output_artifact_id: str | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    command: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class TrainCreateRequest(BaseModel):
    benchmark_names: list[str] = Field(default_factory=list)
    submission_id: str | None = None
    warmstart_artifact_id: str | None = None


class TrainCreateResponse(BaseModel):
    train: TrainOut
    job_id: str


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source: str
    miner_id: str | None = None
    team_name: str | None = None
    repo_full_name: str | None = None
    pr_number: int | None = None
    head_sha: str | None = None
    benchmark: str
    benchmarks: list[str] = Field(default_factory=list)
    status: str
    latest_score: float | None = None
    latest_train_id: int | None = None
    latest_eval_id: int | None = None
    best_eval_id: int | None = None
    current_phase: str | None = None
    current_message: str | None = None
    current_progress_current: int | None = None
    current_progress_total: int | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    cost_usd: float | None = None
    submission_artifact_id: str | None = None
    created_at: datetime
    updated_at: datetime
    evaluations: list[EvaluationOut] = Field(default_factory=list)
    trains: list[TrainOut] = Field(default_factory=list)


class SubmissionCreateResponse(BaseModel):
    submission: SubmissionOut
    evaluation: EvaluationOut | None = None


class LeaderboardEntry(BaseModel):
    rank: int
    submission_id: str
    team: str
    miner_id: str | None = None
    accuracy: float | None = None
    gsm8k: float | None = None
    mmlu: float | None = None
    math: float | None = None
    humaneval: float | None = None
    bbh: float | None = None
    params: int | None = None
    submitted: datetime
    report: str
    status: str


class LeaderboardResponse(BaseModel):
    items: list[LeaderboardEntry]


class HealthResponse(BaseModel):
    status: str = "ok"
