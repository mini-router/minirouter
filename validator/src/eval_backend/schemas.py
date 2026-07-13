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


class JobQueueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_type: str
    kind: str
    job_id: str
    submission_id: str | None = None
    train_id: int | None = None
    queue_name: str
    status: str
    priority: int
    dedupe_key: str | None = None
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    attempts: int
    max_attempts: int
    next_run_at: datetime | None = None
    last_error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class JobQueueResponse(BaseModel):
    items: list[JobQueueOut]


class HealthResponse(BaseModel):
    status: str = "ok"


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_at: datetime


class AdminLogoutResponse(BaseModel):
    ok: bool = True


class AdminMeResponse(BaseModel):
    username: str


class AdminRuntimeConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    benchmark_names: list[str] = Field(default_factory=list)
    eval_max_items: int
    eval_provider: str
    eval_models_config: str
    eval_execution_mode: str
    updated_at: datetime | None = None


class AdminRuntimeConfigUpdate(BaseModel):
    benchmark_names: list[str] = Field(default_factory=list)
    eval_max_items: int = 20
    eval_provider: str = "chutes"
    eval_models_config: str = "configs/models.chutes.yaml"
    eval_execution_mode: str = "remote_gpu"


class ReviewControlOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    enabled: bool
    started_by: str | None = None
    started_at: datetime | None = None
    updated_at: datetime
