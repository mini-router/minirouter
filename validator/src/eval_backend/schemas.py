from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    submission_id: str
    status: str
    score: float | None = None
    phase: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    command: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    results_path: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source: str
    team_name: str | None = None
    repo_full_name: str | None = None
    pr_number: int | None = None
    head_sha: str | None = None
    artifact_name: str
    artifact_path: str
    artifact_sha256: str
    checkpoint_path: str | None = None
    benchmark: str
    status: str
    latest_score: float | None = None
    best_run_id: int | None = None
    current_phase: str | None = None
    current_message: str | None = None
    current_progress_current: int | None = None
    current_progress_total: int | None = None
    created_at: datetime
    updated_at: datetime
    evaluations: list[EvaluationOut] = Field(default_factory=list)


class SubmissionCreateResponse(BaseModel):
    submission: SubmissionOut
    evaluation: EvaluationOut | None = None


class LeaderboardEntry(BaseModel):
    rank: int
    submission_id: str
    team: str
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
