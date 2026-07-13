from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models import CompetitionRuntimeConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RuntimeDefaults:
    benchmark_names: list[str]
    eval_max_items: int
    eval_provider: str
    eval_models_config: str
    eval_execution_mode: str
    king_score: float


def _default_runtime(settings: Settings) -> RuntimeDefaults:
    return RuntimeDefaults(
        benchmark_names=[settings.eval_benchmark] if settings.eval_benchmark else ["math500"],
        eval_max_items=settings.eval_max_items,
        eval_provider=settings.eval_provider,
        eval_models_config=settings.eval_models_config,
        eval_execution_mode=settings.eval_execution_mode,
        king_score=settings.github_review_score_threshold,
    )


def seed_runtime_config(session: Session, settings: Settings) -> CompetitionRuntimeConfig:
    existing = session.execute(select(CompetitionRuntimeConfig).where(CompetitionRuntimeConfig.id == 1)).scalar_one_or_none()
    defaults = _default_runtime(settings)
    if existing is None:
        row = CompetitionRuntimeConfig(
            id=1,
            default_benchmark_names_json=list(defaults.benchmark_names),
            default_eval_max_items=defaults.eval_max_items,
            default_eval_provider=defaults.eval_provider,
            default_eval_models_config=defaults.eval_models_config,
            default_eval_execution_mode=defaults.eval_execution_mode,
            king_score=defaults.king_score,
        )
        session.add(row)
        session.flush()
        return row

    if not existing.default_benchmark_names_json:
        existing.default_benchmark_names_json = list(defaults.benchmark_names)
    if existing.default_eval_max_items <= 0:
        existing.default_eval_max_items = defaults.eval_max_items
    if not existing.default_eval_provider.strip():
        existing.default_eval_provider = defaults.eval_provider
    if not existing.default_eval_models_config.strip():
        existing.default_eval_models_config = defaults.eval_models_config
    if not existing.default_eval_execution_mode.strip():
        existing.default_eval_execution_mode = defaults.eval_execution_mode
    if existing.king_score is None:
        existing.king_score = defaults.king_score
    existing.updated_at = _utcnow()
    session.flush()
    return existing


def get_runtime_config(session: Session, settings: Settings) -> RuntimeDefaults:
    row = session.execute(
        select(CompetitionRuntimeConfig).where(CompetitionRuntimeConfig.id == 1)
    ).scalar_one_or_none()
    if row is None:
        row = seed_runtime_config(session, settings)
    return RuntimeDefaults(
        benchmark_names=list(row.default_benchmark_names_json or _default_runtime(settings).benchmark_names),
        eval_max_items=row.default_eval_max_items or settings.eval_max_items,
        eval_provider=row.default_eval_provider or settings.eval_provider,
        eval_models_config=row.default_eval_models_config or settings.eval_models_config,
        eval_execution_mode=row.default_eval_execution_mode or settings.eval_execution_mode,
        king_score=row.king_score if row.king_score is not None else _default_runtime(settings).king_score,
    )


def apply_runtime_defaults(settings: Settings, runtime: RuntimeDefaults) -> Settings:
    benchmark = runtime.benchmark_names[0] if runtime.benchmark_names else settings.eval_benchmark
    return replace(
        settings,
        eval_benchmark=benchmark,
        train_benchmark=benchmark,
        eval_max_items=runtime.eval_max_items,
        eval_provider=runtime.eval_provider,
        eval_models_config=runtime.eval_models_config,
        eval_execution_mode=runtime.eval_execution_mode,
    )


def update_runtime_config(
    session: Session,
    settings: Settings,
    *,
    benchmark_names: list[str],
    eval_max_items: int,
    eval_provider: str,
    eval_models_config: str,
    eval_execution_mode: str,
) -> CompetitionRuntimeConfig:
    row = session.execute(
        select(CompetitionRuntimeConfig).where(CompetitionRuntimeConfig.id == 1)
    ).scalar_one_or_none()
    if row is None:
        row = CompetitionRuntimeConfig(id=1)
        session.add(row)
        session.flush()
    clean_benchmarks = [item.strip() for item in benchmark_names if item and item.strip()]
    row.default_benchmark_names_json = clean_benchmarks or _default_runtime(settings).benchmark_names
    row.default_eval_max_items = max(1, int(eval_max_items))
    row.default_eval_provider = eval_provider.strip() or settings.eval_provider
    row.default_eval_models_config = eval_models_config.strip() or settings.eval_models_config
    mode = eval_execution_mode.strip().lower()
    row.default_eval_execution_mode = mode if mode in {"local_cpu", "remote_gpu"} else settings.eval_execution_mode
    if row.king_score is None:
        row.king_score = _default_runtime(settings).king_score
    row.updated_at = _utcnow()
    session.flush()
    return row


def update_king_score(session: Session, settings: Settings, king_score: float) -> CompetitionRuntimeConfig:
    row = session.execute(
        select(CompetitionRuntimeConfig).where(CompetitionRuntimeConfig.id == 1)
    ).scalar_one_or_none()
    if row is None:
        row = seed_runtime_config(session, settings)
    row.king_score = float(king_score)
    row.updated_at = _utcnow()
    session.flush()
    return row
