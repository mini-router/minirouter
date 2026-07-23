from __future__ import annotations

from eval_backend.core.config import Settings
from eval_backend.services.runtime_config import (
    apply_runtime_defaults,
    get_runtime_config,
    seed_runtime_config,
    update_runtime_config,
)


def test_runtime_config_persists_eval_batch_size(validator_session) -> None:
    settings = Settings(eval_batch_size=2)

    seed_runtime_config(validator_session, settings)
    update_runtime_config(
        validator_session,
        settings,
        benchmark_names=["math500"],
        eval_max_items=32,
        eval_batch_size=4,
        eval_provider="chutes",
        eval_models_config="configs/models.chutes.light.yaml",
        eval_execution_mode="remote_gpu",
    )

    runtime = get_runtime_config(validator_session, settings)
    runtime_settings = apply_runtime_defaults(settings, runtime)

    assert runtime.eval_batch_size == 4
    assert runtime_settings.eval_batch_size == 4
