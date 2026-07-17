from __future__ import annotations

import pytest

from trinity.eval import _aggregate_runs, _select_pool_models, _selected_benchmarks


class DummyPool:
    models = {
        "glm-5": "zai-org/GLM-5-TEE",
        "openrouter-glm-5p2": "z-ai/glm-5.2",
    }

    def describe_model(self, model: str) -> tuple[str, str]:
        if model == "glm-5":
            return "chutes", "zai-org/GLM-5-TEE"
        if model == "openrouter-glm-5p2":
            return "openrouter", "z-ai/glm-5.2"
        raise KeyError(model)


class Args:
    benchmark = ""
    benchmarks = ""


def test_selected_benchmarks_accepts_grouped_values() -> None:
    args = Args()
    args.benchmarks = "math500, mmlu"

    assert _selected_benchmarks(args) == ["math500", "mmlu"]


def test_select_pool_models_accepts_provider_prefixed_alias() -> None:
    selected = _select_pool_models(DummyPool(), "chutes-glm-5,openrouter-glm-5")

    assert selected == ["glm-5", "openrouter-glm-5p2"]


def test_select_pool_models_rejects_unknown_route() -> None:
    with pytest.raises(ValueError, match="unknown pool model"):
        _select_pool_models(DummyPool(), "chutes-kimi")


def test_aggregate_runs_averages_repeat_scores() -> None:
    out = _aggregate_runs(
        [
            {"benchmark": "math500", "results": {"single::chutes-glm-5": 0.5}},
            {"benchmark": "math500", "results": {"single::chutes-glm-5": 1.0}},
        ],
        repeat=2,
        pool_models=["chutes-glm-5"],
    )

    assert out["results_by_benchmark"]["math500"]["single::chutes-glm-5"] == 0.75
    assert out["results_by_benchmark"]["math500"]["single::chutes-glm-5::repeats"] == [0.5, 1.0]
    assert out["summary"]["macro_avg"] == 0.75
