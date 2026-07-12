from __future__ import annotations

import benchmarks.rlpr as RLPR
import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R


def test_rlpr_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(RLPR, "load_tasks", fake_load_tasks)

    out = RLPR.load("test", max_items=3, seed=7)

    assert out == ["ok"]
    assert seen["args"] == ("rlpr", "test", 3, 7)


def test_rlpr_hf_rows_parse_to_tasks(monkeypatch):
    rows_by_file = {
        "Math-500_Avg2.parquet": [
            {
                "data_source": "Math-500_Avg2",
                "prompt": [
                    {
                        "role": "system",
                        "content": "Answer with <think> and <answer> tags.",
                    },
                    {
                        "role": "user",
                        "content": "Convert the point (0,3) in rectangular coordinates to polar coordinates.",
                    },
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": r"\\left( 3, \\frac{\\pi}{2} \\right)",
                    "style": "rule",
                },
                "extra_info": {"split": "test"},
                "uid": "Math-500-0",
            }
        ],
        "MMLUPro-1000_Avg2.parquet": [
            {
                "data_source": "MMLUPro-1000_Avg2",
                "prompt": [
                    {
                        "role": "system",
                        "content": "Answer with a single choice letter.",
                    },
                    {
                        "role": "user",
                        "content": "Body temperature on the average is 98.6°F. What is this on the Celsius scale and the Kelvin scale?",
                    },
                ],
                "ability": "mmlu_pro",
                "reward_model": {"ground_truth": "A", "style": "rule"},
                "extra_info": {"split": None},
                "uid": "MMLUPro-1000-0",
            }
        ],
    }

    def fake_try_load_parquet(url):
        for filename, rows in rows_by_file.items():
            if url.endswith(filename):
                return rows
        return []

    monkeypatch.setattr(D, "_try_load_parquet", fake_try_load_parquet)

    tasks = D.load_tasks("rlpr", "test", max_items=None, seed=0)

    assert len(tasks) == 2
    assert {task.task_id for task in tasks} == {"Math-500-0", "MMLUPro-1000-0"}
    math_task = next(task for task in tasks if task.task_id == "Math-500-0")
    choice_task = next(task for task in tasks if task.task_id == "MMLUPro-1000-0")
    assert math_task.benchmark == "rlpr"
    assert choice_task.benchmark == "rlpr"
    assert math_task.meta["kind"] == "math"
    assert choice_task.meta["kind"] == "choice"
    assert "SYSTEM:" in math_task.prompt and "USER:" in math_task.prompt
    assert math_task.answer["ground_truth"] == r"\\left( 3, \\frac{\\pi}{2} \\right)"
    assert choice_task.answer["ground_truth"] == "A"


def test_rlpr_reward_scores_math_and_choice_sources():
    math_reference = {
        "ground_truth": r"\\left( 3, \\frac{\\pi}{2} \\right)",
        "source": "Math-500_Avg2",
    }
    choice_reference = {
        "ground_truth": "A",
        "source": "MMLUPro-1000_Avg2",
    }

    assert (
        R.score_text(
            "rlpr",
            r"The answer is \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.",
            math_reference,
        )
        == 1.0
    )
    assert R.score_text("rlpr", "The answer is (A).", choice_reference) == 1.0
    assert R.score_text("rlpr", "The answer is (B).", choice_reference) == 0.0


def test_rlpr_webinstruct_uses_generic_matching():
    reference = {
        "ground_truth": "42",
        "source": "WebInstruct-verified-val_Avg2",
    }

    assert R.score_text("rlpr", "42", reference) == 1.0
    assert R.score_text("rlpr", "41", reference) == 0.0


def test_rlpr_train_split_is_blocked():
    try:
        D.load_tasks("rlpr", "train", max_items=None, seed=0)
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("expected rlpr train split to be rejected")


def test_rlpr_load_failure_is_loud(monkeypatch):
    monkeypatch.setattr(D, "_try_load_parquet", lambda url: None)

    try:
        D.load_tasks("rlpr", "test", max_items=None, seed=0)
    except RuntimeError as exc:
        assert "failed to load RLPR parquet file" in str(exc)
    else:
        raise AssertionError("expected rlpr load failure to raise")
