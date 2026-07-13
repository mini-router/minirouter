from __future__ import annotations

import benchmarks.bfcl_simple as BFCL
import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R


def test_bfcl_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(BFCL, "load_tasks", fake_load_tasks)

    out = BFCL.load("test", max_items=3, seed=7)

    assert out == ["ok"]
    assert seen["args"] == ("bfcl_simple", "test", 3, 7)


def test_bfcl_hf_row_parses_to_task(monkeypatch):
    question_rows = [
        {
            "id": "simple_python_0",
            "question": [[{"role": "user", "content": "Find the area of a triangle."}]],
            "function": [
                {
                    "name": "calculate_triangle_area",
                    "description": "Calculate a triangle area.",
                }
            ],
        }
    ]
    answer_rows = [
        {
            "id": "simple_python_0",
            "ground_truth": [
                {
                    "calculate_triangle_area": {
                        "base": [10],
                        "height": [5],
                        "unit": ["units", ""],
                    }
                }
            ],
        }
    ]

    monkeypatch.setattr(D, "_bfcl_categories_for_split", lambda split: ["BFCL_v4_simple_python.json"])

    def fake_fetch_jsonl_rows(url):
        return answer_rows if "possible_answer" in url else question_rows

    monkeypatch.setattr(D, "_fetch_jsonl_rows", fake_fetch_jsonl_rows)

    tasks = D.load_tasks("bfcl_simple", "test", max_items=None, seed=0)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.benchmark == "bfcl_simple"
    assert task.task_id == "simple_python_0"
    assert task.meta["category"] == "simple_python"
    assert task.answer["ground_truth"] == answer_rows[0]["ground_truth"]
    assert "calculate_triangle_area" in task.prompt


def test_bfcl_reward_scores_matching_json_call():
    reference = {
        "ground_truth": [
            {
                "calculate_triangle_area": {
                    "base": [10],
                    "height": [5],
                    "unit": ["units", ""],
                }
            }
        ]
    }
    candidate = (
        '{"name":"calculate_triangle_area",'
        '"arguments":{"base":10,"height":5,"unit":"units"}}'
    )
    wrong = (
        '{"name":"calculate_triangle_area",'
        '"arguments":{"base":10,"height":6,"unit":"units"}}'
    )

    assert R.score_text("bfcl_simple", candidate, reference) == 1.0
    assert R.score_text("bfcl_simple", wrong, reference) == 0.0


def test_bfcl_train_split_is_blocked():
    try:
        D.load_tasks("bfcl_simple", "train", max_items=None, seed=0)
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("expected bfcl_simple train split to be rejected")
