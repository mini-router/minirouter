from __future__ import annotations

import benchmarks.ifeval as IFEVAL
import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R


def test_ifeval_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(IFEVAL, "load_tasks", fake_load_tasks)

    out = IFEVAL.load("test", max_items=3, seed=7)

    assert out == ["ok"]
    assert seen["args"] == ("ifeval", "test", 3, 7)


def test_ifeval_hf_row_parses_to_task(monkeypatch):
    rows = [
        {
            "key": 1000,
            "prompt": "Write exactly two paragraphs. Do not use commas.",
            "instruction_id_list": [
                "length_constraints:number_paragraphs",
                "punctuation:no_comma",
            ],
            "kwargs": [{"num_paragraphs": 2}, {}],
        }
    ]

    monkeypatch.setattr(D, "_fetch_jsonl_rows", lambda url: rows)

    tasks = D.load_tasks("ifeval", "test", max_items=None, seed=0)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.benchmark == "ifeval"
    assert task.task_id == "1000"
    assert task.answer["instruction_id_list"] == rows[0]["instruction_id_list"]
    assert task.meta["source"] == "google-research/google-research"


def test_ifeval_reward_scores_common_constraints():
    reference = {
        "instruction_id_list": [
            "punctuation:no_comma",
            "startend:quotation",
        ],
        "kwargs": [{}, {}],
    }
    candidate = '"HELLO WORLD"'
    wrong = '"HELLO, WORLD"'

    assert R.score_text("ifeval", candidate, reference) == 1.0
    assert R.score_text("ifeval", wrong, reference) == 0.0


def test_ifeval_language_instruction_fails_closed():
    assert R._ifeval_detect_language("Hello world", "kn") is False
    assert R._ifeval_detect_language("Hello world", "xx") is False
