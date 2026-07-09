from __future__ import annotations

import benchmarks.livecodebench as LCB
import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R


def test_livecodebench_facade_delegates(monkeypatch):
    seen = {}

    def fake_load_tasks(benchmark, split, max_items, seed):
        seen["args"] = (benchmark, split, max_items, seed)
        return ["ok"]

    monkeypatch.setattr(LCB, "load_tasks", fake_load_tasks)

    out = LCB.load("test", max_items=3, seed=7)

    assert out == ["ok"]
    assert seen["args"] == ("livecodebench", "test", 3, 7)


def test_livecodebench_hf_row_parses_to_task(monkeypatch):
    row = {
        "question_id": "lcb-123",
        "question_content": "Write a program that squares the input integer.",
        "public_test_cases": [
            {"input": "3\n", "output": "9\n"},
            {"input": "5\n", "output": "25\n"},
        ],
        "fn_name": None,
        "starter_code": "",
        "difficulty": "easy",
        "platform": "code_generation_lite",
    }

    seen = {}

    def fake_try_load_hf(path, *, name=None, split=None, version_tag=None):
        seen.setdefault("calls", []).append(
            {"path": path, "name": name, "split": split, "version_tag": version_tag}
        )
        if path != "lighteval/code_generation_lite":
            return None
        return [row]

    monkeypatch.setattr(D, "_try_load_hf", fake_try_load_hf)

    tasks = D.load_tasks("livecodebench", "test", max_items=None, seed=0)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.benchmark == "livecodebench"
    assert task.task_id == "lcb-123"
    assert task.meta["version"] == "release_v6"
    assert task.meta["source"] == "lighteval/code_generation_lite"
    assert task.answer["tests"] == [
        {"input": "3\n", "output": "9\n"},
        {"input": "5\n", "output": "25\n"},
    ]
    assert seen["calls"][0] == {
        "path": "lighteval/code_generation_lite",
        "name": "release_v6",
        "split": "test",
        "version_tag": None,
    }


def test_livecodebench_code_scoring_passes_and_fails():
    reference = {
        "tests": [
            {"input": "3\n", "output": "9\n"},
            {"input": "5\n", "output": "25\n"},
        ],
        "fn_name": None,
        "starter_code": None,
    }

    passing = "```python\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)\n```"
    failing = "```python\nimport sys\nn = int(sys.stdin.read())\nprint(n + n)\n```"

    assert R.score_text("livecodebench", passing, reference) == 1.0
    assert R.score_text("livecodebench", failing, reference) == 0.0
