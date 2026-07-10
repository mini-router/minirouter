"""Offline unit tests for HumanEval grading (reward.py).

HumanEval reuses the existing sandboxed pass@1 code executor (CODE_BENCHMARKS):
the model's completed function is executed alongside the dataset's
``check(candidate)`` harness. No network/GPU; each test spawns a short-lived
subprocess (same mechanism already covered by the LiveCodeBench reward tests).
"""
from __future__ import annotations

from trinity.orchestration import reward as R


def test_humaneval_is_registered_as_a_code_benchmark():
    assert "humaneval" in R.CODE_BENCHMARKS


def test_humaneval_passing_completion_scores_correct():
    reference = {
        "tests": [
            "def check(candidate):\n"
            "    assert candidate(2, 3) == 5\n"
            "    assert candidate(-1, 1) == 0\n"
            "check(add)\n"
        ],
        "fn_name": "add",
        "starter_code": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    }
    candidate = "```python\ndef add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n```"

    assert R.score_text("humaneval", candidate, reference) == 1.0


def test_humaneval_failing_completion_scores_incorrect():
    reference = {
        "tests": [
            "def check(candidate):\n"
            "    assert candidate(2, 3) == 5\n"
            "check(add)\n"
        ],
        "fn_name": "add",
        "starter_code": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    }
    candidate = "```python\ndef add(a, b):\n    return a - b\n```"

    assert R.score_text("humaneval", candidate, reference) == 0.0


def test_humaneval_has_answer_predicate():
    assert R.has_answer("humaneval", "```python\ndef f(): pass\n```") is True
    assert R.has_answer("humaneval", "I don't know") is False
