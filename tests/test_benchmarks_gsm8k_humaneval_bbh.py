"""Offline tests for the GSM8K / HumanEval / BBH benchmarks (#99).

Covers the three new loaders (toy-fallback determinism + HF-row parsing via a
monkeypatched `_try_load_hf`) and their graders (pure `score_text` cases):
GSM8K on the math path, HumanEval on the sandboxed assert path, and BBH on the
mixed multiple-choice / normalized-exact-match path. No network / GPU / torch —
the code cases execute in the same isolated subprocess sandbox the grader uses.
"""
from __future__ import annotations

from pathlib import Path

import yaml

import trinity.orchestration.dataset as D
import trinity.orchestration.reward as R

_REPO = Path(__file__).resolve().parents[1]


def _fence(code: str) -> str:
    return "```python\n" + code + "\n```"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_benchmarks_registered():
    for b in ("gsm8k", "humaneval", "bbh"):
        assert b in D.SUPPORTED_BENCHMARKS
        assert b in D._HF_LOADERS
    assert "gsm8k" in R.MATH_BENCHMARKS
    assert "humaneval" in R.CODE_BENCHMARKS
    assert "bbh" in R.BBH_BENCHMARKS


# --------------------------------------------------------------------------- #
# Toy fallbacks load offline and deterministically
# --------------------------------------------------------------------------- #
def test_toy_fallbacks_load_offline():
    for b in ("gsm8k", "humaneval", "bbh"):
        tasks = D.load_tasks(b, "test", max_items=None, seed=0)
        assert tasks and all(t.benchmark == b for t in tasks)
        again = D.load_tasks(b, "test", max_items=None, seed=0)
        assert [t.task_id for t in tasks] == [t.task_id for t in again]


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
def test_gsm8k_final_answer_extraction():
    assert D._gsm8k_final_answer("Steps...\n#### 1,234") == "1234"
    assert D._gsm8k_final_answer("no marker 42") == "no marker 42"
    assert D._gsm8k_final_answer("") is None


def test_gsm8k_hf_row_parses_and_grades(monkeypatch):
    row = {"question": "2+3?", "answer": "Add them.\n#### 5"}
    monkeypatch.setattr(
        D, "_try_load_hf", lambda path, **kw: [row] if path == "openai/gsm8k" else None
    )
    tasks = D.load_tasks("gsm8k", "test", max_items=None, seed=0)
    assert len(tasks) == 1 and tasks[0].answer == "5"
    assert R.score_text("gsm8k", "The answer is 5.", tasks[0].answer) == 1.0
    assert R.score_text("gsm8k", "The answer is 6.", tasks[0].answer) == 0.0


# --------------------------------------------------------------------------- #
# HumanEval
# --------------------------------------------------------------------------- #
def test_humaneval_hf_row_shapes_assert_spec_and_grades(monkeypatch):
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "entry_point": "add",
    }
    monkeypatch.setattr(
        D,
        "_try_load_hf",
        lambda path, **kw: [row] if path == "openai/openai_humaneval" else None,
    )
    tasks = D.load_tasks("humaneval", "test", max_items=None, seed=0)
    assert len(tasks) == 1
    spec = tasks[0].answer
    assert "check(add)" in spec["tests"][0]
    assert R.score_text("humaneval", _fence("def add(a, b):\n    return a + b\n"), spec) == 1.0
    assert R.score_text("humaneval", _fence("def add(a, b):\n    return a - b\n"), spec) == 0.0


def test_humaneval_toy_grades():
    spec = D.load_tasks("humaneval", "test", max_items=None, seed=0)[0].answer
    assert R.score_text("humaneval", _fence("def add(a, b):\n    return a + b\n"), spec) == 1.0
    assert R.score_text("humaneval", _fence("def add(a, b):\n    return 0\n"), spec) == 0.0


# --------------------------------------------------------------------------- #
# BBH
# --------------------------------------------------------------------------- #
def test_bbh_multiple_choice_grading():
    assert R.score_text("bbh", "The answer is (A).", "(A)") == 1.0
    assert R.score_text("bbh", "The answer is (B).", "(A)") == 0.0
    # More than four options: (E)/(G) must not be silently dropped.
    assert R.score_text("bbh", "So the answer is (E).", "(E)") == 1.0
    assert R.score_text("bbh", "Final answer: (G)", "(G)") == 1.0


def test_bbh_freeform_grading():
    assert R.score_text("bbh", "Reasoning...\nThe answer is valid", "valid") == 1.0
    assert R.score_text("bbh", "work\napple banana cherry", "apple banana cherry") == 1.0
    assert R.score_text("bbh", "invalid", "valid") == 0.0
    assert R.score_text("bbh", "", "valid") == 0.0


def test_bbh_hf_row_parses(monkeypatch):
    monkeypatch.setattr(
        D,
        "_try_load_hf",
        lambda path, **kw: [{"input": "Q?", "target": "(A)"}] if kw.get("name") == "boolean_expressions" else None,
    )
    tasks = D.load_tasks("bbh", "test", max_items=None, seed=0)
    assert tasks and tasks[0].benchmark == "bbh" and tasks[0].answer == "(A)"
    assert tasks[0].meta["subtask"] == "boolean_expressions"


# --------------------------------------------------------------------------- #
# Review fixes on #114/#115: has_answer(bbh) + HumanEval eval-only
# --------------------------------------------------------------------------- #
def test_has_answer_recognizes_bbh():
    # has_answer drives _committed_answer + the training format_bonus; BBH must
    # be recognized (both the multiple-choice and free-form shapes).
    assert R.has_answer("bbh", "The answer is (A).") is True
    assert R.has_answer("bbh", "So the answer is (E).") is True
    assert R.has_answer("bbh", "valid") is True
    assert R.has_answer("bbh", "") is False
    assert R.has_answer("bbh", "   ") is False


def test_humaneval_is_eval_only_not_in_train():
    # HumanEval has a single (test) split, so training on it and then evaluating
    # on the same split is not disjoint — it must be eval-only.
    cfg = yaml.safe_load((_REPO / "configs" / "benchmarks.yaml").read_text())
    train_names = [b["name"] for b in cfg["train"]]
    eval_names = [b["name"] for b in cfg["eval"]]
    assert "humaneval" not in train_names
    assert "humaneval" in eval_names
    # GSM8K keeps proper disjoint train/test splits.
    assert "gsm8k" in train_names and "gsm8k" in eval_names
