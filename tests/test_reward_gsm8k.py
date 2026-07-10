"""Offline unit tests for GSM8K grading (reward.py).

GSM8K reuses the existing math grading path (MATH_BENCHMARKS): boxed/last-number
extraction plus math_equal comparison. These tests just confirm gsm8k is wired
into that dispatch and behaves the same as math500.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


def test_gsm8k_is_registered_as_a_math_benchmark():
    assert "gsm8k" in R.MATH_BENCHMARKS


def test_gsm8k_boxed_answer_scores_correct():
    assert R.score_text("gsm8k", r"Step by step... \boxed{18}", "18") == 1.0
    assert R.score_text("gsm8k", r"Step by step... \boxed{17}", "18") == 0.0


def test_gsm8k_last_number_fallback():
    assert R.score_text("gsm8k", "After the math, the total is 42.", "42") == 1.0


def test_gsm8k_has_answer_predicate():
    assert R.has_answer("gsm8k", r"\boxed{5}") is True
    assert R.has_answer("gsm8k", "no answer here") is False
