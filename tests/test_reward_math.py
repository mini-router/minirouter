"""Offline unit tests for math-answer normalization / grading (reward.py).

Regression coverage for the thousands-separator comma bug: a correct large-number
answer written with a comma (in the model output OR the reference) must grade
correct. These are pure stdlib (no torch / GPU / network), matching the existing
``score_text`` / ``math_equal`` test precedent in the suite.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"The answer is \boxed{1,234}.", "1234"),   # comma in the model answer
        (r"The answer is \boxed{2500}.", "2,500"),   # comma in the reference
        (r"\boxed{1,234}", "1,234"),                 # comma on both sides
        (r"\boxed{1,234,567}", "1234567"),           # multiple groups
        (r"The total is 12,000.", "12000"),          # last-number fallback path
    ],
)
def test_thousands_separator_comma_grades_correct(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


def test_thousands_comma_still_distinguishes_wrong_answers():
    # The fix must not turn every comma-number into a match.
    assert R.score_text("math500", r"\boxed{1,234}", "1235") == 0.0
    assert R.score_text("math500", r"\boxed{2,000}", "20000") == 0.0


@pytest.mark.parametrize(
    "a, b",
    [
        ("1,234", "1234"),
        ("2,500", "2500"),
        ("1,234,567", "1234567"),
    ],
)
def test_math_equal_ignores_thousands_commas(a, b):
    assert R.math_equal(a, b) is True


def test_non_thousands_comma_is_not_stripped():
    # A comma that does not group exactly three trailing digits is a real
    # separator (e.g. a coordinate/tuple), not a thousands separator.
    assert R.normalize_math_answer("1,23") == "1,23"
    assert R.normalize_math_answer("(1,2)") == "(1,2)"


def test_existing_math_cases_unaffected():
    # Guardrail: the pre-existing behaviors keep working.
    assert R.score_text("math500", r"\boxed{42}", "42") == 1.0
    assert R.score_text("math500", r"\boxed{41}", "42") == 0.0
    assert R.score_text("math500", "answer: 1/2", "0.5") == 1.0
    assert R.math_equal("18.90", r"\$18.90") is True
