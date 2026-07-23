"""Offline unit tests for multiple-choice letter extraction (reward.py).

Regression coverage for the self-correction bug: when a model states one letter
and then corrects itself, the FINAL letter is the answer. Pure stdlib (no torch /
GPU / network), matching the existing ``reward`` test precedent in the suite.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


def test_self_correction_takes_final_letter():
    # The model corrects itself; the last stated letter wins.
    assert R.extract_choice_letter("The answer is A.\nActually, the answer is C.") == "C"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is (B)", "B"),
        ("Answer: C", "C"),
        ("\\boxed{D}", "D"),
        ("Option A", "A"),
        ("Final answer: B", "B"),
        # Higher-priority pattern still beats a later lower-priority match.
        ("The answer is A.\nOption D", "A"),
        # Last occurrence of the SAME (highest) priority pattern wins.
        ("Answer: A. On reflection, Answer: B.", "B"),
    ],
)
def test_extract_choice_letter_phrasings(text, expected):
    assert R.extract_choice_letter(text) == expected


def test_prose_article_a_is_not_a_choice():
    assert R.extract_choice_letter("A nice approach to think about it") is None


def test_empty_returns_none():
    assert R.extract_choice_letter("") is None
