"""Offline unit tests for BBH grading (reward.py).

BBH targets come in two shapes -- a "(X)" multiple-choice letter or free-form
text -- graded by extract_bbh_answer + _check_bbh. Pure stdlib, no network/GPU.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


def test_bbh_is_registered_as_a_text_benchmark():
    assert "bbh" in R.TEXT_BENCHMARKS


def test_bbh_extracts_trailing_answer_line():
    text = "Let's reason step by step.\nThe pass was thrown.\nAnswer: (A)"
    assert R.extract_bbh_answer(text) == "(A)"


def test_bbh_extract_prefers_last_answer_line():
    text = "Answer: (B)\nWait, reconsidering...\nAnswer: (A)"
    assert R.extract_bbh_answer(text) == "(A)"


def test_bbh_extract_falls_back_to_last_line():
    assert R.extract_bbh_answer("no marker here\njust the final line") == "just the final line"


def test_bbh_extract_empty_text_is_none():
    assert R.extract_bbh_answer("") is None


def test_bbh_choice_shaped_target_matches_letter():
    assert R.score_text("bbh", "Reasoning...\nAnswer: (A)", "(A)") == 1.0
    assert R.score_text("bbh", "Reasoning...\nAnswer: (B)", "(A)") == 0.0


def test_bbh_choice_shaped_target_tolerates_option_text_alongside_letter():
    assert R.score_text("bbh", "Reasoning...\nAnswer: (A) plausible", "(A)") == 1.0


def test_bbh_choice_shaped_target_accepts_bare_letter():
    assert R.score_text("bbh", "Reasoning...\nAnswer: A", "(A)") == 1.0


def test_bbh_free_form_target_normalizes_case_and_punctuation():
    assert R.score_text("bbh", "Reasoning...\nAnswer: True.", "True") == 1.0
    assert R.score_text("bbh", "Reasoning...\nAnswer: \"apple banana cherry\"", "apple banana cherry") == 1.0
    assert R.score_text("bbh", "Reasoning...\nAnswer: false", "True") == 0.0


def test_bbh_has_answer_predicate():
    assert R.has_answer("bbh", "Answer: Yes") is True
    assert R.has_answer("bbh", "") is False
