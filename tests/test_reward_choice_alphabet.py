"""Offline tests for the A-J multiple-choice alphabet (#116 / #122).

RLPR routes MMLU-Pro (`MMLUPro-1000_Avg2`) to the choice grader, and MMLU-Pro has
up to ten options, but the grader was hard-capped at A-D: every correct E-J answer
scored 0.0 — a silently-wrong number, not a crash. These tests pin the widened
A-J alphabet and, just as importantly, that the genuinely 4-option benchmarks
(MMLU / GPQA) are unaffected. Pure stdlib: no network / GPU / torch.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


def _rlpr_ref(gold: str) -> dict:
    return {"ground_truth": gold, "source": "MMLUPro-1000_Avg2"}


# --------------------------------------------------------------------------- #
# The bug: E-J were unrepresentable end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gold", list("ABCDEFGHIJ"))
def test_rlpr_mmlu_pro_grades_every_letter(gold):
    assert R.score_text("rlpr", f"The answer is ({gold}).", _rlpr_ref(gold)) == 1.0


@pytest.mark.parametrize("gold", list("EFGHIJ"))
def test_rlpr_mmlu_pro_still_rejects_wrong_letter(gold):
    wrong = "A" if gold != "A" else "B"
    assert R.score_text("rlpr", f"The answer is ({wrong}).", _rlpr_ref(gold)) == 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
def test_extract_choice_letter_spans_a_to_j(letter):
    assert R.extract_choice_letter(f"The answer is {letter}") == letter
    assert R.extract_choice_letter(f"Final answer: ({letter})") == letter
    assert R.extract_choice_letter(f"\\boxed{{{letter}}}") == letter


@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
def test_normalize_reference_letter_spans_a_to_j(letter):
    assert R._normalize_reference_letter(letter) == letter


def test_normalize_reference_letter_integer_index():
    assert R._normalize_reference_letter(0) == "A"
    assert R._normalize_reference_letter(4) == "E"
    assert R._normalize_reference_letter(9) == "J"
    # Out of range stays unresolvable.
    assert R._normalize_reference_letter(10) is None
    assert R._normalize_reference_letter(-1) is None


def test_normalize_reference_letter_rejects_non_choices():
    assert R._normalize_reference_letter("Z") is None
    assert R._normalize_reference_letter(None) is None
    assert R._normalize_reference_letter(True) is None


# --------------------------------------------------------------------------- #
# Guards: the 4-option benchmarks must be unaffected
# --------------------------------------------------------------------------- #
def test_four_option_benchmarks_unchanged():
    assert R.score_text("mmlu", "The answer is (B).", "B") == 1.0
    assert R.score_text("mmlu", "The answer is (C).", "B") == 0.0
    assert R.score_text("gpqa", "Answer: D", "D") == 1.0


def test_stray_high_letter_cannot_match_four_option_gold():
    # Widening only adds the ability to grade E-J; it can never flip a 4-option
    # answer, because the gold there is always A-D.
    assert R.score_text("mmlu", "The answer is (I).", "B") == 0.0
    assert R.score_text("gpqa", "The answer is (J).", "A") == 0.0


def test_prose_article_a_still_not_read_as_choice():
    # The existing false-positive guard must survive the widened alphabet.
    assert R.extract_choice_letter("A nice approach to think about it") is None
