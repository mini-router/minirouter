r"""Offline tests for boxed-answer grading on RLPR WebInstruct math items (#170).

`_check_rlpr_webinstruct` fed the raw candidate prose straight into `math_equal`,
which never strips `\boxed{...}` — so a WebInstruct math item answered in the
requested boxed format (`format_hint("rlpr")` explicitly asks for it) scored 0
even when correct. The fix extracts the answer before the math compare, mirroring
`_check_math`. These tests pin the fix and guard that the letter and plain-text
paths are unchanged. Pure functions; no GPU/network.
"""
from __future__ import annotations

import pytest

import trinity.orchestration.reward as R

_SOURCE = "WebInstruct-verified-val_Avg2"


def _score(candidate: str, gold: object) -> float:
    return R.score_text("rlpr", candidate, {"ground_truth": gold, "source": _SOURCE})


# --------------------------------------------------------------------------- #
# The bug: a correct boxed answer was scored 0
# --------------------------------------------------------------------------- #
def test_boxed_answer_now_grades_correct():
    # Exact repro from the issue: was 0.0 before the fix.
    assert _score(r"The area is \boxed{15}.", "15") == 1.0


def test_wrong_boxed_answer_still_fails():
    assert _score(r"The area is \boxed{16}.", "15") == 0.0


def test_boxed_answer_beats_other_numbers_in_the_prose():
    # extract_boxed wins over a stray earlier number, so the committed \boxed{}
    # value is what gets graded — not the first number mentioned.
    assert _score(r"We ran 3 trials; the final answer is \boxed{15}.", "15") == 1.0


def test_number_in_prose_without_a_box_is_extracted():
    # No box, but a trailing number: extract_last_number recovers it.
    assert _score("The area of the triangle is 15 square units.", "15") == 1.0


def test_boxed_latex_fraction_matches_plain_fraction():
    assert _score(r"Hence \boxed{\frac{1}{2}}.", "1/2") == 1.0


def test_gold_may_itself_be_boxed():
    # Datasets vary: a boxed gold is unwrapped before comparison.
    assert _score(r"\boxed{15}", r"\boxed{15}") == 1.0


# --------------------------------------------------------------------------- #
# Guards: paths that must NOT change
# --------------------------------------------------------------------------- #
def test_plain_number_answer_unchanged():
    # Already worked before the fix (via the "answer is" prefix strip); must stay.
    assert _score("The answer is 15", "15") == 1.0
    assert _score("The answer is 14", "15") == 0.0


def test_free_text_answer_unchanged():
    # No box and no number -> candidate falls back to itself, so non-math text
    # answers grade exactly as before (the `or cand` fallback).
    assert _score("The answer is Paris", "Paris") == 1.0
    assert _score("The answer is London", "Paris") == 0.0


def test_letter_choice_branch_unchanged():
    # A gold letter + a committed letter still routes through the choice compare.
    assert _score("The answer is (A).", "A") == 1.0
    assert _score("The answer is (B).", "A") == 0.0


# --------------------------------------------------------------------------- #
# Direct unit calls on the checker
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate,gold,expected",
    [
        (r"\boxed{15}", "15", True),
        (r"\boxed{15}", "16", False),
        ("42", "42", True),
        ("41", "42", False),
        ("Paris", "Paris", True),
    ],
)
def test_check_rlpr_webinstruct_direct(candidate, gold, expected):
    assert R._check_rlpr_webinstruct(candidate, gold) is expected


def test_empty_candidate_or_gold_is_false():
    assert R._check_rlpr_webinstruct("", "15") is False
    assert R._check_rlpr_webinstruct(r"\boxed{15}", "") is False
    assert R._check_rlpr_webinstruct(r"\boxed{15}", None) is False
