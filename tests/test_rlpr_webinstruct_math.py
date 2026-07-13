"""Offline unit tests for RLPR WebInstruct math grading (reward.py).

Regression for: `_check_rlpr_webinstruct` fed the raw candidate text into
`math_equal` without extracting the answer, so a correct `\\boxed{...}` answer
(the format the worker is told to use) scored 0. Pure functions — no
network/GPU/sympy required.
"""
from trinity.orchestration import reward as R

_REF = {"ground_truth": "15", "source": "WebInstruct-verified-val_Avg2"}


def test_boxed_answer_scores_correct():
    assert R.score_text("rlpr", r"The area is \boxed{15}.", _REF) == 1.0


def test_plain_number_still_works():
    assert R.score_text("rlpr", "The answer is 15", _REF) == 1.0


def test_wrong_boxed_answer_scores_zero():
    assert R.score_text("rlpr", r"The area is \boxed{16}.", _REF) == 0.0


def test_letter_answer_path_unchanged():
    ref = {"ground_truth": "B", "source": "WebInstruct-verified-val_Avg2"}
    assert R.score_text("rlpr", "After checking, the answer is B.", ref) == 1.0
    assert R.score_text("rlpr", "The answer is C.", ref) == 0.0


def test_matches_math_path_for_boxed():
    # the same boxed answer graded via the pure math path is correct; RLPR should agree
    assert R.score_text("math500", r"The area is \boxed{15}.", "15") == 1.0
    assert R.score_text("rlpr", r"The area is \boxed{15}.", _REF) == 1.0
