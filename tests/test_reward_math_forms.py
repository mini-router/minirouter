"""Offline unit tests for extra MATH-500 answer forms in the grader (reward.py).

Covers issue #69's sibling class of false negatives (same family as the
already-fixed thousands-comma bug #35): a correct answer written as a variable
assignment (``x=5``), in scientific notation (``5\\times10^{3}``), or after an
empty ``\\boxed{}`` must grade correct — while wrong values, inequalities, and
empty boxes with no number must still score 0. Pure stdlib (no torch / GPU /
network), matching the existing ``score_text`` test precedent.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


@pytest.mark.parametrize(
    "candidate, reference",
    [
        # Variable assignment: the value is graded, not the restated variable.
        (r"\boxed{x=5}", "5"),
        (r"\boxed{y = 42}", "42"),
        # Scientific notation (\times / \cdot -> canonical exponent).
        (r"\boxed{5\times10^{3}}", "5000"),
        (r"\boxed{1.5\times10^{3}}", "1500"),
        (r"\boxed{5\cdot10^{3}}", "5000"),
        (r"\boxed{5\times10^{-3}}", "0.005"),
        # Empty \boxed{}: fall back to the last number in the text.
        (r"so the answer is \boxed{}. It is 42", "42"),
        (r"\boxed{} 1/2", "0.5"),
    ],
)
def test_extra_answer_forms_grade_correct(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{x=6}", "5"),      # wrong value, assignment stripped
        (r"\boxed{x<=5}", "5"),     # inequality must NOT be graded as its RHS
        (r"\boxed{x>=5}", "5"),
        (r"\boxed{}", "42"),        # empty box, no number anywhere -> wrong
        (r"\boxed{6\times10^{3}}", "5000"),  # wrong magnitude
    ],
)
def test_extra_answer_forms_reject_wrong(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 0.0


def test_normalize_drops_only_leading_assignment():
    assert R.normalize_math_answer("x=5") == "5"
    # Equality assertion and inequalities are left intact (not turned into "=5").
    assert R.normalize_math_answer("x==5") != "5"
    assert R.normalize_math_answer("x<=5") != "5"


def test_normalize_rewrites_scientific_notation():
    # \times becomes * upstream; the exponent form parses numerically.
    assert R._as_number(R.normalize_math_answer(r"5\times10^{3}")) == 5000.0
    assert R._as_number(R.normalize_math_answer(r"5\times10^{-3}")) == 0.005
