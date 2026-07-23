"""Offline tests for common MATH-500 answer forms the grader used to miss.

`normalize_math_answer` / `_check_math` previously scored several *correct*
answers as wrong because the model wrote the answer in an equivalent-but-
unnormalized form:

* a single-variable assignment, ``\\boxed{x=5}`` vs reference ``5``;
* scientific notation, ``\\boxed{5\\times10^{3}}`` vs ``5000``;
* an empty ``\\boxed{}`` with the real answer stated in prose.

These are pure-stdlib checks (no torch / GPU / network), following the existing
``score_text`` / ``math_equal`` precedent. They also pin the guards so the fixes
never turn a wrong answer into a match.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


# --------------------------------------------------------------------------- #
# Single-variable assignment: "x = 5" grades against "5"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{x=5}", "5"),
        (r"The answer is \boxed{n = 42}.", "42"),
        (r"\boxed{y=-3}", "-3"),
        (r"\boxed{t = 1/2}", "0.5"),
    ],
)
def test_variable_assignment_grades_on_the_value(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


def test_variable_assignment_still_distinguishes_wrong_values():
    assert R.score_text("math500", r"\boxed{x=5}", "6") == 0.0


def test_inequality_is_not_reduced_to_its_rhs():
    # "x <= 5" must NOT be silently turned into "5" (that would be a wrong grade).
    assert R.score_text("math500", r"\boxed{x \le 5}", "3") == 0.0
    assert R.score_text("math500", r"\boxed{x \le 5}", "5") == 0.0


# --------------------------------------------------------------------------- #
# Scientific notation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{5\times10^{3}}", "5000"),
        (r"\boxed{5 \cdot 10^3}", "5000"),
        (r"\boxed{1.5\times10^{2}}", "150"),
        (r"\boxed{2\times10^{-1}}", "0.2"),
    ],
)
def test_scientific_notation_grades_numerically(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


def test_scientific_notation_distinguishes_wrong_magnitude():
    assert R.score_text("math500", r"\boxed{5\times10^{3}}", "500") == 0.0
    assert R.score_text("math500", r"\boxed{5\times10^{3}}", "50000") == 0.0


def test_e_notation_still_parses():
    assert R.score_text("math500", r"\boxed{5e3}", "5000") == 1.0


# --------------------------------------------------------------------------- #
# Empty \boxed{} falls back to the last number in the text
# --------------------------------------------------------------------------- #
def test_empty_boxed_falls_back_to_last_number():
    assert R.score_text("math500", r"so the answer is \boxed{}. It is 42", "42") == 1.0
    assert R.score_text("math500", r"the result \boxed{ } equals 7", "7") == 1.0


def test_empty_boxed_with_no_number_is_wrong():
    assert R.score_text("math500", r"\boxed{}", "7") == 0.0


# --------------------------------------------------------------------------- #
# Guardrail: pre-existing behavior is unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate, reference, expected",
    [
        (r"\boxed{42}", "42", 1.0),
        (r"\boxed{41}", "42", 0.0),
        ("answer: 1/2", "0.5", 1.0),
        (r"\boxed{1,234}", "1234", 1.0),
        (r"\boxed{-5}", "-5", 1.0),
        (r"\boxed{(1,2)}", "(1,2)", 1.0),
        (r"\boxed{\frac{3}{4}}", "0.75", 1.0),
    ],
)
def test_existing_math_cases_unaffected(candidate, reference, expected):
    assert R.score_text("math500", candidate, reference) == expected


def test_normalize_helpers_directly():
    assert R.normalize_math_answer("x=5") == "5"
    assert R.normalize_math_answer(r"5\times10^{3}") == "5e3"
    # a bare identifier with no assignment is untouched
    assert R.normalize_math_answer("x") == "x"
