"""Offline unit tests for LaTeX-escaped set-brace normalization in reward.py.

Pure string logic — no network, GPU, or sympy required.
Regression for: `normalize_math_answer(r'\\{1,2,3\\}')` used to leave a trailing
backslash (`'1,2,3\\'`) because the outer-brace strip used a greedy capture,
so `math_equal(r'\\{1,2,3\\}', '1,2,3')` wrongly returned False.
"""
from trinity.orchestration.reward import math_equal, normalize_math_answer


def test_escaped_set_braces_stripped_cleanly():
    assert normalize_math_answer(r"\{5\}") == "5"
    assert normalize_math_answer(r"\{1,2,3\}") == "1,2,3"
    assert normalize_math_answer(r"\{a\}") == "a"


def test_unescaped_braces_still_stripped():
    # the already-working case must be unchanged
    assert normalize_math_answer("{5}") == "5"
    assert normalize_math_answer("{1,2,3}") == "1,2,3"


def test_math_equal_across_escaped_and_plain_set():
    assert math_equal(r"\{1,2,3\}", "1,2,3") is True
    assert math_equal(r"\{5\}", "5") is True
    # a genuine mismatch must still be unequal
    assert math_equal(r"\{1,2,3\}", "1,2,4") is False


def test_no_trailing_backslash_left():
    for raw in (r"\{5\}", r"\{1,2,3\}", r"\{x,y\}"):
        assert not normalize_math_answer(raw).endswith("\\")
