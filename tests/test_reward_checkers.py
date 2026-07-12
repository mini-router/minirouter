"""Offline unit tests for per-task reward checkers (SPEC smoke test S5).

``orchestration/reward.py`` is the single source of truth for the binary reward
that drives sep-CMA-ES training and eval. The smoke ladder exercises S5 in
``tests/smoke/run_smoke.py``, but there was no dedicated pytest module locking
math, multiple-choice, and code checkers offline.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


# ---------------------------------------------------------------------------
# Math (math500 / aime)
# ---------------------------------------------------------------------------
def test_math_boxed_correct_and_wrong():
    assert R.score_text("math500", r"Thus \boxed{42}.", "42") == 1.0
    assert R.score_text("math500", r"Thus \boxed{41}.", "42") == 0.0


def test_math_fraction_equivalence():
    assert R.score_text("math500", "answer: 1/2", "0.5") == 1.0


def test_extract_boxed_nested_braces():
    text = r"Final: \boxed{\frac{1}{2}}"
    assert R.extract_boxed(text) == r"\frac{1}{2}"


def test_extract_last_number_ignores_thousands_commas():
    assert R.extract_last_number("The value is 1,234.") == "1234"


def test_has_answer_math_detects_boxed_or_number():
    assert R.has_answer("math500", r"\boxed{7}") is True
    assert R.has_answer("math500", "no numbers here") is False


# ---------------------------------------------------------------------------
# Multiple choice (mmlu / gpqa)
# ---------------------------------------------------------------------------
def test_choice_letter_grading():
    assert R.score_text("mmlu", "The answer is (C).", "C") == 1.0
    assert R.score_text("mmlu", "The answer is (C).", "B") == 0.0


def test_choice_prose_a_is_not_a_choice():
    assert R.extract_choice_letter("A nice approach to think about it") is None


def test_choice_final_line_fallback():
    assert R.extract_choice_letter("Final answer:\nB") == "B"


# MMLU-Pro has up to ten options (A-J); RLPR routes it to the same choice
# grader, so E-J must be extractable and matchable, not silently scored 0.
def test_choice_letters_span_a_through_j():
    for letter in "ABCDEFGHIJ":
        assert R.extract_choice_letter(f"The answer is ({letter}).") == letter
        assert R._normalize_reference_letter(letter) == letter


def test_choice_letters_stop_at_j():
    # K is out of range for a ten-option scheme.
    assert R.extract_choice_letter("The answer is (K).") is None
    assert R._normalize_reference_letter("K") is None


def test_choice_reference_integer_index_covers_ten_options():
    assert R._normalize_reference_letter(5) == "F"   # 0-based
    assert R._normalize_reference_letter(9) == "J"
    assert R._normalize_reference_letter(10) is None


def test_rlpr_mmlu_pro_grades_e_through_j_correctly():
    """The regression: MMLU-Pro golds E-J scored 0 even when answered right."""
    for gold in "EFGHIJ":
        ref = {"ground_truth": gold, "source": "MMLUPro-1000_Avg2"}
        assert R.score_text("rlpr", f"The answer is ({gold}).", ref) == 1.0
        # a wrong letter still scores 0 (no blanket pass)
        wrong = "A" if gold != "A" else "B"
        assert R.score_text("rlpr", f"The answer is ({wrong}).", ref) == 0.0


# ---------------------------------------------------------------------------
# Code (livecodebench stdin/stdout)
# ---------------------------------------------------------------------------
def test_code_pass_at_1_honors_input_output_keys():
    code_ok = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    tests = [{"input": "5\n", "output": "25"}, {"input": "3\n", "output": "9"}]
    assert R.run_pass_at_1(code_ok, tests, timeout_s=10) is True


def test_code_pass_at_1_rejects_wrong_answer():
    code_bad = "import sys\nn=int(sys.stdin.read())\nprint(n+1)"
    tests = [{"input": "5\n", "output": "25"}]
    assert R.run_pass_at_1(code_bad, tests, timeout_s=10) is False


def test_code_empty_tests_fail_closed():
    assert R.run_pass_at_1("print(1)", [], timeout_s=5) is False


def test_extract_code_returns_last_fenced_block():
    text = "```python\nold = 1\n```\nSome text\n```python\nnew = 2\n```"
    assert "new = 2" in R.extract_code(text)
