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


# ---------------------------------------------------------------------------
# Multiple choice: echoed option list must not win (issue #124)
# ---------------------------------------------------------------------------
def test_choice_echoed_options_then_committed_letter():
    # The model restates the options (A) .. D)) then commits to B on the last
    # line. The committed answer must win, not the first listed option "A".
    text = "A) Paris\nB) London\nC) Berlin\nD) Rome\n\nB"
    assert R.extract_choice_letter(text) == "B"


def test_choice_echoed_options_then_parenthesized_letter():
    text = "A) w\nB) x\nC) y\nD) z\n\n(C)."
    assert R.extract_choice_letter(text) == "C"


def test_choice_leading_letter_on_final_line():
    # A committed answer line that leads with the letter and trailing text.
    assert R.extract_choice_letter("A) Paris\nB) London\n\nB) London is correct.") == "B"


def test_choice_score_ignores_first_echoed_option():
    # Full grading path: gold is B, options are echoed, answer committed last.
    assert R.score_text("mmlu", "A) Paris\nB) London\nC) Berlin\nD) Rome\n\nB", "B") == 1.0
    # And an echoed-first "A" must not produce a false positive when gold is A.
    assert R.score_text("mmlu", "A) Paris\nB) London\n\nB", "A") == 0.0


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
