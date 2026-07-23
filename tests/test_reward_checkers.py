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


# ---------------------------------------------------------------------------
# Committed-answer selection across a multi-turn trajectory
# ---------------------------------------------------------------------------
from trinity.orchestration.session import _final_answer  # noqa: E402
from trinity.types import Role, Task, Trajectory, TurnRecord  # noqa: E402


def _turn(role: Role, out: str) -> TurnRecord:
    return TurnRecord(turn=1, agent_name="m", role=role, raw_output=out, processed_output=out)


def _traj(turns: list[TurnRecord], *, answer: str = "42", benchmark: str = "math500") -> Trajectory:
    traj = Trajectory(task=Task("t", benchmark, "q", answer), turns=turns)
    traj.final_answer = _final_answer(traj)  # mirror the real pipeline
    return traj


def test_committed_answer_skips_verifier_that_quotes_gold():
    # The Worker produced no answer; the Verifier quoted the gold value while
    # REVISE-ing. The trajectory was rejected and must not score as correct.
    traj = _traj(
        [
            _turn(Role.WORKER, "I am not sure how to finish this."),
            _turn(Role.VERIFIER, "The correct value is 42, but the worker never computed it.\nVERDICT: REVISE"),
        ]
    )
    assert R._committed_answer("math500", traj) == "I am not sure how to finish this."
    assert R.score(traj) == 0.0


def test_committed_answer_still_recovers_earlier_nonverifier_answer():
    # Legitimate recovery is preserved: when the final (last-Worker) output has no
    # parseable answer, an earlier NON-verifier turn's answer is still credited.
    traj = _traj(
        [
            _turn(Role.THINKER, r"Working it through, the result is \boxed{42}."),
            _turn(Role.WORKER, "On reflection I am not certain, let me not commit a value."),
        ]
    )
    assert R.score(traj) == 1.0


def test_committed_answer_uses_worker_answer_with_verifier_accept():
    # Control: a real Worker answer followed by an ACCEPT is scored from the Worker.
    traj = _traj(
        [
            _turn(Role.WORKER, r"The answer is \boxed{42}."),
            _turn(Role.VERIFIER, "Looks right.\nVERDICT: ACCEPT"),
        ]
    )
    assert R.score(traj) == 1.0
