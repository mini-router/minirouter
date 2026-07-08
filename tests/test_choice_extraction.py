"""Offline unit tests for multiple-choice letter extraction in reward.py.

`extract_choice_letter` grades MMLU/GPQA answers into the binary reward. It must
read the model's FINAL choice, matching the "final answer comes last" convention
used by extract_boxed / extract_last_number / extract_code and this function's own
fallback. These tests make NO API calls and need no GPU/network.
"""
from trinity.orchestration.reward import extract_choice_letter, _check_choice


def test_self_correction_returns_last_choice():
    """When a model revises its answer, the last stated choice wins."""
    text = "The answer is A.\nWait, let me reconsider. The answer is C."
    assert extract_choice_letter(text) == "C"
    # and it grades correctly against the true reference
    assert _check_choice(text, "C") is True
    assert _check_choice(text, "A") is False


def test_single_answer_unchanged():
    assert extract_choice_letter("The answer is B.") == "B"
    assert extract_choice_letter("\\boxed{D}") == "D"
    assert extract_choice_letter("Option A") == "A"


def test_pattern_priority_preserved():
    """The explicit 'answer is' pattern still outranks a later bare-letter line;
    within that pattern the last occurrence is taken."""
    # highest-priority pattern matches twice -> take the last (C)
    assert extract_choice_letter("answer is A ... final answer C ... answer is C") == "C"


def test_prose_article_not_matched():
    # regression guard for the existing smoke assertion
    assert extract_choice_letter("A nice approach to think about it") is None


def test_final_answer_on_next_line():
    # regression guard for the existing smoke assertion
    assert extract_choice_letter("Final answer:\nB") == "B"


def test_repeated_boxed_takes_last():
    assert extract_choice_letter("\\boxed{A} then corrected to \\boxed{D}") == "D"
