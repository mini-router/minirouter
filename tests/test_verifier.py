"""Tests for verifier verdict parsing (issue #9).

The verdict token must match only as a whole word: a longer word that merely starts
with ACCEPT/REVISE (e.g. "ACCEPTABLE") must NOT be read as a committed verdict, or
the coordinator terminates early on an answer the Verifier meant to reject.
"""
from __future__ import annotations

import pytest

from trinity.roles.verifier import extract_diagnosis, parse_verdict


@pytest.mark.parametrize(
    "text,expected",
    [
        ("VERDICT: ACCEPT", "ACCEPT"),
        ("VERDICT: ACCEPT.", "ACCEPT"),
        ("VERDICT: ACCEPT\n", "ACCEPT"),
        ("some reasoning\nVERDICT: REVISE", "REVISE"),
        ("verdict: accept", "ACCEPT"),  # case-insensitive
        ("VERDICT:ACCEPT", "ACCEPT"),  # no space
    ],
)
def test_valid_verdicts_still_parse(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "VERDICT: ACCEPTABLE only once the null-pointer bug is fixed",
        "VERDICT: ACCEPTED with reservations",
        "VERDICT: ACCEPTS the premise but not the answer",
        "VERDICT: REVISED draft attached",  # 'REVISE' prefix inside 'REVISED'
    ],
)
def test_prefix_words_do_not_count_as_a_verdict(text):
    # Prefix-only matches must not be read as a committed ACCEPT/REVISE.
    assert parse_verdict(text) is None


def test_last_verdict_wins():
    text = "I first thought VERDICT: REVISE\nbut on reflection VERDICT: ACCEPT"
    assert parse_verdict(text) == "ACCEPT"


def test_acceptable_does_not_shadow_a_real_later_verdict():
    text = "This is ACCEPTABLE in isolation.\nFinal call: VERDICT: REVISE"
    assert parse_verdict(text) == "REVISE"


def test_no_verdict_returns_none():
    assert parse_verdict("I have no opinion.") is None
    assert parse_verdict("") is None


def test_diagnosis_not_truncated_by_in_word_false_match():
    # "ACCEPTABLE" earlier in the text must not be treated as the verdict boundary,
    # so the diagnosis keeps everything up to the *real* verdict line.
    text = "The proof is ACCEPTABLE but incomplete.\nVERDICT: REVISE"
    diag = extract_diagnosis(text)
    assert "ACCEPTABLE but incomplete" in diag
    assert "VERDICT" not in diag
