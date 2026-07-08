"""Offline unit tests for transcript post-processing (SPEC §4.5).

Exercises ``trinity.roles.postprocess`` only — no LLM calls, no torch.
"""
import pytest

from trinity.roles.postprocess import ELISION_MARKER, postprocess
from trinity.types import Role


def test_passthrough_under_budget():
    raw = "  hello world  "
    assert postprocess(raw, Role.WORKER, max_chars=100) == "hello world"


def test_none_becomes_empty():
    assert postprocess(None, Role.VERIFIER) == ""


def test_empty_string_stays_empty():
    assert postprocess("", Role.THINKER) == ""


def test_non_positive_max_chars_disables_truncation():
    long = "x" * 20_000
    assert postprocess(long, Role.WORKER, max_chars=0) == long
    assert postprocess(long, Role.WORKER, max_chars=-1) == long


def test_truncation_inserts_elision_marker():
    raw = "A" * 100 + "VERDICT: ACCEPT"
    out = postprocess(raw, Role.VERIFIER, max_chars=40)
    assert ELISION_MARKER in out
    assert len(out) <= 40


def test_truncation_preserves_head_and_tail():
    raw = "HEAD-" + ("m" * 200) + "-TAIL"
    out = postprocess(raw, Role.WORKER, max_chars=60)
    assert out.startswith("HEAD-")
    assert out.endswith("-TAIL")
    assert ELISION_MARKER in out


def test_truncation_when_marker_longer_than_budget():
    raw = "abcdefghijklmnop"
    out = postprocess(raw, Role.WORKER, max_chars=5)
    assert out == raw[:5]
    assert ELISION_MARKER not in out


def test_role_argument_does_not_change_default_policy():
    raw = "same text"
    for role in (Role.THINKER, Role.WORKER, Role.VERIFIER):
        assert postprocess(raw, role, max_chars=100) == "same text"


def test_verifier_verdict_survives_truncation():
    preamble = "The calculation appears correct but consider edge cases. " * 30
    raw = preamble + "VERDICT: ACCEPT"
    out = postprocess(raw, Role.VERIFIER, max_chars=120)
    assert "VERDICT: ACCEPT" in out
    assert len(out) <= 120
