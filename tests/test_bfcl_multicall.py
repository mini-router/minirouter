"""Offline unit tests for BFCL multi-call scoring in reward.py.

Regression for: `_check_bfcl` aligned candidate and gold calls by sorting both on a
shared JSON key, but candidate args are concrete values and gold args are
allowed-value *lists*, so the scalar-vs-list serialization sorts them differently
and the positional `zip` paired up the wrong calls — scoring a correct multi-call
answer 0. The fix matches calls as a bipartite pairing. Pure functions; no
network/GPU.
"""
import json

from trinity.orchestration import reward as R


def _score(cand_calls, gold):
    return R.score_text("bfcl_simple", json.dumps(cand_calls), {"ground_truth": gold})


def test_multicall_correct_answer_scores_one():
    # Two calls to 'f'; call A accepts x in {1,2}, call B accepts x in {1}.
    # Candidate {x=1, x=2} has a valid bijection (x=1->[1], x=2->[1,2]).
    gold = [{"f": {"x": [1, 2]}}, {"f": {"x": [1]}}]
    cand = [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]
    assert _score(cand, gold) == 1.0


def test_multicall_order_independent():
    gold = [{"a": {"p": [1]}}, {"b": {"q": [2]}}]
    cand = [{"name": "b", "arguments": {"q": 2}}, {"name": "a", "arguments": {"p": 1}}]
    assert _score(cand, gold) == 1.0


def test_multicall_wrong_answer_scores_zero():
    # Candidate has no valid pairing: 'b' with q=3 matches neither gold call.
    gold = [{"a": {"p": [1]}}, {"b": {"q": [2]}}]
    cand = [{"name": "a", "arguments": {"p": 1}}, {"name": "b", "arguments": {"q": 3}}]
    assert _score(cand, gold) == 0.0


def test_multicall_no_greedy_false_negative():
    # Greedy pairing could consume the only x=1 candidate for call A (which also
    # accepts 2) and then fail call B (needs 1). A correct matcher backtracks.
    gold = [{"f": {"x": [1, 2]}}, {"f": {"x": [1]}}]
    cand = [{"name": "f", "arguments": {"x": 2}}, {"name": "f", "arguments": {"x": 1}}]
    assert _score(cand, gold) == 1.0


def test_singlecall_unchanged():
    gold = [{"f": {"x": [5]}}]
    assert _score([{"name": "f", "arguments": {"x": 5}}], gold) == 1.0
    assert _score([{"name": "f", "arguments": {"x": 6}}], gold) == 0.0


def test_wrong_call_count_scores_zero():
    gold = [{"a": {"p": [1]}}, {"b": {"q": [2]}}]
    assert _score([{"name": "a", "arguments": {"p": 1}}], gold) == 0.0
