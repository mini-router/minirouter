"""Offline unit tests for BFCL multi-call grading (issue #180)."""
from __future__ import annotations

import json

from trinity.orchestration import reward as R


def test_bfcl_multi_call_bipartite_match_scores_correct():
    # Gold accepts x in {1,2} for call A and x in {1} for call B.
    # Candidate emits x=1 and x=2 — a valid pairing exists either way under
    # bipartite matching, but sort-and-zip misaligned and scored 0 (#180).
    gold = [{"f": {"x": [1, 2]}}, {"f": {"x": [1]}}]
    cand = [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]
    assert R.score_text("bfcl_simple", json.dumps(cand), {"ground_truth": gold}) == 1.0


def test_bfcl_multi_call_rejects_unmatchable_set():
    gold = [{"f": {"x": [1]}}, {"f": {"x": [1]}}]
    cand = [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]
    assert R.score_text("bfcl_simple", json.dumps(cand), {"ground_truth": gold}) == 0.0


def test_bfcl_single_call_unaffected():
    gold = [{"f": {"x": [1, 2]}}]
    cand = [{"name": "f", "arguments": {"x": 2}}]
    assert R.score_text("bfcl_simple", json.dumps(cand), {"ground_truth": gold}) == 1.0
    cand_bad = [{"name": "f", "arguments": {"x": 3}}]
    assert R.score_text("bfcl_simple", json.dumps(cand_bad), {"ground_truth": gold}) == 0.0
