"""Offline tests for bare-int access indices in the fugu workflow parser (#225).

`_normalize_access` accepted a scalar string digit ("0" -> [0]) and lists of
indices, but a bare scalar int (0) fell through every branch to `None`, which
rejects the *whole* workflow — so a semantically identical proposal scored
`parsed_ok=False` / `training_reward=0.0` purely on notation. These tests pin
that the three equivalent forms agree, and that the rejections that matter
(bool, negative, forward reference, self reference) still hold. No GPU/network.
"""
from __future__ import annotations

import pytest

from trinity.fugu.workflow import _normalize_access, parse_workflow

_BASE = 'model_id=[0,1]\nsubtasks=["solve","answer"]\naccess_list=[[], {}]'


# --------------------------------------------------------------------------- #
# The bug: bare int was rejected end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form", ["0", '"0"', "[0]"])
def test_equivalent_access_forms_all_parse(form):
    _, parsed_ok = parse_workflow(_BASE.format(form), 3)
    assert parsed_ok is True


def test_equivalent_access_forms_produce_identical_workflow():
    bare, _ = parse_workflow(_BASE.format("0"), 3)
    string, _ = parse_workflow(_BASE.format('"0"'), 3)
    listed, _ = parse_workflow(_BASE.format("[0]"), 3)
    assert bare == string == listed


# --------------------------------------------------------------------------- #
# Helper: accepted bare ints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_bare_int_normalizes_like_its_list_form(idx):
    assert _normalize_access(idx, 3) == [idx]
    # ...and matches the already-supported string / list spellings.
    assert _normalize_access(idx, 3) == _normalize_access(str(idx), 3)
    assert _normalize_access(idx, 3) == _normalize_access([idx], 3)


# --------------------------------------------------------------------------- #
# Guards: what must still be rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_not_a_step_index(bad):
    # bool subclasses int, so it must be rejected before the int branch.
    assert _normalize_access(bad, 3) is None


@pytest.mark.parametrize("bad", [-1, -5])
def test_negative_index_rejected(bad):
    assert _normalize_access(bad, 3) is None


@pytest.mark.parametrize("bad", [3, 4, 99])
def test_forward_reference_rejected(bad):
    # An index at/after the current step is an invalid DAG edge.
    assert _normalize_access(bad, 3) is None


def test_self_reference_at_first_step_rejected():
    assert _normalize_access(0, 0) is None


def test_bare_int_forward_reference_rejects_whole_workflow():
    # Step 1 referencing step 5 is still invalid, bare-int notation or not.
    _, parsed_ok = parse_workflow(_BASE.format("5"), 3)
    assert parsed_ok is False


# --------------------------------------------------------------------------- #
# Unchanged behaviour
# --------------------------------------------------------------------------- #
def test_existing_forms_unchanged():
    assert _normalize_access(None, 3) == []
    assert _normalize_access("all", 3) == "all"
    assert _normalize_access("", 3) == []
    assert _normalize_access("query", 3) == []
    assert _normalize_access([0, 1], 3) == [0, 1]
    assert _normalize_access("not-an-index", 3) is None
