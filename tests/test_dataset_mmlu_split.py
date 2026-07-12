"""Offline unit tests for MMLU split-name mapping (dataset.py).

Regression coverage for the bug where `train.py` requests the logical split
"train", which `cais/mmlu` does not have (its training pool is `auxiliary_train`),
causing the loader to silently fall back to the 2-item toy set. These tests are
pure (no `datasets`, no network): they check the mapping helper resolves logical
splits to real MMLU split names.
"""
from __future__ import annotations

import pytest

from trinity.orchestration.dataset import _mmlu_split_for_split

# The only split names cais/mmlu (config "all") actually exposes.
_VALID_MMLU_SPLITS = {"auxiliary_train", "dev", "validation", "test"}


@pytest.mark.parametrize(
    "logical, expected",
    [
        ("train", "auxiliary_train"),        # the bug: "train" must not pass through
        ("auxiliary_train", "auxiliary_train"),
        ("test", "test"),
        ("", "test"),                        # default / unspecified -> graded split
        ("validation", "validation"),
        ("val", "validation"),
        ("dev", "dev"),
        ("TRAIN", "auxiliary_train"),        # case-insensitive
    ],
)
def test_mmlu_split_mapping(logical, expected):
    assert _mmlu_split_for_split(logical) == expected


@pytest.mark.parametrize("logical", ["train", "test", "", "validation", "dev", "auxiliary_train"])
def test_mmlu_split_always_valid(logical):
    # Whatever the caller asks for, we must resolve to a split cais/mmlu has,
    # so the load never fails purely because of a bad split name.
    assert _mmlu_split_for_split(logical) in _VALID_MMLU_SPLITS
