"""Offline synthetic unit tests for the per-question agreement diagnostic (issue #60).

These tests exercise ONLY the pure analysis functions in
``scripts/agreement_diagnostic.py``. They make NO live API calls and need no GPU or
network — they lock the per-question semantics (contested mask, unique solves, pairwise
agreement, perfect-router headroom) and the ``--write-contested`` subset export.

The three synthetic regimes mirror ``tests/test_oracle_ceiling.py`` so the two diagnostics
agree on the same fixtures:
  (a) 3 disjoint specialists -> every query contested, each model owns 1/3 unique solves,
      pairwise agreement 1/3, perfect-router headroom 2/3.
  (b) 3 identical models     -> nothing contested, zero unique solves, agreement 1.0.
  (c) mixed matrix           -> contested_task_ids / write_contested keep only the
      disagreement rows and preserve the on-disk schema.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Both diagnostics live under scripts/, not the importable package; load by file path.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load oracle_ceiling first so agreement_diagnostic reuses that exact copy for the shared
# matrix_to_tensor / p_hat helpers (and for the _tensor_to_matrix fixture builder).
oc = _load("oracle_ceiling")
ad = _load("agreement_diagnostic")


# --------------------------------------------------------------------------- #
# (a) 3 disjoint specialists: each query solved by exactly one (different) model.
# --------------------------------------------------------------------------- #
def _disjoint_specialists(Q=30, K=5):
    S = np.zeros((Q, 3, K))
    for q in range(Q):
        S[q, q % 3, :] = 1.0
    return S


def test_a_disjoint_all_contested():
    S = _disjoint_specialists()
    assert int(ad.contested_mask(S).sum()) == 30


def test_a_disjoint_unique_solves_split_evenly():
    S = _disjoint_specialists()
    assert ad.per_model_unique_solves(S) == [10, 10, 10]


def test_a_disjoint_pairwise_agreement_one_third():
    # Two disjoint specialists agree only on the third they both get wrong.
    S = _disjoint_specialists()
    A = ad.pairwise_agreement(S)
    assert A[0, 1] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert A[1, 0] == pytest.approx(1.0 / 3.0, abs=1e-9)  # symmetric
    assert A[0, 0] == pytest.approx(1.0, abs=1e-9)        # 1.0 diagonal


def test_a_disjoint_perfect_router_headroom():
    S = _disjoint_specialists()
    prs = ad.perfect_router_stats(S)
    assert prs["perfect_router"] == pytest.approx(1.0, abs=1e-9)
    assert prs["best_single"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert prs["headroom"] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_a_disjoint_matches_oracle_disagreement_rate():
    # "contested" here must mean exactly "disagreement" in oracle_ceiling.
    S = _disjoint_specialists()
    rate = ad.contested_mask(S).mean()
    assert rate == pytest.approx(oc.disagreement_rate(S), abs=1e-9)


# --------------------------------------------------------------------------- #
# (b) 3 identical models: verdicts identical -> no contest, no specialization.
# --------------------------------------------------------------------------- #
def _identical_models(Q=40, K=5, seed=1):
    rng = np.random.default_rng(seed)
    base = (rng.random((Q, 1, K)) < 0.6).astype(float)
    return np.repeat(base, 3, axis=1)


def test_b_identical_nothing_contested():
    S = _identical_models()
    assert int(ad.contested_mask(S).sum()) == 0


def test_b_identical_zero_unique_solves():
    S = _identical_models()
    assert ad.per_model_unique_solves(S) == [0, 0, 0]


def test_b_identical_full_pairwise_agreement():
    S = _identical_models()
    A = ad.pairwise_agreement(S)
    assert np.allclose(A, 1.0)


def test_b_identical_zero_headroom():
    S = _identical_models()
    assert ad.perfect_router_stats(S)["headroom"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# (c) matrix I/O: contested_task_ids + subset/write keep only disagreement rows.
# --------------------------------------------------------------------------- #
def _mixed_matrix():
    return {
        "benchmark": "mix", "k": 1, "level": "L0",
        "tasks": [
            {"id": "all_solve", "per_model": {"a": [1], "b": [1], "c": [1]}},
            {"id": "all_fail", "per_model": {"a": [0], "b": [0], "c": [0]}},
            {"id": "contest1", "per_model": {"a": [1], "b": [0], "c": [0]}},
            {"id": "contest2", "per_model": {"a": [0], "b": [1], "c": [1]}},
        ],
    }


def test_c_contested_task_ids_selects_only_contested():
    ids = ad.contested_task_ids(_mixed_matrix())
    assert ids == ["contest1", "contest2"]


def test_c_subset_matrix_preserves_schema_and_provenance():
    m = _mixed_matrix()
    sub = ad.subset_matrix(m, ad.contested_task_ids(m))
    assert len(sub["tasks"]) == 2
    assert sub["benchmark"] == "mix" and sub["k"] == 1
    assert sub["subset"] == "contested" and sub["subset_of"] == "mix"
    assert sub["n_tasks"] == 2
    # The subset must round-trip through the shared reader used by oracle_ceiling.
    S_sub, qids_sub, models = ad.matrix_to_tensor(sub)
    assert S_sub.shape[0] == 2
    assert qids_sub == ["contest1", "contest2"]
    assert models == ["a", "b", "c"]


def test_c_write_contested_roundtrips_to_disk(tmp_path):
    m = _mixed_matrix()
    out = tmp_path / "contested_mix.json"
    n = ad.write_contested(m, out)
    assert n == 2
    written = json.loads(out.read_text())
    assert [t["id"] for t in written["tasks"]] == ["contest1", "contest2"]


# --------------------------------------------------------------------------- #
# Report assembly + degenerate inputs.
# --------------------------------------------------------------------------- #
def test_analyze_matrix_report_shape():
    S = _disjoint_specialists()
    report = ad.analyze_matrix(oc._tensor_to_matrix(S, "disjoint"))
    assert report["n_queries"] == 30
    assert report["contested"]["n_contested"] == 30
    assert report["contested"]["contested_rate"] == pytest.approx(1.0, abs=1e-9)
    assert report["perfect_router"]["headroom"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    # Every model appears in the per-model views.
    assert set(report["per_model_accuracy"]) == {"m0", "m1", "m2"}
    assert set(report["per_model_unique_solves"].values()) == {10}


def test_empty_matrix_is_safe():
    empty = {"benchmark": "empty", "tasks": []}
    report = ad.analyze_matrix(empty)
    assert report["n_queries"] == 0
    assert report["contested"]["n_contested"] == 0
    assert report["contested"]["contested_rate"] == 0.0
    assert ad.contested_task_ids(empty) == []


def test_selftest_passes():
    assert ad._selftest() == 0
