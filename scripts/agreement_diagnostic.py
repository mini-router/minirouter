#!/usr/bin/env python3
"""Per-question agreement diagnostic + contested-subset export (issue #60).

`scripts/oracle_ceiling.py` answers the *statistical* question — with winner's-curse
debiasing and bootstrap CIs, is there real routing headroom on the current pool? This
module answers the complementary *per-question* question the scalar CI cannot:

  * which specific queries are **contested** (some models solve them, some do not) — the
    only queries where a router can change correctness at all;
  * how each model **specializes** — the queries only it solves (its unique solves);
  * the **pairwise agreement** between every pair of models;
  * the **perfect-router headroom** — how far a clairvoyant per-question picker beats the
    best single model on the raw hard-verdict view.

It reads the SAME on-disk ``oracle_matrix_<bench>.json`` format that
``oracle_ceiling.py --collect`` emits and reuses that module's ``matrix_to_tensor`` /
``p_hat`` helpers, so the two tools can never drift on the schema or on the definition of
the per-(query, model) solve probability. It **complements, and does not duplicate**,
oracle_ceiling.py: that module owns the debiased CI and the pool-vs-router *verdict*; this
one owns the per-question breakdown and the ``--write-contested`` export of the
disagreement subset. The JOURNAL entry 2026-06-25 (dead GRPO gradient on math500)
explicitly recommends training on exactly that contested subset, but there was no tool to
produce it — this is that tool.

Design contract (same as oracle_ceiling.py): every function here is pure and offline — no
network, no torch/GPU, numpy only — so the whole module is unit-testable on the dev box.

Modes:

  --matrix MATRIX_JSON      analyze a matrix and print the per-question report; add
                            --write-contested OUT.json to also export the disagreement
                            subset in matrix schema (ready to re-feed to train/eval).
  --selftest                offline synthetic unit tests of the diagnostic math.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
# scripts/ is not an importable package, so make the sibling diagnostic importable by name
# whether this file is run as a script or loaded via importlib in a test.
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load_oracle_ceiling():
    """Return the sibling ``oracle_ceiling`` module, importing it by path if needed.

    Reuses an already-loaded copy from ``sys.modules`` (e.g. one a test loaded first) so
    there is a single shared definition of ``matrix_to_tensor`` / ``p_hat``.
    """
    cached = sys.modules.get("oracle_ceiling")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "oracle_ceiling", _SCRIPTS / "oracle_ceiling.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracle_ceiling"] = module
    spec.loader.exec_module(module)
    return module


_oc = _load_oracle_ceiling()
matrix_to_tensor = _oc.matrix_to_tensor
p_hat = _oc.p_hat


# =============================================================================
# Per-question agreement core (pure, offline, unit-tested). Operates on the same
# "solves" tensor S of shape (Q, M, K) with S[q, m, k] in {0, 1} that
# oracle_ceiling uses. The per-model *verdict* on a query is the K-sample MAJORITY
# (p_hat[q, m] >= 0.5), the identical convention oracle_ceiling.disagreement_rate
# uses, so "contested" here means exactly "disagreement" there.
# =============================================================================
def verdicts(S: np.ndarray) -> np.ndarray:
    """Per-(query, model) hard verdict as an int array of shape ``(Q, M)``.

    ``verdict[q, m] = 1`` iff model ``m`` solves query ``q`` by majority vote over its K
    samples (``p_hat[q, m] >= 0.5``), else 0. Ties (``p_hat == 0.5``) count as a solve,
    matching ``oracle_ceiling.disagreement_rate``.
    """
    S = np.asarray(S, dtype=float)
    if S.size == 0:
        # Empty (no tasks or K=0): no samples to average, so no solves. Short-circuit
        # before p_hat to avoid numpy's "Mean of empty slice" warning on a 0-size mean.
        return np.zeros(S.shape[:2], dtype=int)
    p = p_hat(S)  # (Q, M); also validates the tensor is 3-D and 0/1.
    return (p >= 0.5).astype(int)


def per_model_accuracy(S: np.ndarray) -> list[float]:
    """Soft per-model accuracy ``mean_q p_hat[q, m]`` — one float per model."""
    S = np.asarray(S, dtype=float)
    if S.size == 0:
        return [0.0] * (S.shape[1] if S.ndim == 3 else 0)
    p = p_hat(S)
    return [float(x) for x in p.mean(axis=0)]


def contested_mask(S: np.ndarray) -> np.ndarray:
    """Boolean ``(Q,)`` mask, True where the models DISAGREE on the hard verdict.

    A query is *contested* when at least one model solves it and at least one does not —
    the only queries where routing can change correctness. All-solve and all-fail queries
    are uncontested (the router is irrelevant on them). Returns all-False when there are
    0 or 1 models (nothing to disagree about).
    """
    v = verdicts(S)
    Q, M = v.shape
    if Q == 0 or M <= 1:
        return np.zeros(Q, dtype=bool)
    return v.min(axis=1) != v.max(axis=1)


def per_model_unique_solves(S: np.ndarray) -> list[int]:
    """Count, per model, of queries that ONLY that model solves (hard verdict).

    A *unique solve* for model ``m`` is a query where ``verdict[q, m] == 1`` and every
    other model scores 0 — the specialization signal, i.e. what the pool would lose if
    model ``m`` were dropped. Returns one int per model.
    """
    v = verdicts(S)
    Q, M = v.shape
    if Q == 0 or M == 0:
        return [0] * M
    solved_by = v.sum(axis=1)          # (Q,) how many models solve each query
    only_one = solved_by == 1          # queries solved by exactly one model
    return [int(np.sum(only_one & (v[:, m] == 1))) for m in range(M)]


def pairwise_agreement(S: np.ndarray) -> np.ndarray:
    """``(M, M)`` agreement matrix on the hard verdict.

    ``A[i, j] = mean_q 1[verdict[q, i] == verdict[q, j]]`` — the fraction of queries on
    which models ``i`` and ``j`` reach the same verdict. Symmetric, with a 1.0 diagonal.
    With no queries it is the identity-agreement matrix of ones.
    """
    v = verdicts(S)
    Q, M = v.shape
    A = np.ones((M, M))
    if Q == 0 or M == 0:
        return A
    for i in range(M):
        for j in range(i, M):
            agree = float(np.mean(v[:, i] == v[:, j]))
            A[i, j] = agree
            A[j, i] = agree
    return A


def perfect_router_stats(S: np.ndarray) -> dict:
    """Raw per-question 'perfect router' accuracy, best-single, and their headroom.

    On the hard verdict a clairvoyant per-question router solves ``q`` iff ANY model does,
    so its accuracy is ``mean_q max_m verdict``. ``best_single`` is the best FIXED model,
    ``max_m mean_q verdict``. ``headroom = perfect - best_single >= 0``.

    NOTE: this is the *optimistic* per-question view with NO winner's-curse debiasing. Its
    job is to size the contested opportunity, not to make a pool-vs-router call — for the
    honest, CI-gated routing ceiling use ``scripts/oracle_ceiling.py``.
    """
    v = verdicts(S)
    Q, M = v.shape
    if Q == 0 or M == 0:
        return {"perfect_router": 0.0, "best_single": 0.0,
                "best_single_model": 0, "headroom": 0.0}
    perfect = float(v.max(axis=1).mean())
    per_model = v.mean(axis=0)
    bm = int(np.argmax(per_model))
    bs = float(per_model[bm])
    return {"perfect_router": perfect, "best_single": bs,
            "best_single_model": bm, "headroom": perfect - bs}


# =============================================================================
# Matrix-level helpers (operate on the on-disk oracle_matrix_<bench>.json dict).
# =============================================================================
def _task_id(task: dict, fallback) -> str:
    """The stable id of a matrix task row, matching oracle_ceiling.matrix_to_tensor."""
    return str(task.get("id", task.get("task_id", fallback)))


def contested_task_ids(matrix: dict) -> list[str]:
    """Ids (in matrix order) of the tasks whose query is contested (see contested_mask)."""
    S, qids, _ = matrix_to_tensor(matrix)
    mask = contested_mask(S)
    return [qids[i] for i in range(len(qids)) if mask[i]]


def subset_matrix(matrix: dict, task_ids) -> dict:
    """Return a new matrix dict with only the tasks whose id is in ``task_ids``.

    Preserves the on-disk schema (all top-level keys plus ``tasks``) so the subset can be
    re-fed to training/eval unchanged, and stamps ``subset``/``subset_of``/``n_tasks`` so
    a downstream reader knows it is a derived disagreement slice, not a fresh collection.
    """
    wanted = set(str(t) for t in task_ids)
    tasks = matrix.get("tasks", [])
    kept = [t for i, t in enumerate(tasks) if _task_id(t, i) in wanted]
    out = {k: v for k, v in matrix.items() if k != "tasks"}
    out["tasks"] = kept
    out["subset"] = "contested"
    out["subset_of"] = matrix.get("benchmark")
    out["n_tasks"] = len(kept)
    return out


def write_contested(matrix: dict, out_path) -> int:
    """Write the contested (disagreement) subset to ``out_path``; return the task count."""
    ids = contested_task_ids(matrix)
    sub = subset_matrix(matrix, ids)
    Path(out_path).write_text(json.dumps(sub, indent=2))
    return len(ids)


def analyze_matrix(matrix: dict) -> dict:
    """Full per-question agreement report for a matrix dict -> machine-readable JSON."""
    S, qids, models = matrix_to_tensor(matrix)
    Q, M, K = S.shape
    acc = per_model_accuracy(S)
    unique = per_model_unique_solves(S)
    mask = contested_mask(S)
    n_contested = int(mask.sum())
    prs = perfect_router_stats(S)
    A = pairwise_agreement(S)
    pair = {
        f"{models[i]}|{models[j]}": float(A[i, j])
        for i in range(M)
        for j in range(i + 1, M)
    }
    return {
        "benchmark": matrix.get("benchmark"),
        "level": matrix.get("level"),
        "k": K,
        "n_queries": Q,
        "models": models,
        "per_model_accuracy": dict(zip(models, acc)),
        "per_model_unique_solves": dict(zip(models, unique)),
        "contested": {
            "n_contested": n_contested,
            "contested_rate": float(n_contested / Q) if Q else 0.0,
            "task_ids": [qids[i] for i in range(Q) if mask[i]],
        },
        "pairwise_agreement": pair,
        "perfect_router": {
            "accuracy": prs["perfect_router"],
            "best_single": prs["best_single"],
            "best_single_model": models[prs["best_single_model"]] if models else None,
            "headroom": prs["headroom"],
        },
        "notes": {
            "verdict": "per-model hard verdict is the K-sample majority (p_hat >= 0.5)",
            "contested": "only contested queries can change correctness under routing",
            "perfect_router": "optimistic per-question view; NOT winner's-curse debiased — "
            "use scripts/oracle_ceiling.py for the CI-gated routing verdict",
        },
    }


def _run_analyze(args) -> int:
    matrix = json.loads(Path(args.matrix).read_text())
    report = analyze_matrix(matrix)
    bench = matrix.get("benchmark", "unknown")
    out = _REPO / "experiments" / "final" / f"agreement_report_{bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[agreement] wrote {out}")
    if args.write_contested:
        n = write_contested(matrix, args.write_contested)
        print(f"[agreement] wrote {n} contested tasks -> {args.write_contested}")
    return 0


# =============================================================================
# Self-test (synthetic, offline). Mirrors tests/test_agreement_diagnostic.py.
# =============================================================================
def _selftest() -> int:
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # (a) 3 disjoint specialists: each query solved by exactly one distinct model.
    Q, K = 30, 5
    S = np.zeros((Q, 3, K))
    for q in range(Q):
        S[q, q % 3, :] = 1.0
    check("(a) every query contested", int(contested_mask(S).sum()) == Q)
    check("(a) each model owns Q/3 unique solves",
          per_model_unique_solves(S) == [10, 10, 10])
    check("(a) pairwise agreement == 1/3 (both wrong on the other third)",
          abs(pairwise_agreement(S)[0, 1] - 1.0 / 3.0) < 1e-9)
    prs = perfect_router_stats(S)
    check("(a) perfect router == 1.0", abs(prs["perfect_router"] - 1.0) < 1e-9)
    check("(a) best single == 1/3", abs(prs["best_single"] - 1.0 / 3.0) < 1e-9)
    check("(a) headroom == 2/3", abs(prs["headroom"] - 2.0 / 3.0) < 1e-9)

    # (b) 3 identical models -> nothing contested, no unique solves, agreement 1.0.
    rng = np.random.default_rng(1)
    base = (rng.random((Q, 1, K)) < 0.6).astype(float)
    S_id = np.repeat(base, 3, axis=1)
    check("(b) identical models: nothing contested",
          int(contested_mask(S_id).sum()) == 0)
    check("(b) identical models: zero unique solves",
          per_model_unique_solves(S_id) == [0, 0, 0])
    check("(b) identical models: pairwise agreement == 1.0",
          abs(pairwise_agreement(S_id)[0, 1] - 1.0) < 1e-9)
    check("(b) identical models: zero headroom",
          abs(perfect_router_stats(S_id)["headroom"]) < 1e-9)

    # (c) matrix I/O: contested_task_ids + write_contested keep only disagreement rows.
    matrix = {
        "benchmark": "mix", "k": 1, "level": "L0",
        "tasks": [
            {"id": "all_solve", "per_model": {"a": [1], "b": [1], "c": [1]}},
            {"id": "all_fail", "per_model": {"a": [0], "b": [0], "c": [0]}},
            {"id": "contest1", "per_model": {"a": [1], "b": [0], "c": [0]}},
            {"id": "contest2", "per_model": {"a": [0], "b": [1], "c": [1]}},
        ],
    }
    ids = contested_task_ids(matrix)
    check("(c) contested_task_ids finds exactly the two contested rows",
          ids == ["contest1", "contest2"])
    sub = subset_matrix(matrix, ids)
    check("(c) subset keeps only contested tasks", len(sub["tasks"]) == 2)
    check("(c) subset preserves schema (benchmark/k) + stamps provenance",
          sub["benchmark"] == "mix" and sub["k"] == 1 and sub["subset"] == "contested")
    # The subset must round-trip through the shared tensor reader.
    S_sub, qids_sub, _ = matrix_to_tensor(sub)
    check("(c) subset round-trips through matrix_to_tensor",
          S_sub.shape[0] == 2 and qids_sub == ["contest1", "contest2"])

    print(f"\n[selftest] {'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 0 if not failures else 1


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-question agreement diagnostic + contested-subset export"
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--matrix", metavar="MATRIX_JSON",
                      help="analyze an oracle_matrix_<bench>.json produced by oracle_ceiling.py")
    mode.add_argument("--selftest", action="store_true",
                      help="offline synthetic tests of the diagnostic math")
    ap.add_argument("--write-contested", metavar="OUT_JSON", dest="write_contested",
                    default="",
                    help="also export the disagreement subset (matrix schema) to OUT_JSON")

    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if args.matrix:
        sys.exit(_run_analyze(args))


if __name__ == "__main__":
    main()
