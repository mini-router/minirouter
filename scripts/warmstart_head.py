#!/usr/bin/env python3
"""Build a supervised warm-start theta for sep-CMA-ES (IMPROVEMENTS.md #2).

Two modes:

  --encode    (torch/GPU, run on the box) encode a split's query features with the
              frozen SLM into an encodings .npy, aligned with load_tasks(...) order.

  (default)   (pure numpy, runs anywhere) fit the head's agent-selection rows from a
              per-(query,model) correctness matrix + the cached encodings, pack into a
              full-length theta, and save it for `trinity.train --warmstart-theta`.

LEAKAGE WARNING: the labels (correctness matrix) AND the encodings used for the
warm-start must come from the TRAIN split, never the held-out eval/test split the
coordinator is scored on. The oracle-ceiling matrices in experiments/final/ were
collected on the TEST split (for the diagnostic), so for a clean warm-start run you
must collect a TRAIN-split correctness matrix first (oracle_ceiling.py --collect on
the train split) and encode the train queries. This script enforces row-count
alignment but cannot detect a wrong split — that is on the caller.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P  # noqa: E402
from trinity.coordinator import warmstart as WS  # noqa: E402


def _run_fit(args) -> int:
    qids, solve_prob, models = WS.load_labels(args.matrix)
    enc = np.load(args.encodings)
    if enc.ndim != 2:
        raise SystemExit(f"encodings must be 2D (N, d_h); got {enc.shape}")
    if enc.shape[0] != len(qids):
        raise SystemExit(
            f"encodings rows ({enc.shape[0]}) != matrix tasks ({len(qids)}); "
            "they must be the SAME queries in the SAME order (re-run --encode on this split)"
        )
    n_models = len(models)
    spec = P.make_spec(n_a=args.n_a, d_h=enc.shape[1], n_svf=args.n_svf)
    Wa, losses = WS.fit_agent_head(
        enc, solve_prob, n_models=n_models, steps=args.steps, lr=args.lr,
        l2=args.l2, tau=args.tau, prefer_disagree=not args.no_disagree,
        target_temp=args.target_temp, seed=args.seed, return_history=True,
    )
    theta = WS.pack_warmstart_theta(Wa, spec)
    np.save(args.out, theta)
    # quick self-report: training-set routing the warm head would pick
    pred = np.argmax(enc @ Wa.T, axis=1)
    routed = {models[m]: int((pred == m).sum()) for m in range(n_models)}
    print(f"[warmstart] models={models}")
    print(f"[warmstart] fit loss {losses[0]:.4f} -> {losses[-1]:.4f} over {args.steps} steps")
    print(f"[warmstart] warm head would route train queries as: {routed}")
    print(f"[warmstart] wrote {args.out}  (len={theta.size}, n_total={spec.n_total})")
    return 0


def _run_encode(args) -> int:
    # Lazy: only this mode needs torch + the GPU encoder + the dataset loader.
    import yaml

    from trinity.coordinator import warmstart as _ws
    from trinity.orchestration.dataset import load_tasks

    cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
    tasks = load_tasks(args.benchmark, args.split, max_items=args.max_items, seed=args.seed)
    prompts = [t.prompt for t in tasks]
    print(f"[encode] {len(prompts)} {args.benchmark}/{args.split} queries -> encoding on {args.device}")
    feats = _ws.encode_queries(
        prompts, model_name=cfg["encoder_model"], device=args.device,
        dtype=cfg.get("dtype", "bfloat16"), target_layer=cfg["svf"]["target_layer"],
        l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
        instruction=args.instruction or None,
    )
    np.save(args.out_encodings, feats)
    # also dump the aligned ids for a sanity cross-check against the matrix
    Path(str(args.out_encodings) + ".ids.json").write_text(json.dumps([t.task_id for t in tasks]))
    print(f"[encode] wrote {args.out_encodings}  shape={feats.shape}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised warm-start theta builder")
    ap.add_argument("--encode", action="store_true",
                    help="GPU mode: encode a split's queries to an encodings .npy")
    # fit mode
    ap.add_argument("--matrix", help="oracle_matrix_<bench>.json (per-(query,model) labels)")
    ap.add_argument("--encodings", help="encodings .npy aligned with the matrix tasks")
    ap.add_argument("--out", default="warm_theta.npy")
    ap.add_argument("--n-a", type=int, default=P.DEFAULT_N_A, dest="n_a")
    ap.add_argument("--n-svf", type=int, default=P.DEFAULT_N_SVF, dest="n_svf")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--target-temp", type=float, default=0.5, dest="target_temp")
    ap.add_argument("--no-disagree", action="store_true",
                    help="disable disagreement weighting (weight all queries equally)")
    # encode mode
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--out-encodings", default="encodings.npy", dest="out_encodings")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--instruction", default="",
                    help="optional instruction prefix prepended before encoding")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.encode:
        sys.exit(_run_encode(args))
    if not args.matrix or not args.encodings:
        ap.error("fit mode requires --matrix and --encodings (or pass --encode for GPU encoding)")
    sys.exit(_run_fit(args))


if __name__ == "__main__":
    main()
