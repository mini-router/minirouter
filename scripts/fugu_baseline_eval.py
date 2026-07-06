#!/usr/bin/env python3
"""Prompted-baseline evaluation of the Fugu Conductor over the open-source pool.

ZERO training: a :class:`PromptedConductor` (a Fireworks model) emits the
workflow, the pool (deepseek-v4-pro / glm-5p2 / kimi-k2p6) executes it, and we
report PURE-binary accuracy + parse rate + EXACT API cost. The per-query 0/1 it
writes feeds ``scripts/oracle_ceiling.py --analyze --trinity-per-query`` so the
Conductor can be compared against best-single and the routing ceiling on the
SAME held-out tasks (read the verdict off the CI, never the point estimate).

Cost is metered and capped (``--max-cost-usd``); set ``TRINITY_COST_LEDGER`` for
the exact per-call ledger that ``scripts/cost_report.py`` reads.

Tasks come from ``--tasks-json`` (a list of {task_id,benchmark,prompt,answer})
so the real benchmark can be materialized once by an env that has HF ``datasets``
and the paid eval can run in the lite env. Without it, the built-in loader runs
(toy set on a box with no ``datasets``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load_tasks(args):
    from trinity.types import Task

    if args.tasks_json:
        data = json.loads(Path(args.tasks_json).read_text())
        tasks = [
            Task(
                task_id=t["task_id"], benchmark=t["benchmark"],
                prompt=t["prompt"], answer=t["answer"], meta=t.get("meta", {}),
            )
            for t in data
        ]
    else:
        from trinity.orchestration.dataset import load_tasks
        tasks = load_tasks(args.benchmark, args.split, max_items=None, seed=args.seed)
    if args.max_items:
        tasks = tasks[: args.max_items]
    return tasks


async def _run(args) -> int:
    from trinity.fugu.conductor import PromptedConductor
    from trinity.fugu.cost import price_table
    from trinity.fugu.eval import evaluate
    from trinity.llm.fireworks_client import FireworksPool

    pool = FireworksPool(args.models)
    pool_models = list(pool.models)
    conductor = PromptedConductor(
        pool, args.conductor_model, max_tokens=args.conductor_max_tokens
    )
    tasks = _load_tasks(args)
    prices = price_table(args.conductor_model, conductor_local=False)

    print(
        f"[eval] benchmark={args.benchmark} tasks={len(tasks)} "
        f"conductor={args.conductor_model} pool={pool_models} reps={args.reps} "
        f"max_depth={args.max_depth} cap=${args.max_cost_usd}",
        flush=True,
    )
    res = await evaluate(
        conductor, tasks, pool, pool_models,
        reps=args.reps, max_depth=args.max_depth, prices=prices,
        cap_usd=args.max_cost_usd,
    )

    out = {
        "benchmark": args.benchmark,
        "conductor_model": args.conductor_model,
        "pool": pool_models,
        "n_tasks": res.n_tasks,
        "reps": res.reps,
        "max_depth": args.max_depth,
        "accuracy": res.accuracy,
        "parse_rate": res.parse_rate,
        "aborted": res.aborted,
        "cost": res.cost,
        "per_task": res.per_task,
    }
    out_dir = _REPO / "experiments" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fugu_baseline_{args.benchmark}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    (out_dir / f"fugu_baseline_perquery_{args.benchmark}.json").write_text(
        json.dumps(res.per_query_binary, indent=2)
    )

    print(json.dumps(
        {k: out[k] for k in ["n_tasks", "accuracy", "parse_rate", "aborted"]},
        indent=2,
    ))
    print("[cost] " + json.dumps(res.cost))
    print(f"[eval] wrote fugu_baseline_{args.benchmark}.json + perquery file")
    return 2 if res.aborted else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Fugu Conductor prompted-baseline eval")
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks-json", default="", dest="tasks_json")
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--conductor-model", default="deepseek-v4-pro", dest="conductor_model")
    ap.add_argument("--conductor-max-tokens", type=int, default=1024, dest="conductor_max_tokens")
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=0, dest="max_depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost-usd", type=float, default=5.0, dest="max_cost_usd")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
