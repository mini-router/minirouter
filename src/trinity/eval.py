"""Entrypoint: evaluate a trained coordinator + baselines on a benchmark.

Reports the relative invariants from SPEC §1.3:
  - TRINITY (trained coordinator, argmax) vs
  - each single model alone (one direct Worker turn) [R1, R2] vs
  - random routing (random agent+role each turn) [R4].

Usage:
    python -m trinity.eval --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy
Put your API key in `secrets.env` at the repo root or in
`~/.config/trinity/secrets.env`; the pool loader reads either one automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .coordinator.runtime import resolve_device_dtype
from .orchestration.async_utils import gather_in_batches
from .llm.pool_factory import build_pool
from .orchestration import reward as R
from .orchestration.dataset import load_tasks
from .orchestration.session import run_trajectory
from .types import ROLE_ORDER, Role

_REPO = Path(__file__).resolve().parents[2]
_COST_PRICES = {
    "fireworks:accounts/fireworks/models/deepseek-v4-pro": (1.74, 3.48),
    "fireworks:accounts/fireworks/models/glm-5p2": (1.40, 4.40),
    "fireworks:accounts/fireworks/models/kimi-k2p6": (0.95, 4.00),
    "openrouter:deepseek-v4-pro": (0.435, 0.87),
    "openrouter:glm-5p2": (1.40, 4.40),
    "openrouter:kimi-k2p6": (0.95, 4.00),
    "openrouter:nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    "openrouter:google/gemma-4-31b-it:free": (0.0, 0.0),
    "openrouter:openai/gpt-oss-120b:free": (0.0, 0.0),
    "openrouter:qwen/qwen3-coder:free": (0.0, 0.0),
    "chutes:deepseek-ai/DeepSeek-V3.2-TEE": (1.00, 1.00),
    "chutes:zai-org/GLM-5-TEE": (1.40, 4.40),
    "chutes:moonshotai/Kimi-K2.5-TEE": (0.66, 3.50),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": (0.15, 1.20),
    "chutes:google/gemma-4-31B-turbo-TEE": (0.12, 0.37),
    "chutes:Qwen/Qwen3-32B-TEE": (0.10, 0.42),
}


def _cost_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _ledger_cost_report(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {
            "cost_usd": 0.0,
            "cost_missing": True,
            "cost_ledger": str(ledger_path),
        }

    per_model: dict[str, dict[str, float | int]] = {}
    total = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            provider = str(row.get("provider", "")).strip()
            model = str(row.get("m", "")).strip()
            pt = int(row.get("p", 0) or 0)
            ct = int(row.get("c", 0) or 0)
            pin, pout = _COST_PRICES.get(_cost_key(provider, model), (0.0, 0.0))
            usd = pt / 1e6 * pin + ct / 1e6 * pout
            total += usd
            prompt_tokens += pt
            completion_tokens += ct
            calls += 1
            bucket = per_model.setdefault(
                _cost_key(provider, model),
                {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usd": 0.0},
            )
            bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + pt
            bucket["completion_tokens"] = int(bucket["completion_tokens"]) + ct
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["usd"] = float(bucket["usd"]) + usd

    return {
        "cost_usd": round(total, 4),
        "cost_missing": False,
        "cost_ledger": str(ledger_path),
        "cost_calls": calls,
        "cost_prompt_tokens": prompt_tokens,
        "cost_completion_tokens": completion_tokens,
        "cost_per_model": {
            key: {
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "calls": int(row["calls"]),
                "usd": round(float(row["usd"]), 4),
            }
            for key, row in sorted(per_model.items())
        },
    }


def _default_cost_ledger_path(out_path: str | None) -> Path:
    if os.environ.get("TRINITY_COST_LEDGER"):
        return Path(os.environ["TRINITY_COST_LEDGER"]).expanduser()
    if out_path:
        return Path(out_path).expanduser().with_suffix(".cost_ledger.jsonl")
    return Path.cwd() / "cost_ledger.jsonl"


class RandomPolicy:
    """Random (agent, role) each turn — the R4 routing baseline (no GPU)."""

    def __init__(self, n_models: int, seed: int = 0):
        self.n_models = n_models
        self.rng = random.Random(seed)

    def decide(self, transcript_text, *, sample=False, rng=None):
        return self.rng.randrange(self.n_models), self.rng.choice(ROLE_ORDER)


async def _score_policy(
    tasks,
    policy,
    pool,
    pool_models,
    *,
    sample,
    batch_size: int = 1,
    **run_kwargs,
) -> float:
    import httpx

    async with httpx.AsyncClient() as cli:
        total = len(tasks)
        if total == 0:
            return 0.0

        async def one(task, i: int):
            print(f"[eval] TRINITY task {i}/{total} id={task.task_id}", flush=True)
            traj = await run_trajectory(
                task, policy, pool, pool_models, sample=sample, client=cli, **run_kwargs
            )
            score = R.score(traj)
            print(
                f"[eval] TRINITY task {i}/{total} done turns={len(traj.turns)} "
                f"score={score:.3f}",
                flush=True,
            )
            return traj

        trajs = await gather_in_batches(
            [one(task, i) for i, task in enumerate(tasks, start=1)],
            batch_size=batch_size,
        )
    return float(mean(R.score(t) for t in trajs))


async def _score_submission_policy(
    tasks,
    policy,
    pool,
    pool_models,
    *,
    sample,
    batch_size: int = 1,
    **run_kwargs,
) -> float:
    import httpx

    async with httpx.AsyncClient() as cli:
        total = len(tasks)
        benchmark = tasks[0].benchmark if tasks else "unknown"
        print(
            f"[submission] model initiated benchmark={benchmark} items={total} "
            f"batch_size={max(1, int(batch_size))} pool={pool_models}",
            flush=True,
        )
        if total == 0:
            print("[submission] completed score=0.0000", flush=True)
            return 0.0

        async def one(task, i: int):
            print(f"[submission] item {i}/{total} start id={task.task_id}", flush=True)
            traj = await run_trajectory(
                task, policy, pool, pool_models, sample=sample, client=cli, **run_kwargs
            )
            score = R.score(traj)
            verdict = "pass" if score >= 0.5 else "fail"
            print(
                f"[submission] item {i}/{total} done {verdict} score={score:.3f}",
                flush=True,
            )
            return traj

        trajs = await gather_in_batches(
            [one(task, i) for i, task in enumerate(tasks, start=1)],
            batch_size=batch_size,
        )
    score = float(mean(R.score(t) for t in trajs)) if trajs else 0.0
    print(f"[submission] completed score={score:.4f}", flush=True)
    return score


async def _score_single_model(
    tasks,
    pool,
    model,
    benchmark,
    *,
    max_tokens,
    reasoning,
    batch_size: int = 1,
) -> float:
    """Baseline: ask one model directly (one Worker-style turn), score its answer."""
    import httpx

    from .roles.prompts import build_messages

    async with httpx.AsyncClient() as cli:
        total = len(tasks)
        if total == 0:
            return 0.0

        async def one(task, idx: int):
            msgs = build_messages(Role.WORKER, task.prompt, [])
            res = await pool.chat(model, msgs, max_tokens=max_tokens, temperature=0.0,
                                  reasoning=reasoning, client=cli)
            return R.score_text(benchmark, res.text, task.answer)

        async def run_one(task, i: int):
            print(f"[eval] single::{model} task {i}/{total} id={task.task_id}", flush=True)
            score = await one(task, i)
            print(f"[eval] single::{model} task {i}/{total} done score={score:.3f}", flush=True)
            return score

        scores = await gather_in_batches(
            [run_one(task, i) for i, task in enumerate(tasks, start=1)],
            batch_size=batch_size,
        )
    return float(mean(scores))


async def evaluate(args) -> dict:
    cost_ledger_path = _default_cost_ledger_path(args.out)
    cost_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRINITY_COST_LEDGER", str(cost_ledger_path))

    t0 = time.perf_counter()
    pool = build_pool(args.provider, args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)
    batch_size = max(1, int(args.batch_size))

    tasks = load_tasks(args.benchmark, "test", max_items=args.max_items, seed=args.seed)
    print(
        f"[eval] benchmark={args.benchmark}  {len(tasks)} test tasks  "
        f"batch_size={batch_size} pool={pool_models}"
    )
    run_kwargs = dict(max_turns=args.max_turns, max_tokens=args.max_tokens, reasoning=args.reasoning)

    if args.submission_only:
        cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
        device, dtype = resolve_device_dtype(
            requested_device=args.device,
            requested_dtype=args.dtype,
            default_device=cfg.get("device", "cuda:0"),
            default_dtype=cfg.get("dtype", "bfloat16"),
            context="eval",
        )
        print(f"[eval] building coordinator on {device}/{dtype}...")
        policy, spec = CoordinatorPolicy.build(
            model_name=cfg["encoder_model"], device=device,
            dtype=dtype, target_layer=cfg["svf"]["target_layer"],
            svf_matrices=cfg["svf"].get("matrices"), n_models=n_models,
            l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
        )
        theta = np.load(args.theta)
        policy.configure(theta, spec)
        s_trinity = await _score_submission_policy(
            tasks,
            policy,
            pool,
            pool_models,
            sample=False,
            batch_size=batch_size,
            **run_kwargs,
        )
        results = {"TRINITY": s_trinity}
        runtime_seconds = round(time.perf_counter() - t0, 2)
        cost = _ledger_cost_report(cost_ledger_path)
        out = {
            "benchmark": args.benchmark,
            "results": results,
            "invariants": {},
            "runtime": {"duration_seconds": runtime_seconds},
            "cost": cost,
        }
        print(f"[eval] runtime={runtime_seconds:.2f}s cost=${cost['cost_usd']:.4f}", flush=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return out

    results: dict[str, float] = {}

    # --- single-model baselines (R1/R2) ---
    for m in pool_models:
        reps = [await _score_single_model(tasks, pool, m, args.benchmark,
                                          max_tokens=args.max_tokens, reasoning=args.reasoning,
                                          batch_size=batch_size)
                for _ in range(max(1, args.single_reps))]
        s = float(mean(reps))
        results[f"single::{m}"] = s
        if len(reps) > 1:
            sd = (sum((x - s) ** 2 for x in reps) / len(reps)) ** 0.5
            results[f"single_std::{m}"] = sd
            print(f"  single  {m:20s} = {s:.4f} ± {sd:.4f}  (reps={reps})")
        else:
            print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
    device, dtype = resolve_device_dtype(
        requested_device=args.device,
        requested_dtype=args.dtype,
        default_device=cfg.get("device", "cuda:0"),
        default_dtype=cfg.get("dtype", "bfloat16"),
        context="eval",
    )
    print(f"[eval] building coordinator on {device}/{dtype}...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cfg["encoder_model"], device=device,
        dtype=dtype, target_layer=cfg["svf"]["target_layer"],
        svf_matrices=cfg["svf"].get("matrices"), n_models=n_models,
        l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
    )
    theta = np.load(args.theta)
    policy.configure(theta, spec)
    s_trinity = await _score_policy(
        tasks,
        policy,
        pool,
        pool_models,
        sample=False,
        batch_size=batch_size,
        **run_kwargs,
    )
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- random routing (R4) ---
    rand = RandomPolicy(n_models, seed=args.seed)
    s_rand = await _score_policy(
        tasks,
        rand,
        pool,
        pool_models,
        sample=False,
        batch_size=batch_size,
        **run_kwargs,
    )
    results["random_routing"] = s_rand
    print(f"  random routing           = {s_rand:.4f}")

    best_single = max(results[k] for k in results if k.startswith("single::"))
    invariants = {
        "R1/R2 TRINITY > best single model": s_trinity > best_single,
        "R4 TRINITY > random routing": s_trinity > s_rand,
        "best_single": best_single,
    }
    runtime_seconds = round(time.perf_counter() - t0, 2)
    cost = _ledger_cost_report(cost_ledger_path)
    out = {
        "benchmark": args.benchmark,
        "results": results,
        "invariants": invariants,
        "runtime": {"duration_seconds": runtime_seconds},
        "cost": cost,
    }
    print("[eval] invariants:", json.dumps(invariants, indent=2))
    print(f"[eval] runtime={runtime_seconds:.2f}s cost=${cost['cost_usd']:.4f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TRINITY + baselines")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--provider", default="fireworks",
                    choices=["fireworks", "openrouter", "chutes"])
    ap.add_argument("--device", default="", help="override coordinator device (for example cpu or cuda:0)")
    ap.add_argument("--dtype", default="", help="override coordinator dtype (for example float32 or bfloat16)")
    ap.add_argument("--max-items", type=int, default=100, dest="max_items")
    ap.add_argument("--single-reps", type=int, default=1, dest="single_reps",
                    help="average each single-model baseline over K runs (cuts nondeterminism noise)")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EVAL_BATCH_SIZE", "1")),
        dest="batch_size",
        help="number of benchmark items to evaluate concurrently",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--trace-llm", action="store_true",
                    help="emit per-request OpenRouter/LLM trace logs")
    ap.add_argument("--submission-only", action="store_true",
                    help="evaluate the submitted router only and skip offline baselines")
    args = ap.parse_args()
    if args.trace_llm:
        os.environ["TRINITY_TRACE_LLM"] = "1"
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
