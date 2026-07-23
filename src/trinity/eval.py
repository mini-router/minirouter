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
from .costing import default_cost_ledger_path, ledger_cost_report
from .orchestration.async_utils import gather_in_batches
from .llm.pool_factory import build_pool
from .orchestration import reward as R
from .orchestration.dataset import load_tasks
from .orchestration.session import run_trajectory
from .types import ROLE_ORDER, Role

_REPO = Path(__file__).resolve().parents[2]


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _selected_benchmarks(args) -> list[str]:
    benchmarks = _split_csv(getattr(args, "benchmarks", ""))
    if benchmarks:
        return benchmarks
    benchmarks = _split_csv(getattr(args, "benchmark", ""))
    if benchmarks:
        return benchmarks
    raise ValueError("set --benchmark or --benchmarks")


def _select_pool_models(pool, raw: str | None) -> list[str]:
    available = list(pool.models)
    requested = _split_csv(raw)
    if not requested:
        return available

    aliases: dict[str, str] = {}
    for name in available:
        aliases[name.lower()] = name
        try:
            provider, model_id = pool.describe_model(name)
        except Exception:
            provider, model_id = "", str(pool.models.get(name, ""))
        for alias in (
            model_id,
            f"{provider}-{name}",
            f"{provider}:{name}",
            f"{provider}/{name}",
        ):
            if alias:
                aliases[alias.lower()] = name
        if model_id == "z-ai/glm-5.2":
            aliases[f"{provider}-glm-5"] = name
            aliases[f"{provider}:glm-5"] = name

    selected: list[str] = []
    for item in requested:
        match = aliases.get(item.lower())
        if match is None:
            known = ", ".join(available)
            raise ValueError(f"unknown pool model {item!r}; known logical routes: {known}")
        if match not in selected:
            selected.append(match)
    return selected


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


def _aggregate_runs(runs: list[dict], *, repeat: int, pool_models: list[str]) -> dict:
    by_benchmark: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        bucket = by_benchmark.setdefault(str(run["benchmark"]), {})
        for key, value in run.get("results", {}).items():
            if isinstance(value, (int, float)):
                bucket.setdefault(key, []).append(float(value))

    results_by_benchmark: dict[str, dict[str, float | list[float]]] = {}
    macro_values: list[float] = []
    for benchmark, metrics in by_benchmark.items():
        out: dict[str, float | list[float]] = {}
        main_values: list[float] = []
        for key, values in sorted(metrics.items()):
            avg = float(mean(values)) if values else 0.0
            out[key] = avg
            out[f"{key}::repeats"] = values
            if len(values) > 1:
                out[f"{key}::std"] = _std(values)
            if not key.startswith("single_std::"):
                main_values.append(avg)
        results_by_benchmark[benchmark] = out
        if main_values:
            macro_values.append(float(mean(main_values)))

    return {
        "benchmarks": list(results_by_benchmark),
        "pool_models": pool_models,
        "repeat": repeat,
        "results_by_benchmark": results_by_benchmark,
        "summary": {
            "macro_avg": float(mean(macro_values)) if macro_values else 0.0,
            "n_runs": len(runs),
        },
    }


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
            try:
                traj_coro = run_trajectory(
                    task,
                    policy,
                    pool,
                    pool_models,
                    sample=sample,
                    client=cli,
                    **{k: v for k, v in run_kwargs.items() if k != "trajectory_timeout_s"},
                )
                traj_timeout_s = run_kwargs.get("trajectory_timeout_s")
                if traj_timeout_s and traj_timeout_s > 0:
                    traj = await asyncio.wait_for(traj_coro, timeout=float(traj_timeout_s))
                else:
                    traj = await traj_coro
            except Exception as exc:
                print(
                    f"[eval] TRINITY task {i}/{total} failed score=0.000 "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                return None
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
            return_exceptions=True,
        )
    scores = []
    for traj in trajs:
        if isinstance(traj, BaseException):
            scores.append(0.0)
            continue
        scores.append(0.0 if traj is None else float(R.score(traj)))
    return float(mean(scores))


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
            try:
                traj_coro = run_trajectory(
                    task,
                    policy,
                    pool,
                    pool_models,
                    sample=sample,
                    client=cli,
                    **{k: v for k, v in run_kwargs.items() if k != "trajectory_timeout_s"},
                )
                traj_timeout_s = run_kwargs.get("trajectory_timeout_s")
                if traj_timeout_s and traj_timeout_s > 0:
                    traj = await asyncio.wait_for(traj_coro, timeout=float(traj_timeout_s))
                else:
                    traj = await traj_coro
            except Exception as exc:
                print(
                    f"[submission] item {i}/{total} failed score=0.000 "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                return None
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
            return_exceptions=True,
        )
    scores = []
    for traj in trajs:
        if isinstance(traj, BaseException):
            scores.append(0.0)
            continue
        scores.append(0.0 if traj is None else float(R.score(traj)))
    score = float(mean(scores)) if scores else 0.0
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
            try:
                res = await pool.chat(model, msgs, max_tokens=max_tokens, temperature=0.0,
                                      reasoning=reasoning, client=cli)
            except Exception as exc:
                print(
                    f"[eval] single::{model} task {idx}/{total} failed score=0.000 "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                return 0.0
            return R.score_text(benchmark, res.text, task.answer)

        async def run_one(task, i: int):
            print(f"[eval] single::{model} task {i}/{total} id={task.task_id}", flush=True)
            score = await one(task, i)
            print(f"[eval] single::{model} task {i}/{total} done score={score:.3f}", flush=True)
            return score

        scores = await gather_in_batches(
            [run_one(task, i) for i, task in enumerate(tasks, start=1)],
            batch_size=batch_size,
            return_exceptions=True,
        )
        scores = [0.0 if isinstance(score, BaseException) else float(score) for score in scores]
    return float(mean(scores))


async def _evaluate_once(args, *, benchmark: str, pool, pool_models: list[str], batch_size: int) -> dict:
    n_models = len(pool_models)

    tasks = load_tasks(benchmark, "test", max_items=args.max_items, seed=args.seed)
    print(
        f"[eval] benchmark={benchmark}  {len(tasks)} test tasks  "
        f"batch_size={batch_size} pool={pool_models}"
    )
    run_kwargs = dict(
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        request_timeout_s=args.request_timeout_s,
        trajectory_timeout_s=args.trajectory_timeout_s,
    )
    if args.request_timeout_s or args.trajectory_timeout_s:
        print(
            "[eval] timeouts "
            f"request={args.request_timeout_s or 'default'}s "
            f"trajectory={args.trajectory_timeout_s or 'default'}s",
            flush=True,
        )

    if args.single_only:
        results: dict[str, float] = {}
        for m in pool_models:
            reps = [
                await _score_single_model(
                    tasks,
                    pool,
                    m,
                    benchmark,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,
                    batch_size=batch_size,
                )
                for _ in range(max(1, args.single_reps))
            ]
            s = float(mean(reps))
            results[f"single::{m}"] = s
            if len(reps) > 1:
                sd = _std(reps)
                results[f"single_std::{m}"] = sd
                print(f"  single  {m:20s} = {s:.4f} +/- {sd:.4f}  (reps={reps})")
            else:
                print(f"  single  {m:20s} = {s:.4f}")
        return {
            "benchmark": benchmark,
            "results": results,
            "invariants": {},
        }

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
        return {
            "benchmark": benchmark,
            "results": results,
            "invariants": {},
        }

    results: dict[str, float] = {}

    # --- single-model baselines (R1/R2) ---
    for m in pool_models:
        reps = [await _score_single_model(tasks, pool, m, benchmark,
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
    print("[eval] invariants:", json.dumps(invariants, indent=2))
    return {
        "benchmark": benchmark,
        "results": results,
        "invariants": invariants,
    }


async def evaluate(args) -> dict:
    cost_ledger_path = default_cost_ledger_path(args.out)
    cost_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRINITY_COST_LEDGER", str(cost_ledger_path))

    t0 = time.perf_counter()
    pool = build_pool(args.provider, args.models)
    pool_models = _select_pool_models(pool, args.pool_models)
    if not pool_models:
        raise ValueError("no pool models selected")
    batch_size = max(1, int(args.batch_size))
    benchmarks = _selected_benchmarks(args)
    repeat = max(1, int(args.repeat))

    runs: list[dict] = []
    for repeat_index in range(1, repeat + 1):
        for benchmark in benchmarks:
            if repeat > 1 or len(benchmarks) > 1:
                print(
                    f"[eval] run repeat={repeat_index}/{repeat} benchmark={benchmark}",
                    flush=True,
                )
            run = await _evaluate_once(
                args,
                benchmark=benchmark,
                pool=pool,
                pool_models=pool_models,
                batch_size=batch_size,
            )
            run["repeat_index"] = repeat_index
            runs.append(run)

    runtime_seconds = round(time.perf_counter() - t0, 2)
    cost = ledger_cost_report(cost_ledger_path)
    if len(runs) == 1 and repeat == 1 and len(benchmarks) == 1:
        out = dict(runs[0])
    else:
        out = _aggregate_runs(runs, repeat=repeat, pool_models=pool_models)
        out["runs"] = runs
        if len(benchmarks) == 1:
            out["benchmark"] = benchmarks[0]
            out["results"] = out["results_by_benchmark"].get(benchmarks[0], {})
    out["runtime"] = {"duration_seconds": runtime_seconds}
    out["cost"] = cost
    print(f"[eval] runtime={runtime_seconds:.2f}s cost=${cost['cost_usd']:.4f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TRINITY + baselines")
    ap.add_argument("--benchmark", default="", help="benchmark name, or comma-separated benchmark names")
    ap.add_argument(
        "--benchmarks",
        default="",
        help="comma-separated benchmark names; overrides --benchmark when set",
    )
    ap.add_argument("--theta", default="", help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--provider", default="fireworks",
                    choices=["fireworks", "openrouter", "chutes", "minibridge", "compatible", "openai-compatible"])
    ap.add_argument(
        "--pool-models",
        default="",
        help=(
            "comma-separated logical routes to evaluate, for example "
            "openrouter-glm-5p2,chutes-glm-5"
        ),
    )
    ap.add_argument("--device", default="", help="override coordinator device (for example cpu or cuda:0)")
    ap.add_argument("--dtype", default="", help="override coordinator dtype (for example float32 or bfloat16)")
    ap.add_argument("--max-items", type=int, default=100, dest="max_items")
    ap.add_argument("--single-reps", type=int, default=1, dest="single_reps",
                    help="average each single-model baseline over K runs (cuts nondeterminism noise)")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument(
        "--request-timeout-s",
        type=float,
        default=0.0,
        dest="request_timeout_s",
        help="per-provider request timeout for each LLM call inside a trajectory",
    )
    ap.add_argument(
        "--trajectory-timeout-s",
        type=float,
        default=0.0,
        dest="trajectory_timeout_s",
        help="wall-clock timeout for one evaluation trajectory",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each selected benchmark/route job K times and report averaged scores",
    )
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
    ap.add_argument("--single-only", action="store_true",
                    help="evaluate selected pool routes directly and skip the router/random baselines")
    args = ap.parse_args()
    if not args.single_only and not args.theta:
        ap.error("--theta is required unless --single-only is set")
    try:
        _selected_benchmarks(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    if args.single_reps < 1:
        ap.error("--single-reps must be >= 1")
    if args.single_only and args.submission_only:
        ap.error("--single-only and --submission-only cannot be used together")
    if not args.single_only and args.pool_models:
        print(
            "[eval] warning: --pool-models changes the coordinator route count; "
            "the theta file must match the selected route count.",
            flush=True,
        )
    if args.trace_llm:
        os.environ["TRINITY_TRACE_LLM"] = "1"
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
