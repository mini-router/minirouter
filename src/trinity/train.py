"""Entrypoint: evolve the coordinator with sep-CMA-ES or an R8 baseline.

One coordinator is trained per benchmark (SPEC §6.1). Each candidate θ is scored
by the mean binary reward over a freshly-sampled minibatch of train tasks.

Optimizers (``--optimizer``, claim R8):

* ``cma`` (default) — separable CMA-ES
* ``rs`` — budget-matched random search
* ``sft`` — head-only imitation from oracle soft labels (no online env loop)
* ``reinforce`` — head-only REINFORCE on sampled trajectories

Usage (on GPU or on CPU fallback if CUDA is unavailable)::

    python -m trinity.train --benchmark math500 --optimizer cma \\
        --config configs/trinity.yaml --models configs/models.yaml

Put your API key in ``secrets.env`` at the repo root or in
``~/.config/trinity/secrets.env``; the pool loader reads either one automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .coordinator.runtime import resolve_device_dtype
from .costing import ledger_cost_report
from .llm.pool_factory import build_pool
from .optim import baselines as B
from .optim.fitness import FitnessConfig, evaluate_candidate, evaluate_population
from .optim.sep_cmaes import SepCMAES, default_popsize
from .orchestration import reward as _reward
from .orchestration.dataset import load_tasks, sample_minibatch
from .orchestration.session import run_trajectory

_REPO = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _resolve_x0(args, spec) -> np.ndarray:
    """CMA-ES initial mean: a supervised warm-start theta if given, else the zero init.

    ``--warmstart-theta`` (IMPROVEMENTS.md #2) loads a pre-fit head produced by
    ``scripts/warmstart_head.py``. Its length must match ``spec.n_total`` exactly,
    otherwise it is a layout mismatch and we refuse to start (a silent reshape would
    corrupt the head/SVF split).
    """
    from .coordinator import warmstart as WS

    warm = getattr(args, "warmstart_theta", "") or ""
    if not warm:
        return P.initial_theta(spec)
    theta = WS.load_warmstart_theta(warm, spec)  # validates length == spec.n_total
    print(f"[train] warm-start x0 from {warm} (||head||={np.linalg.norm(theta[:spec.n_head]):.3f}, "
          f"deviates from zero-init by {float(np.linalg.norm(theta - P.initial_theta(spec))):.3f})")
    return theta


def _prepare_run(args):
    """Shared setup: config, pool, policy, tasks, run_dir, fitness_cfg, run_kwargs."""
    cfg = _load_yaml(args.config)
    cc = cfg["coordinator"]
    sc = cfg["sep_cmaes"]
    sess = cfg.get("session", {})
    bl = cfg.get("baselines", {})
    fitness_cfg = FitnessConfig.from_dict(cfg.get("fitness"))
    if getattr(args, "enable_reweight", False) and not fitness_cfg.enable_reweight:
        import dataclasses

        fitness_cfg = dataclasses.replace(fitness_cfg, enable_reweight=True)
    if fitness_cfg.enable_reweight or fitness_cfg.shaping_active:
        print(f"[train] fitness shaping ACTIVE: {fitness_cfg}")

    pool = build_pool(args.provider, args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    print(f"[train] optimizer={args.optimizer} benchmark={args.benchmark}  pool={pool_models}")
    device, dtype = resolve_device_dtype(
        requested_device=args.device,
        requested_dtype=args.dtype,
        default_device=cc.get("device", "cuda:0"),
        default_dtype=cc.get("dtype", "bfloat16"),
        context="train",
    )
    print(f"[train] building coordinator on {device}/{dtype} (this loads Qwen3-0.6B)...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=device,
        dtype=dtype,
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=n_models,
        n_roles=cc["head"].get("n_roles", 3),
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    assert spec.n_svf == int(policy.svf.num_scales), (
        f"spec.n_svf={spec.n_svf} != svf.num_scales={policy.svf.num_scales}"
    )
    print(f"[train] θ dimension n = {spec.n_total} (head {spec.n_head} + SVF {spec.n_svf})")

    tasks = load_tasks(args.benchmark, "train", max_items=args.max_items, seed=args.seed)
    print(f"[train] loaded {len(tasks)} train tasks")

    run_dir = _REPO / "experiments" / args.benchmark / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cost_ledger_path = Path(
        os.environ.get("TRINITY_COST_LEDGER", run_dir / "cost_ledger.jsonl")
    ).expanduser()
    cost_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRINITY_COST_LEDGER", str(cost_ledger_path))

    run_kwargs = dict(
        max_turns=args.max_turns or sess.get("max_turns", 5),
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        verifier_requires_prior_worker=sess.get("verifier_requires_prior_worker", True),
        request_timeout_s=args.request_timeout_s or sess.get("request_timeout_s"),
        trajectory_timeout_s=args.trajectory_timeout_s or sess.get("trajectory_timeout_s"),
    )
    if run_kwargs.get("request_timeout_s") or run_kwargs.get("trajectory_timeout_s"):
        print(
            "[train] timeouts "
            f"request={run_kwargs.get('request_timeout_s') or 'default'}s "
            f"trajectory={run_kwargs.get('trajectory_timeout_s') or 'default'}s",
            flush=True,
        )

    return {
        "cfg": cfg,
        "sc": sc,
        "bl": bl,
        "fitness_cfg": fitness_cfg,
        "pool": pool,
        "pool_models": pool_models,
        "n_models": n_models,
        "policy": policy,
        "spec": spec,
        "tasks": tasks,
        "run_dir": run_dir,
        "cost_ledger_path": cost_ledger_path,
        "run_kwargs": run_kwargs,
        "cc": cc,
    }


def _write_summary(run_dir: Path, summary: dict, cost_ledger_path: Path) -> dict:
    cost = ledger_cost_report(cost_ledger_path)
    summary = dict(summary)
    summary["cost"] = cost
    summary["run_dir"] = str(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[train] DONE. optimizer={summary.get('optimizer')} "
        f"best_fitness={summary.get('best_fitness', float('nan')):.4f} "
        f"cost=${cost['cost_usd']:.4f} -> {run_dir}",
        flush=True,
    )
    return summary


async def _train_cma(args, ctx) -> dict:
    sc, spec, policy = ctx["sc"], ctx["spec"], ctx["policy"]
    pool, pool_models = ctx["pool"], ctx["pool_models"]
    tasks, fitness_cfg, run_kwargs = ctx["tasks"], ctx["fitness_cfg"], ctx["run_kwargs"]
    run_dir, cost_ledger_path = ctx["run_dir"], ctx["cost_ledger_path"]

    popsize = args.popsize or sc.get("population_size") or default_popsize(spec.n_total)
    m_cma = args.m_cma or sc.get("m_cma", 16)
    generations = args.generations or sc.get("generations", 60)
    sigma0 = sc.get("sigma0", 0.1)
    x0 = _resolve_x0(args, spec)

    es = SepCMAES(
        n=spec.n_total,
        sigma0=sigma0,
        x0=x0,
        popsize=popsize,
        seed=args.seed,
        maxiter=generations,
    )
    print(
        f"[train] sep-CMA-ES: λ={es.popsize}, σ0={sigma0}, m_cma={m_cma}, T={generations}, "
        f"budget≈{es.popsize * m_cma * generations}"
    )

    history: list[dict] = []
    gen = 0
    while not es.stop() and gen < generations:
        t0 = time.time()
        thetas = es.ask()
        gen_rng = random.Random(args.seed * 100000 + gen)
        gen_minibatch = sample_minibatch(tasks, m_cma, gen_rng)

        def minibatch_fn(i, _mb=gen_minibatch):
            return _mb

        def _on_cand(i, fit, elapsed, _g=gen):
            print(
                f"    [gen {_g} cand {i + 1}/{len(thetas)}] fit={fit:.3f} ({elapsed:.0f}s)",
                flush=True,
            )

        fits = await evaluate_population(
            thetas,
            spec,
            policy,
            pool,
            pool_models,
            minibatch_fn,
            sample=True,
            on_candidate=_on_cand,
            fitness_cfg=fitness_cfg,
            **run_kwargs,
        )
        es.tell(thetas, fits)

        best_x, best_f = es.best()
        rec = {
            "generation": gen,
            "gen_mean_fitness": float(np.mean(fits)),
            "gen_max_fitness": float(np.max(fits)),
            "best_fitness": float(best_f),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[gen {gen:3d}] mean={rec['gen_mean_fitness']:.3f} "
            f"max={rec['gen_max_fitness']:.3f} best={rec['best_fitness']:.3f} "
            f"({rec['seconds']}s)"
        )
        cost = ledger_cost_report(cost_ledger_path)
        print(
            f"[train] runtime={time.time() - t0:.2f}s cost=${cost['cost_usd']:.4f} "
            f"ledger={cost['cost_ledger']}",
            flush=True,
        )
        np.save(run_dir / "best_theta.npy", best_x)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        gen += 1

    best_x, best_f = es.best()
    np.save(run_dir / "best_theta.npy", best_x)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return _write_summary(
        run_dir,
        {
            "optimizer": "cma",
            "benchmark": args.benchmark,
            "pool": pool_models,
            "n_total": spec.n_total,
            "popsize": es.popsize,
            "m_cma": m_cma,
            "generations": gen,
            "best_fitness": float(best_f),
        },
        cost_ledger_path,
    )


async def _train_rs(args, ctx) -> dict:
    sc, bl, spec = ctx["sc"], ctx["bl"], ctx["spec"]
    policy, pool, pool_models = ctx["policy"], ctx["pool"], ctx["pool_models"]
    tasks, fitness_cfg, run_kwargs = ctx["tasks"], ctx["fitness_cfg"], ctx["run_kwargs"]
    run_dir, cost_ledger_path = ctx["run_dir"], ctx["cost_ledger_path"]

    budget = int(sc.get("budget_b_env", 31680))
    rs_cfg = B.rs_config_from_dict(bl.get("random_search"), budget_fallback=budget)
    rs_cfg = B.RSConfig(
        sample_lo=rs_cfg.sample_lo,
        sample_hi=rs_cfg.sample_hi,
        trials_per_candidate=rs_cfg.trials_per_candidate,
        budget_b_env=budget,
        seed=args.seed,
    )
    n_cand = B.budget_matched_rs_candidates(rs_cfg.budget_b_env, rs_cfg.trials_per_candidate)
    print(
        f"[train] RS: range=[{rs_cfg.sample_lo}, {rs_cfg.sample_hi}], "
        f"m_rs={rs_cfg.trials_per_candidate}, candidates={n_cand}, B_env={rs_cfg.budget_b_env}"
    )

    async def evaluate_fn(theta: np.ndarray, n_trials: int) -> float:
        rng = random.Random(args.seed ^ (hash(theta.tobytes()) & 0xFFFFFFFF))
        mb = sample_minibatch(tasks, n_trials, rng)
        fit, _ = await evaluate_candidate(
            theta,
            spec,
            policy,
            pool,
            pool_models,
            mb,
            sample=True,
            fitness_cfg=fitness_cfg,
            **run_kwargs,
        )
        return float(fit)

    t0 = time.time()
    best_x, best_f, history = await B.run_random_search(evaluate_fn, spec, rs_cfg)
    print(f"[train] RS finished in {time.time() - t0:.1f}s best={best_f:.4f}")
    np.save(run_dir / "best_theta.npy", best_x)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return _write_summary(
        run_dir,
        {
            "optimizer": "rs",
            "benchmark": args.benchmark,
            "pool": pool_models,
            "n_total": spec.n_total,
            "m_rs": rs_cfg.trials_per_candidate,
            "candidates": n_cand,
            "best_fitness": float(best_f),
        },
        cost_ledger_path,
    )


async def _train_sft(args, ctx) -> dict:
    """Head-only imitation; no online env loop. Needs oracle labels (+ encodings)."""
    from .coordinator import warmstart as WS

    bl, spec, cc = ctx["bl"], ctx["spec"], ctx["cc"]
    pool_models, run_dir = ctx["pool_models"], ctx["run_dir"]
    cost_ledger_path = ctx["cost_ledger_path"]
    n_models = ctx["n_models"]

    labels_path = args.sft_labels or ""
    if not labels_path:
        raise SystemExit(
            "SFT requires --sft-labels pointing to an oracle_matrix_*.json "
            "(from scripts/oracle_ceiling.py)"
        )
    qids, solve_prob, models = WS.load_labels(labels_path)
    if not qids:
        raise SystemExit(f"no tasks in SFT labels file: {labels_path}")
    print(f"[train] SFT labels from {labels_path}: N={len(qids)} models={models}")

    enc_path = args.sft_encodings or ""
    if enc_path:
        encodings = np.load(enc_path)
        print(f"[train] SFT encodings from {enc_path}: {encodings.shape}")
    else:
        # Encode the query texts stored alongside the matrix if present; else fail.
        matrix = json.loads(Path(labels_path).read_text())
        prompts = []
        for t in matrix["tasks"]:
            p = t.get("prompt") or t.get("query") or t.get("question")
            if not p:
                raise SystemExit(
                    "SFT labels lack prompt text and --sft-encodings was not given; "
                    "pass a cached encodings .npy from warmstart_head.py"
                )
            prompts.append(p)
        print(f"[train] encoding {len(prompts)} queries with frozen SLM...")
        encodings = WS.encode_queries(
            prompts,
            model_name=cc["encoder_model"],
            device=cc.get("device", "cuda:0"),
            dtype=cc.get("dtype", "bfloat16"),
            l2_normalize=cc["hidden_state"].get("l2_normalize", True),
        )

    if encodings.shape[0] != solve_prob.shape[0]:
        raise SystemExit(
            f"encodings N={encodings.shape[0]} != solve_prob N={solve_prob.shape[0]}"
        )

    sft_cfg = B.sft_config_from_dict(bl.get("sft"))
    sft_cfg = B.SFTConfig(
        lr=sft_cfg.lr,
        batch_size=sft_cfg.batch_size,
        steps=sft_cfg.steps,
        l2=sft_cfg.l2,
        seed=args.seed,
        numpy_lr=sft_cfg.numpy_lr,
    )
    print(f"[train] SFT: steps={sft_cfg.steps} numpy_lr={sft_cfg.numpy_lr} n_models={n_models}")
    theta, losses = B.run_sft(
        encodings, solve_prob, spec, sft_cfg, n_models=min(n_models, solve_prob.shape[1])
    )
    history = [{"step": i, "loss": float(l)} for i, l in enumerate(losses)]
    np.save(run_dir / "best_theta.npy", theta)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    # SFT has no env fitness during training; report final CE loss as best_fitness
    # negated so "higher is better" still holds for summary consumers.
    best_fit = -float(losses[-1]) if losses else 0.0
    return _write_summary(
        run_dir,
        {
            "optimizer": "sft",
            "benchmark": args.benchmark,
            "pool": pool_models,
            "n_total": spec.n_total,
            "steps": sft_cfg.steps,
            "final_loss": float(losses[-1]) if losses else None,
            "best_fitness": best_fit,
            "labels": labels_path,
        },
        cost_ledger_path,
    )


class _ReinforceProbe:
    """Policy wrapper that records per-turn features for analytical REINFORCE."""

    def __init__(self, policy: CoordinatorPolicy):
        self.policy = policy
        self.turns: list[dict] = []

    def decide(self, transcript_text: str, *, sample: bool = False, rng=None):
        agent_idx, role, info = self.policy.decide_with_info(
            transcript_text, sample=sample, rng=rng
        )
        self.turns.append(
            {
                "h": np.asarray(info["h"], dtype=float).copy(),
                "agent_idx": int(info["agent_idx"]),
                "role_pos": int(info["role_pos"]),
                "n_models": int(info.get("n_models", self.policy.n_models)),
            }
        )
        return agent_idx, role


async def _train_reinforce(args, ctx) -> dict:
    sc, bl, spec = ctx["sc"], ctx["bl"], ctx["spec"]
    policy, pool, pool_models = ctx["policy"], ctx["pool"], ctx["pool_models"]
    tasks, run_kwargs = ctx["tasks"], ctx["run_kwargs"]
    run_dir, cost_ledger_path = ctx["run_dir"], ctx["cost_ledger_path"]
    n_models = ctx["n_models"]

    popsize = args.popsize or sc.get("population_size") or default_popsize(spec.n_total)
    m_cma = args.m_cma or sc.get("m_cma", 16)
    batch_fallback = int(m_cma) * int(popsize)
    rf_cfg = B.reinforce_config_from_dict(bl.get("reinforce"), batch_fallback=batch_fallback)
    # Keep smoke/dev runs affordable: --generations / a smaller batch via config.
    iterations = args.generations or rf_cfg.iterations
    rf_cfg = B.ReinforceConfig(
        batch_size=rf_cfg.batch_size,
        iterations=int(iterations),
        lr=rf_cfg.lr,
        entropy_coef=rf_cfg.entropy_coef,
        seed=args.seed,
    )
    print(
        f"[train] REINFORCE: batch={rf_cfg.batch_size}, T={rf_cfg.iterations}, "
        f"lr={rf_cfg.lr}, entropy_coef={rf_cfg.entropy_coef}"
    )

    probe = _ReinforceProbe(policy)
    task_rng = random.Random(args.seed)

    # run_trajectory accepts request_timeout_s but not trajectory_timeout_s
    # (that guard lives in evaluate_candidate).
    traj_kwargs = {
        k: v
        for k, v in run_kwargs.items()
        if k in ("max_turns", "max_tokens", "reasoning", "verifier_requires_prior_worker",
                 "request_timeout_s")
        and v is not None
    }

    async def collect_fn(theta: np.ndarray):
        policy.configure(theta, spec)
        probe.turns = []
        task = task_rng.choice(tasks)
        traj = await run_trajectory(
            task, probe, pool, pool_models, sample=True, **traj_kwargs
        )
        r = float(_reward.score(traj))
        return r, list(probe.turns)

    t0 = time.time()
    best_x, best_f, history = await B.run_reinforce(
        collect_fn, spec, rf_cfg, n_models=n_models
    )
    print(f"[train] REINFORCE finished in {time.time() - t0:.1f}s best_batch_mean={best_f:.4f}")
    np.save(run_dir / "best_theta.npy", best_x)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return _write_summary(
        run_dir,
        {
            "optimizer": "reinforce",
            "benchmark": args.benchmark,
            "pool": pool_models,
            "n_total": spec.n_total,
            "batch_size": rf_cfg.batch_size,
            "iterations": rf_cfg.iterations,
            "best_fitness": float(best_f),
        },
        cost_ledger_path,
    )


async def train(args) -> dict:
    ctx = _prepare_run(args)
    opt = (args.optimizer or "cma").lower()
    if opt == "cma":
        return await _train_cma(args, ctx)
    if opt == "rs":
        return await _train_rs(args, ctx)
    if opt == "sft":
        return await _train_sft(args, ctx)
    if opt == "reinforce":
        return await _train_reinforce(args, ctx)
    raise SystemExit(f"unknown --optimizer {args.optimizer!r}; expected cma|rs|sft|reinforce")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evolve the TRINITY coordinator (sep-CMA-ES or R8 baselines)"
    )
    ap.add_argument(
        "--benchmark",
        required=True,
        help="ifeval | math500 | mmlu | gpqa | livecodebench",
    )
    ap.add_argument(
        "--optimizer",
        default="cma",
        choices=["cma", "rs", "sft", "reinforce"],
        help="training algorithm (default cma; rs/sft/reinforce are R8 baselines)",
    )
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument(
        "--provider",
        default="fireworks",
        choices=["fireworks", "openrouter", "chutes", "minibridge"],
    )
    ap.add_argument("--device", default="", help="override coordinator device (for example cpu or cuda:0)")
    ap.add_argument("--dtype", default="", help="override coordinator dtype (for example float32 or bfloat16)")
    ap.add_argument("--max-items", type=int, default=256, dest="max_items")
    ap.add_argument("--max-turns", type=int, default=0, dest="max_turns", help="override K")
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
        help="wall-clock timeout for one training trajectory",
    )
    ap.add_argument("--generations", type=int, default=0, help="override config T (CMA/REINFORCE iters)")
    ap.add_argument("--popsize", type=int, default=0, help="override λ")
    ap.add_argument("--m-cma", type=int, default=0, dest="m_cma", help="override replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="run", dest="run_name")
    ap.add_argument(
        "--warmstart-theta",
        default="",
        dest="warmstart_theta",
        help="path to a warm-start theta .npy (scripts/warmstart_head.py); "
        "used as the sep-CMA-ES initial mean instead of the zero init",
    )
    ap.add_argument(
        "--enable-reweight",
        action="store_true",
        dest="enable_reweight",
        help="turn on variance-aware task reweighting (#3) regardless of config",
    )
    ap.add_argument(
        "--sft-labels",
        default="",
        dest="sft_labels",
        help="oracle_matrix JSON for --optimizer sft (from scripts/oracle_ceiling.py)",
    )
    ap.add_argument(
        "--sft-encodings",
        default="",
        dest="sft_encodings",
        help="optional cached (N, d_h) encodings .npy for --optimizer sft",
    )
    args = ap.parse_args()
    # argparse stores 0 for "not set" on the int overrides; normalize to None-ish.
    args.generations = args.generations or None
    args.popsize = args.popsize or None
    args.m_cma = args.m_cma or None
    args.max_turns = args.max_turns or None
    args.request_timeout_s = args.request_timeout_s or None
    args.trajectory_timeout_s = args.trajectory_timeout_s or None
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
