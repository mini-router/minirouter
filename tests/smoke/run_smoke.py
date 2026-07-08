"""TRINITY smoke-test ladder (docs/SPEC.md §11).

Each rung is cheap and gates the next. Run the CPU-only subset anywhere; run the
full ladder on the GPU box (GPU 5) with the Fireworks key sourced.

    # CPU only (no torch/GPU/network needed): S3, S4, S5, S7
    python -m tests.smoke.run_smoke --cpu

    # full ladder on the GPU box
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m tests.smoke.run_smoke --all

    # pick specific rungs
    python -m tests.smoke.run_smoke S3 S5

Exit code 0 iff every selected rung passes.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from types import SimpleNamespace

import numpy as np

CPU_RUNGS = ["S3", "S4", "S5", "S7"]
GPU_RUNGS = ["S1", "S2", "S8"]
NET_RUNGS = ["S6"]
ALL_RUNGS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


# ---------------------------------------------------------------------------
# Mocks for the CPU rungs (no torch, no network)
# ---------------------------------------------------------------------------
class MockPool:
    """Async chat stub. Scripted by (role-in-prompt) -> canned text."""

    def __init__(self, script):
        self._script = script
        self.calls = 0

    async def chat(self, model, messages, **kw):
        self.calls += 1
        sys_text = messages[0]["content"].lower() if messages else ""
        if "verifier" in sys_text:
            text = self._script.get("verifier", "Looks fine.\nVERDICT: ACCEPT")
        elif "worker" in sys_text:
            text = self._script.get("worker", "The answer is \\boxed{42}.")
        else:
            text = self._script.get("thinker", "Plan: solve it.")
        return SimpleNamespace(
            text=text, prompt_tokens=10, completion_tokens=5, finish_reason="stop"
        )


class MockPolicy:
    """Returns a scripted (agent_idx, Role) sequence, cycling if exhausted."""

    def __init__(self, sequence):
        self.sequence = sequence
        self.i = 0

    def decide(self, transcript_text, *, sample=False, rng=None):
        a, r = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return a, r


# ---------------------------------------------------------------------------
# S3 — params pack/unpack identity
# ---------------------------------------------------------------------------
def s3() -> tuple[bool, str]:
    from trinity.coordinator import params as P

    spec = P.make_spec()
    assert spec.n_total == 13312, spec
    assert spec.head_shape == (6, 1024), spec
    rng = np.random.default_rng(0)
    W = rng.standard_normal(spec.head_shape)
    svf = rng.standard_normal(spec.n_svf)
    theta = P.pack(W, svf)
    assert theta.shape == (spec.n_total,), theta.shape
    W2, svf2 = P.unpack(theta, spec)
    assert np.allclose(W, W2) and np.allclose(svf, svf2), "round-trip mismatch"
    t0 = P.initial_theta(spec)
    W0, svf0 = P.unpack(t0, spec)
    assert np.allclose(W0, 0.0) and np.allclose(svf0, 1.0), "initial_theta not (0, 1)"
    return True, f"n_total={spec.n_total}, head={spec.head_shape}, n_svf={spec.n_svf}; round-trip exact"


# ---------------------------------------------------------------------------
# S4 — one full inner trajectory with mocked LLM + termination rule
# ---------------------------------------------------------------------------
def s4() -> tuple[bool, str]:
    from trinity.orchestration.session import run_trajectory
    from trinity.types import Role, Task

    task = Task(task_id="t", benchmark="math500", prompt="2+2?", answer="4")
    models = ["m0", "m1", "m2"]

    # (a) Worker then Verifier-ACCEPT -> terminates at turn 2 (worker guard satisfied).
    pol = MockPolicy([(1, Role.WORKER), (2, Role.VERIFIER)])
    pool = MockPool({"verifier": "ok\nVERDICT: ACCEPT"})
    traj = asyncio.run(run_trajectory(task, pol, pool, models, max_turns=5))
    assert traj.terminated_by == "accept" and traj.n_turns == 2, (traj.terminated_by, traj.n_turns)

    # (b) Verifier-ACCEPT on turn 1 (no prior Worker) must NOT terminate (guard).
    pol = MockPolicy([(0, Role.VERIFIER), (1, Role.WORKER), (2, Role.VERIFIER)])
    pool = MockPool({"verifier": "ok\nVERDICT: ACCEPT"})
    traj = asyncio.run(run_trajectory(task, pol, pool, models, max_turns=5))
    assert traj.turns[0].verdict == "ACCEPT", "verdict not parsed"
    assert traj.terminated_by == "accept" and traj.n_turns == 3, (
        "turn-1 verifier should be blocked by worker guard",
        traj.terminated_by,
        traj.n_turns,
    )

    # (c) No VERDICT line -> fail-safe REVISE -> runs to max_turns.
    pol = MockPolicy([(1, Role.WORKER), (2, Role.VERIFIER)])
    pool = MockPool({"verifier": "I have no opinion."})
    traj = asyncio.run(run_trajectory(task, pol, pool, models, max_turns=3))
    assert traj.terminated_by == "max_turns" and traj.n_turns == 3, (traj.terminated_by, traj.n_turns)
    return True, "termination rule, worker-guard, and fail-safe REVISE all correct"


# ---------------------------------------------------------------------------
# S5 — reward checkers on known cases (incl. the P0 LCB input-key fix)
# ---------------------------------------------------------------------------
def s5() -> tuple[bool, str]:
    from trinity.orchestration import reward as R

    # math
    assert R.score_text("math500", "Thus \\boxed{42}.", "42") == 1.0
    assert R.score_text("math500", "Thus \\boxed{41}.", "42") == 0.0
    assert R.score_text("math500", "answer: 1/2", "0.5") == 1.0
    # choice
    assert R.score_text("mmlu", "The answer is (C).", "C") == 1.0
    assert R.score_text("mmlu", "The answer is (C).", "B") == 0.0
    assert R.extract_choice_letter("A nice approach to think about it") is None, "prose 'A' must not match"
    assert R.extract_choice_letter("Final answer:\nB") == "B"
    # code pass@1 via stdin/stdout, using the dataset's input/output key convention
    code_ok = "import sys\nn=int(sys.stdin.read())\nprint(n*n)"
    tests = [{"input": "5\n", "output": "25"}, {"input": "3\n", "output": "9"}]
    assert R.run_pass_at_1(code_ok, tests, timeout_s=10) is True, "correct code must pass (P0 fix)"
    code_bad = "import sys\nn=int(sys.stdin.read())\nprint(n+1)"
    assert R.run_pass_at_1(code_bad, tests, timeout_s=10) is False
    return True, "math / choice / code checkers correct; LCB input-key honored"


# ---------------------------------------------------------------------------
# S7 — sep-CMA-ES on a synthetic objective at the real dimensionality
# ---------------------------------------------------------------------------
def s7() -> tuple[bool, str]:
    from trinity.optim.sep_cmaes import default_popsize, run

    n = 13312
    assert default_popsize(n) == 33, default_popsize(n)
    # Small n for speed but verify popsize logic separately above.
    rng = np.random.default_rng(0)
    target = rng.standard_normal(64)

    def objective(x):  # maximize -> peak at target
        return -float(np.sum((x - target) ** 2))

    best_x, best_f, history = run(objective, 64, sigma0=0.5, maxiter=30, seed=0)
    first = history[0]["best_fitness"]
    last = history[-1]["best_fitness"]
    assert last > first, f"J did not improve: {first} -> {last}"
    return True, f"popsize(13312)=33; synthetic J improved {first:.2f} -> {last:.2f} over {len(history)} iters"


# ---------------------------------------------------------------------------
# S1 / S2 / S8 — GPU rungs; S6 — network rung
# ---------------------------------------------------------------------------
def s1() -> tuple[bool, str]:
    from trinity.coordinator.slm import CoordinatorEncoder

    enc = CoordinatorEncoder()
    assert enc.hidden_size == 1024, enc.hidden_size
    txt = "QUERY:\nWhat is 2+2?\n\n[Turn 1 | worker | m0]\nThe answer is 4."
    h1 = enc.encode(txt)
    h2 = enc.encode(txt)
    assert h1.shape == (1024,), h1.shape
    assert np.allclose(h1, h2), "encode is not deterministic"
    norm = float(np.linalg.norm(h1))
    return True, f"hidden_size=1024, layers={enc.num_layers}, deterministic; ||h||={norm:.4f}"


def s2() -> tuple[bool, str]:
    import torch

    from trinity.coordinator.params import make_spec
    from trinity.coordinator.slm import CoordinatorEncoder
    from trinity.coordinator.svf import SVFAdapter

    enc = CoordinatorEncoder()
    svf = SVFAdapter(enc.model, target_layer=26)
    n_svf = int(svf.num_scales)
    spec = make_spec(n_svf=n_svf)
    assert spec.n_svf == n_svf, "spec/svf scale-count mismatch"

    # capture a target matrix before/after identity scales
    layer = enc.model.model.layers[26]
    w_before = layer.mlp.down_proj.weight.detach().float().clone()
    svf.set_scales(np.ones(n_svf))
    w_ident = layer.mlp.down_proj.weight.detach().float()
    max_diff = float((w_ident - w_before).abs().max())
    assert max_diff < 1e-2, f"identity scales did not round-trip: max|Δ|={max_diff}"

    # perturb one scale -> the corresponding matrix's weight must change
    pert = np.ones(n_svf)
    pert[0] = 1.5
    svf.set_scales(pert)
    w_pert = layer.mlp.down_proj.weight.detach().float()
    delta = float((w_pert - w_before).abs().max())
    svf.reset()
    w_reset = layer.mlp.down_proj.weight.detach().float()
    reset_diff = float((w_reset - w_before).abs().max())
    assert delta > 0.0, "perturbing a scale did not change any weight"
    assert reset_diff < 1e-3, f"reset() did not restore original weight: {reset_diff}"
    return True, (
        f"num_scales={n_svf} (expected 7168), identity max|Δ|={max_diff:.2e}, "
        f"perturb Δ={delta:.2e}, reset Δ={reset_diff:.2e}"
    )


def s6() -> tuple[bool, str]:
    from trinity.llm.fireworks_client import FireworksPool

    pool = FireworksPool()

    async def _run():
        import httpx

        async with httpx.AsyncClient() as cli:
            outs = []
            for name in pool.models:
                r = await pool.chat(
                    name,
                    [{"role": "user", "content": "Reply with OK"}],
                    max_tokens=8,
                    temperature=0.0,
                    reasoning="minimal",
                    client=cli,
                )
                outs.append((name, r.completion_tokens))
            return outs

    outs = asyncio.run(_run())
    return True, "live: " + ", ".join(f"{n}({t}t)" for n, t in outs)


def s8() -> tuple[bool, str]:
    from trinity.coordinator.policy import CoordinatorPolicy
    from trinity.llm.fireworks_client import FireworksPool
    from trinity.optim.fitness import evaluate_candidate
    from trinity.orchestration.dataset import load_tasks

    pool = FireworksPool()
    pool_models = list(pool.models)
    policy, spec = CoordinatorPolicy.build(n_models=len(pool_models))
    assert spec.n_svf == int(policy.svf.num_scales), "spec/svf mismatch"

    from trinity.coordinator.params import initial_theta

    theta = initial_theta(spec)
    tasks = load_tasks("math500", "test", max_items=2, seed=0, allow_toy_fallback=True)
    fit, trajs = asyncio.run(
        evaluate_candidate(
            theta, spec, policy, pool, pool_models, tasks,
            sample=False, max_turns=2, max_tokens=512, return_trajectories=True,
        )
    )
    assert 0.0 <= fit <= 1.0, fit
    calls = sum(t.n_turns for t in trajs)
    assert calls <= 2 * 2, f"too many calls: {calls}"
    return True, f"end-to-end fitness={fit:.3f} over {len(trajs)} instances, {calls} LLM calls"


RUNGS = {
    "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6, "S7": s7, "S8": s8,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rungs", nargs="*", help="specific rungs e.g. S3 S5")
    ap.add_argument("--cpu", action="store_true", help="run CPU-only rungs (S3,S4,S5,S7)")
    ap.add_argument("--all", action="store_true", help="run the full ladder")
    args = ap.parse_args()

    if args.all:
        selected = ALL_RUNGS
    elif args.cpu:
        selected = CPU_RUNGS
    elif args.rungs:
        selected = [r.upper() for r in args.rungs]
    else:
        selected = CPU_RUNGS

    print(f"Running rungs: {selected}\n" + "=" * 60)
    failures = 0
    for name in selected:
        fn = RUNGS.get(name)
        if fn is None:
            print(f"[SKIP] {name}: unknown rung")
            continue
        try:
            ok, msg = fn()
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"[{status}] {name}: {msg}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {exc!r}")
            traceback.print_exc()
    print("=" * 60)
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
