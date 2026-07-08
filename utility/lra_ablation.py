"""Offline ablation: LRA vs vanilla sep-CMA-ES under a NOISY fitness.

This is the honest gate docs/IMPROVEMENTS.md #4 prescribes ("validate on the S7
synthetic objective with injected noise") for the Learning-Rate Adaptation
controller. It reproduces, in a controlled setting, the failure mode LRA targets:
sep-CMA-ES fed a noisy fitness marches its mean on noise and some seeds collapse.

We optimize the S7 synthetic objective ``f(x) = -||x - x*||^2`` (maximized at
``x*``) with additive Gaussian noise on the *observed* fitness, and compare the
final true distance-to-optimum ``||x_best - x*||`` across many seeds for:

    * vanilla  — sep-CMA-ES as shipped (lra disabled)
    * lra       — the same optimizer with LRAConfig defaults

Success criteria (both must hold to justify shipping, default-off):
    1. NEUTRAL on a clean (noise-free) objective: LRA must not regress vanilla.
    2. ROBUST under injected noise: LRA lowers the mean and/or across-seed spread
       of the final distance (fewer collapsed seeds).

No GPU, no network, no API spend — pure numpy + pycma. Run:

    python utility/lra_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.optim.lra import LRAConfig  # noqa: E402
from trinity.optim.sep_cmaes import SepCMAES  # noqa: E402

N = 64            # search dimension (small so the ablation is fast)
GENERATIONS = 60  # matches the TRINITY training horizon T
SEEDS = 24        # independent optimizer runs per arm
SIGMA0 = 0.3
# Injected fitness-noise standard deviations to sweep (0.0 = clean control).
NOISE_LEVELS = (0.0, 0.75, 1.5, 3.0)


def _run_one(seed: int, noise_sd: float, lra: LRAConfig | None) -> float:
    """One optimizer run; returns the TRUE distance ||x_best - x*|| at the end.

    The optimizer only ever sees the *noisy* fitness; we score the incumbent on
    the clean objective to measure how far noise pushed it from the optimum.
    """
    rng = np.random.default_rng(seed)
    x_star = rng.standard_normal(N) * 0.1
    noise_rng = np.random.default_rng(1_000_003 * seed + 7)

    opt = SepCMAES(
        n=N, sigma0=SIGMA0, x0=np.zeros(N),
        seed=seed, maxiter=GENERATIONS, lra=lra,
    )
    # Each candidate's fitness carries iid noise of variance noise_sd**2; that is
    # exactly the estimation-noise signal the controller consumes. On the clean
    # arm noise_var == 0, so eta stays 1.0 (LRA is a no-op).
    noise_var = noise_sd * noise_sd
    while not opt.stop():
        sols = opt.ask()
        fits = []
        for x in sols:
            d = x - x_star
            true_f = -float(d @ d)
            noisy_f = true_f + (noise_sd * float(noise_rng.standard_normal()) if noise_sd else 0.0)
            fits.append(noisy_f)
        opt.tell(sols, fits, fitness_noise_var=noise_var)

    best_x, _ = opt.best()
    return float(np.linalg.norm(best_x - x_star))


def _arm(noise_sd: float, lra: LRAConfig | None) -> dict:
    dists = np.array([_run_one(s, noise_sd, lra) for s in range(SEEDS)])
    return {
        "mean": float(dists.mean()),
        "std": float(dists.std()),
        "worst": float(dists.max()),
        "p90": float(np.percentile(dists, 90)),
    }


def _report(noise_sd: float) -> tuple[dict, dict]:
    van = _arm(noise_sd, None)
    lra = _arm(noise_sd, LRAConfig())
    tag = "CLEAN " if noise_sd == 0.0 else "noisy "
    d_mean = 100.0 * (van["mean"] - lra["mean"]) / van["mean"]
    print(f"  {tag}sd={noise_sd:<4} | vanilla mean={van['mean']:7.4f} p90={van['p90']:7.4f} "
          f"| lra mean={lra['mean']:7.4f} p90={lra['p90']:7.4f} | dist -{d_mean:+.1f}%")
    return van, lra


def main() -> int:
    print(f"LRA vs vanilla sep-CMA-ES | {SEEDS} seeds, n={N}, T={GENERATIONS}, sigma0={SIGMA0}")
    print("final TRUE distance-to-optimum ||x_best - x*|| (lower is better)\n")
    results = {sd: _report(sd) for sd in NOISE_LEVELS}

    clean_v, clean_l = results[0.0]
    # Criterion 1: neutral on clean. With noise_var==0 the controller forces
    # eta==1.0 every generation, so LRA must be ESSENTIALLY IDENTICAL to vanilla
    # (differences only from float round-trips of mean/sigma, hence tiny slack).
    neutral = clean_l["mean"] <= clean_v["mean"] * 1.02 + 1e-6
    # Criterion 2: robust under noise — LRA lowers the mean final distance at
    # EVERY injected noise level (not just on average).
    noisy = [sd for sd in NOISE_LEVELS if sd > 0.0]
    robust = all(results[sd][1]["mean"] < results[sd][0]["mean"] for sd in noisy)
    # Bonus signal: the relative gain should grow with the noise level.
    gains = [100.0 * (results[sd][0]["mean"] - results[sd][1]["mean"]) / results[sd][0]["mean"]
             for sd in noisy]
    growing = gains == sorted(gains)

    print("\n=== VERDICT ===")
    print(f"  neutral on clean:      {neutral}  "
          f"(lra {clean_l['mean']:.4f} vs vanilla {clean_v['mean']:.4f})")
    print(f"  better at every noise: {robust}")
    print(f"  gain grows with noise: {growing}  ({[round(g, 1) for g in gains]}%)")
    ok = neutral and robust
    print(f"  SHIP: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
