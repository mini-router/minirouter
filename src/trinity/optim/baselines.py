"""R8 optimizer baselines: Random Search, SFT (imitation), and REINFORCE.

Replication claim R8 (docs/SPEC.md §1.3 / §7.1 Table 4) requires comparing
sep-CMA-ES against three budget-matched trainers. This module supplies those
trainers; ``trinity.train --optimizer {rs,sft,reinforce}`` wires them into the
same experiment-dir / cost-ledger layout as CMA.

Design notes (paper Appendix A.2 / PAPER_NOTES §9; SPEC §8 ``baselines.py``):

* **RS** — sample ``θ ∼ U[lo, hi]^n`` with the SVF block centered on identity
  (``1 + U[lo, hi]``), average fitness over ``m_rs`` trials per candidate, stop
  when the cumulative trial count reaches ``B_env`` (budget-matched to CMA).
* **SFT** — single-step state→agent MLE on the linear head only (frozen SLM,
  SVF identity). Reuses :func:`trinity.coordinator.warmstart.fit_agent_head`
  (pure numpy) so it unit-tests without torch/GPU. Live training needs an
  oracle-label matrix (+ optional cached encodings).
* **REINFORCE** — Williams (1992) score-function on the two-softmax head.
  Advantage = reward − batch mean. Analytical ``∇_W log π`` (no autograd),
  head-only by default ([OUR CHOICE]: SVF stays identity, matching SFT's
  freeze; LR / entropy coef are also [OUR CHOICE] — paper left them
  unspecified; see JOURNAL).

The expensive env/fitness function is injected by the caller (real
``evaluate_candidate`` on GPU5, or a synthetic objective in tests). Core math
is pure numpy with **no torch dependency**.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

import numpy as np

from ..coordinator import params as P
from ..coordinator import warmstart as WS

__all__ = [
    "RSConfig",
    "SFTConfig",
    "ReinforceConfig",
    "sample_rs_theta",
    "run_random_search",
    "run_sft",
    "log_prob_grad_head",
    "reinforce_update_head",
    "run_reinforce",
    "budget_matched_rs_candidates",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RSConfig:
    """Random-search hyperparameters (SPEC ``baselines.random_search``)."""

    sample_lo: float = -0.5
    sample_hi: float = 0.5
    trials_per_candidate: int = 32
    budget_b_env: int = 31680
    seed: int = 0


@dataclass(frozen=True)
class SFTConfig:
    """Imitation / SFT hyperparameters (SPEC ``baselines.sft``).

    ``lr`` / ``batch_size`` match the paper (Adam 1e-6 / 64). The numpy path
    used here is full-batch GD (same as warm-start); ``steps`` is [OUR CHOICE]
    so a short offline fit still moves the head on specialist labels.
    """

    lr: float = 1.0e-6
    batch_size: int = 64
    steps: int = 400
    l2: float = 1e-3
    seed: int = 0
    # When True, use a higher numpy-friendly LR (warm-start default). The paper
    # Adam 1e-6 is too small for the pure-numpy full-batch path to move in 400
    # steps on unit tests; live GPU Adam can override via config later.
    numpy_lr: float = 0.5


@dataclass(frozen=True)
class ReinforceConfig:
    """REINFORCE hyperparameters (SPEC ``baselines.reinforce``).

    ``batch_size`` defaults to ``m_cma * λ = 16 * 33 = 528``. ``lr`` and
    ``entropy_coef`` are [OUR CHOICE] — SPEC_REVIEW flagged them as unspecified.
    """

    batch_size: int = 528
    iterations: int = 60
    lr: float = 1.0e-3
    entropy_coef: float = 0.01
    seed: int = 0


def budget_matched_rs_candidates(budget_b_env: int, trials_per_candidate: int) -> int:
    """Number of RS candidates that exhaust ``B_env`` at ``m_rs`` trials each."""
    if trials_per_candidate < 1:
        raise ValueError(f"trials_per_candidate must be >= 1, got {trials_per_candidate}")
    if budget_b_env < 1:
        raise ValueError(f"budget_b_env must be >= 1, got {budget_b_env}")
    return max(1, budget_b_env // trials_per_candidate)


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def sample_rs_theta(
    spec: P.ParamSpec,
    rng: np.random.Generator,
    *,
    lo: float = -0.5,
    hi: float = 0.5,
) -> np.ndarray:
    """Draw one RS candidate: head ~ U[lo,hi], SVF = 1 + U[lo,hi] (identity-centered)."""
    if hi <= lo:
        raise ValueError(f"sample range requires hi > lo, got [{lo}, {hi}]")
    head = rng.uniform(lo, hi, size=spec.head_shape)
    svf = 1.0 + rng.uniform(lo, hi, size=spec.n_svf)
    return P.pack(head, svf)


async def run_random_search(
    evaluate_fn: Callable[[np.ndarray, int], Awaitable[float]],
    spec: P.ParamSpec,
    cfg: RSConfig | None = None,
) -> tuple[np.ndarray, float, list[dict]]:
    """Budget-matched random search that **maximizes** ``evaluate_fn``.

    Args:
        evaluate_fn: ``async (theta, n_trials) -> fitness`` — mean reward over
            ``n_trials`` task instances (the RS ``m_rs`` average).
        spec: θ layout.
        cfg: RS knobs; ``None`` → defaults.

    Returns:
        ``(best_theta, best_fitness, history)`` where each history row has
        ``candidate``, ``fitness``, ``best_fitness``, ``trials_used``.
    """
    cfg = cfg or RSConfig()
    rng = np.random.default_rng(cfg.seed)
    n_cand = budget_matched_rs_candidates(cfg.budget_b_env, cfg.trials_per_candidate)

    best_x: np.ndarray | None = None
    best_f = -math.inf
    history: list[dict] = []
    trials_used = 0

    for i in range(n_cand):
        theta = sample_rs_theta(
            spec, rng, lo=cfg.sample_lo, hi=cfg.sample_hi
        )
        fit = float(await evaluate_fn(theta, cfg.trials_per_candidate))
        trials_used += cfg.trials_per_candidate
        if fit > best_f:
            best_f = fit
            best_x = theta.copy()
        history.append(
            {
                "candidate": i,
                "fitness": fit,
                "best_fitness": float(best_f),
                "trials_used": trials_used,
            }
        )

    if best_x is None:
        best_x = P.initial_theta(spec)
        best_f = 0.0
    return best_x, float(best_f), history


# ---------------------------------------------------------------------------
# SFT (imitation)
# ---------------------------------------------------------------------------
def run_sft(
    encodings: np.ndarray,
    solve_prob: np.ndarray,
    spec: P.ParamSpec,
    cfg: SFTConfig | None = None,
    *,
    n_models: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Fit a head-only SFT θ from soft per-(query, model) solve rates.

    Returns ``(theta, losses)``. Role rows stay 0 and SVF stays 1.0 so the
    packed vector is eval-ready and CMA-comparable.
    """
    cfg = cfg or SFTConfig()
    n_a, d_h = spec.head_shape
    n_models = int(n_models if n_models is not None else min(3, n_a))
    Wa, losses = WS.fit_agent_head(
        encodings,
        solve_prob,
        n_models=n_models,
        steps=cfg.steps,
        lr=cfg.numpy_lr,
        l2=cfg.l2,
        seed=cfg.seed,
        return_history=True,
    )
    theta = WS.pack_warmstart_theta(Wa, spec)
    return theta, list(losses)


# ---------------------------------------------------------------------------
# REINFORCE (analytical head gradient)
# ---------------------------------------------------------------------------
def _softmax_1d(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).ravel()
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def log_prob_grad_head(
    h: np.ndarray,
    agent_idx: int,
    role_pos: int,
    head_W: np.ndarray,
    n_models: int,
) -> tuple[np.ndarray, float, float]:
    """Analytical ``∇_W log π(a,r | h)`` for the two-softmax linear head.

    Returns ``(grad_W, log_prob, entropy)`` where ``grad_W`` has the same shape
    as ``head_W`` (``n_a × d_h``), ``log_prob = log π_agent(a) + log π_role(r)``,
    and ``entropy`` is the sum of the two categorical entropies (for the optional
    entropy bonus).
    """
    h = np.asarray(h, dtype=float).ravel()
    W = np.asarray(head_W, dtype=float)
    n_a, d_h = W.shape
    if h.shape != (d_h,):
        raise ValueError(f"h shape {h.shape} != (d_h={d_h},)")
    if not (0 <= agent_idx < n_models):
        raise ValueError(f"agent_idx {agent_idx} out of range [0, {n_models})")
    n_roles = n_a - n_models
    if not (0 <= role_pos < n_roles):
        raise ValueError(f"role_pos {role_pos} out of range [0, {n_roles})")

    agent_logits = W[:n_models] @ h
    role_logits = W[n_models:] @ h
    pa = _softmax_1d(agent_logits)
    pr = _softmax_1d(role_logits)

    grad = np.zeros_like(W)
    # ∂logπ(a)/∂W_agent[j] = (1[j=a] - π_j) h
    for j in range(n_models):
        grad[j] = ((1.0 if j == agent_idx else 0.0) - pa[j]) * h
    for j in range(n_roles):
        grad[n_models + j] = ((1.0 if j == role_pos else 0.0) - pr[j]) * h

    log_prob = float(np.log(pa[agent_idx] + 1e-12) + np.log(pr[role_pos] + 1e-12))
    entropy = float(-(pa * np.log(pa + 1e-12)).sum() - (pr * np.log(pr + 1e-12)).sum())
    return grad, log_prob, entropy


def reinforce_update_head(
    head_W: np.ndarray,
    turns: Sequence[dict],
    reward: float,
    advantage: float,
    cfg: ReinforceConfig,
) -> np.ndarray:
    """One REINFORCE parameter update given a trajectory's turn records.

    Each turn dict needs keys ``h``, ``agent_idx``, ``role_pos``. The terminal
    ``advantage`` multiplies the sum of per-turn ``∇ log π`` (episodic REINFORCE).
    Entropy bonus uses the mean per-turn entropy.
    """
    W = np.asarray(head_W, dtype=float).copy()
    if not turns:
        return W
    n_models = int(turns[0].get("n_models", 3))
    g_sum = np.zeros_like(W)
    ent_sum = 0.0
    for t in turns:
        g, _lp, ent = log_prob_grad_head(
            t["h"], int(t["agent_idx"]), int(t["role_pos"]), W, n_models
        )
        g_sum += g
        ent_sum += ent
    # Maximize E[R]; ascent on advantage * ∇logπ, plus entropy bonus.
    ent_grad_scale = cfg.entropy_coef * (ent_sum / max(1, len(turns)))
    # Entropy w.r.t. W is not free analytically here; we use a small isotropic
    # jitter toward uniform by shrinking W when entropy_coef > 0 ([OUR CHOICE]).
    W = W + cfg.lr * (float(advantage) * g_sum) - cfg.lr * ent_grad_scale * W
    return W


async def run_reinforce(
    collect_fn: Callable[[np.ndarray], Awaitable[tuple[float, list[dict]]]],
    spec: P.ParamSpec,
    cfg: ReinforceConfig | None = None,
    *,
    n_models: int = 3,
) -> tuple[np.ndarray, float, list[dict]]:
    """Run head-only REINFORCE for ``cfg.iterations`` batches.

    Args:
        collect_fn: ``async (theta) -> (reward, turns)`` for **one** env episode.
            The trainer calls it ``batch_size`` times per iteration. ``turns`` is
            a list of dicts accepted by :func:`reinforce_update_head`.
        spec: θ layout.
        cfg: REINFORCE knobs.
        n_models: pool size (agent logits).

    Returns:
        ``(best_theta, best_batch_mean_reward, history)``.
    """
    cfg = cfg or ReinforceConfig()
    theta = P.initial_theta(spec)
    best_x = theta.copy()
    best_f = -math.inf
    history: list[dict] = []

    for it in range(cfg.iterations):
        rewards: list[float] = []
        episodes: list[list[dict]] = []
        for _ in range(cfg.batch_size):
            r, turns = await collect_fn(theta)
            for t in turns:
                t.setdefault("n_models", n_models)
            rewards.append(float(r))
            episodes.append(turns)

        batch_mean = float(np.mean(rewards)) if rewards else 0.0
        advantages = [r - batch_mean for r in rewards]

        head_W, svf = P.unpack(theta, spec)
        for turns, adv, r in zip(episodes, advantages, rewards):
            head_W = reinforce_update_head(head_W, turns, r, adv, cfg)
        # SVF stays identity ([OUR CHOICE] head-only REINFORCE).
        svf = np.ones(spec.n_svf, dtype=np.float64)
        theta = P.pack(head_W, svf)

        if batch_mean > best_f:
            best_f = batch_mean
            best_x = theta.copy()
        history.append(
            {
                "iteration": it,
                "batch_mean_reward": batch_mean,
                "best_fitness": float(best_f),
            }
        )

    return best_x, float(best_f), history


def rs_config_from_dict(d: dict | None, *, budget_fallback: int = 31680) -> RSConfig:
    """Build :class:`RSConfig` from the YAML ``baselines.random_search`` block."""
    d = d or {}
    lo_hi = d.get("sample_range", [-0.5, 0.5])
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return RSConfig(
        sample_lo=lo,
        sample_hi=hi,
        trials_per_candidate=int(d.get("trials_per_candidate", 32)),
        budget_b_env=int(d.get("budget_b_env", budget_fallback)),
        seed=int(d.get("seed", 0)),
    )


def sft_config_from_dict(d: dict | None) -> SFTConfig:
    """Build :class:`SFTConfig` from the YAML ``baselines.sft`` block."""
    d = d or {}
    return SFTConfig(
        lr=float(d.get("lr", 1.0e-6)),
        batch_size=int(d.get("batch_size", 64)),
        steps=int(d.get("steps", 400)),
        l2=float(d.get("l2", 1e-3)),
        seed=int(d.get("seed", 0)),
        numpy_lr=float(d.get("numpy_lr", 0.5)),
    )


def reinforce_config_from_dict(d: dict | None, *, batch_fallback: int = 528) -> ReinforceConfig:
    """Build :class:`ReinforceConfig` from the YAML ``baselines.reinforce`` block."""
    d = d or {}
    return ReinforceConfig(
        batch_size=int(d.get("batch_size", batch_fallback)),
        iterations=int(d.get("iterations", 60)),
        lr=float(d.get("lr", 1.0e-3)),
        entropy_coef=float(d.get("entropy_coef", 0.01)),
        seed=int(d.get("seed", 0)),
    )
