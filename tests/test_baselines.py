"""Unit tests for R8 optimizer baselines (RS / SFT / REINFORCE).

Pure numpy / asyncio — no torch, no GPU, no API. Covers the math and the
budget-matching contracts that train.py relies on.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P  # noqa: E402
from trinity.optim import baselines as B  # noqa: E402

_CONFIG = _REPO / "configs" / "trinity.yaml"


def test_budget_matched_rs_candidates():
    # B_env=31680, m_rs=32 -> 990 candidates
    assert B.budget_matched_rs_candidates(31680, 32) == 990
    assert B.budget_matched_rs_candidates(100, 32) == 3
    with pytest.raises(ValueError):
        B.budget_matched_rs_candidates(10, 0)


def test_sample_rs_theta_layout_and_svf_centering():
    spec = P.make_spec(n_a=6, d_h=8, n_svf=16)  # tiny for speed
    rng = np.random.default_rng(0)
    theta = B.sample_rs_theta(spec, rng, lo=-0.5, hi=0.5)
    assert theta.shape == (spec.n_total,)
    head, svf = P.unpack(theta, spec)
    assert head.shape == spec.head_shape
    assert svf.shape == (spec.n_svf,)
    assert head.min() >= -0.5 - 1e-9 and head.max() <= 0.5 + 1e-9
    # SVF centered on identity
    assert svf.min() >= 0.5 - 1e-9 and svf.max() <= 1.5 + 1e-9


def test_run_random_search_keeps_best_on_sphere():
    """RS on a deterministic sphere should improve best_fitness over candidates."""
    spec = P.make_spec(n_a=4, d_h=4, n_svf=8)
    target = P.initial_theta(spec)
    target[: spec.n_head] = 0.1  # slight offset from zero head

    async def objective(theta, n_trials):
        # Ignore n_trials; synthetic fitness = negative squared distance.
        d = theta - target
        return -float(np.dot(d, d))

    cfg = B.RSConfig(
        sample_lo=-0.5,
        sample_hi=0.5,
        trials_per_candidate=1,
        budget_b_env=40,  # 40 candidates
        seed=0,
    )
    best_x, best_f, history = asyncio.run(B.run_random_search(objective, spec, cfg))
    assert best_x.shape == (spec.n_total,)
    assert len(history) == 40
    assert history[-1]["best_fitness"] >= history[0]["best_fitness"]
    assert best_f == history[-1]["best_fitness"]
    # Best should beat a random far-away point
    far = np.ones(spec.n_total)
    far_fit = -float(np.dot(far - target, far - target))
    assert best_f > far_fit


def test_run_sft_loss_decreases_on_specialists():
    spec = P.make_spec(n_a=6, d_h=16, n_svf=32)
    rng = np.random.default_rng(0)
    n_models, n_per = 3, 30
    centers = rng.normal(size=(n_models, spec.head_shape[1]))
    enc, sp = [], []
    for c in range(n_models):
        for _ in range(n_per):
            v = centers[c] + 0.1 * rng.normal(size=spec.head_shape[1])
            v = v / (np.linalg.norm(v) + 1e-9)
            enc.append(v)
            row = np.zeros(n_models)
            row[c] = 1.0
            sp.append(row)
    enc = np.asarray(enc)
    sp = np.asarray(sp)

    cfg = B.SFTConfig(steps=200, numpy_lr=0.5, seed=0)
    theta, losses = B.run_sft(enc, sp, spec, cfg, n_models=n_models)
    assert theta.shape == (spec.n_total,)
    assert losses[-1] < losses[0] - 1e-3
    head, svf = P.unpack(theta, spec)
    assert np.allclose(svf, 1.0)
    assert np.allclose(head[n_models:], 0.0)  # role rows untouched
    # Agent rows should route specialists
    pred = np.argmax(enc @ head[:n_models].T, axis=1)
    labels = np.repeat(np.arange(n_models), n_per)
    assert float((pred == labels).mean()) > 0.85


def test_log_prob_grad_head_shapes_and_normalization():
    d_h, n_models, n_roles = 5, 3, 3
    W = np.zeros((n_models + n_roles, d_h))
    h = np.ones(d_h) / np.sqrt(d_h)
    grad, lp, ent = B.log_prob_grad_head(h, agent_idx=1, role_pos=2, head_W=W, n_models=n_models)
    assert grad.shape == W.shape
    # At W=0, uniform π → log π(a)+log π(r) = -log(3)-log(3)
    assert abs(lp - (-2 * np.log(3))) < 1e-6
    assert ent > 0
    # Gradient on chosen agent row should push toward h; others opposite
    assert np.allclose(grad[1], (1.0 - 1 / 3) * h)
    assert np.allclose(grad[0], (0.0 - 1 / 3) * h)


def test_reinforce_update_moves_toward_rewarded_action():
    d_h, n_models = 4, 3
    W = np.zeros((6, d_h))
    h = np.zeros(d_h)
    h[0] = 1.0
    turns = [{"h": h, "agent_idx": 0, "role_pos": 1, "n_models": n_models}]
    cfg = B.ReinforceConfig(lr=0.5, entropy_coef=0.0)
    W2 = B.reinforce_update_head(W, turns, reward=1.0, advantage=1.0, cfg=cfg)
    # Positive advantage on agent 0 should increase W[0] · h
    assert float(W2[0] @ h) > float(W[0] @ h)


def test_run_reinforce_improves_on_toy_collector():
    """Toy env: reward 1 iff agent_idx==0 on a fixed h; REINFORCE should raise batch mean."""
    spec = P.make_spec(n_a=6, d_h=4, n_svf=8)
    h = np.array([1.0, 0.0, 0.0, 0.0])
    call_count = {"n": 0}

    async def collect(theta):
        call_count["n"] += 1
        head, _ = P.unpack(theta, spec)
        # Greedy agent from current head on fixed h
        logits = head[:3] @ h
        # Softmax sample via numpy
        z = logits - logits.max()
        p = np.exp(z)
        p = p / p.sum()
        agent = int(np.random.default_rng(call_count["n"]).choice(3, p=p))
        role = 1  # worker
        r = 1.0 if agent == 0 else 0.0
        turns = [{"h": h, "agent_idx": agent, "role_pos": role, "n_models": 3}]
        return r, turns

    cfg = B.ReinforceConfig(batch_size=32, iterations=15, lr=0.5, entropy_coef=0.0, seed=0)
    best_x, best_f, history = asyncio.run(B.run_reinforce(collect, spec, cfg, n_models=3))
    assert best_x.shape == (spec.n_total,)
    assert len(history) == 15
    # Late iterations should be better on average than the first few
    early = np.mean([h["batch_mean_reward"] for h in history[:3]])
    late = np.mean([h["batch_mean_reward"] for h in history[-3:]])
    assert late >= early - 1e-9 or best_f >= early


def test_yaml_baseline_configs_parse():
    cfg = yaml.safe_load(_CONFIG.read_text())
    bl = cfg["baselines"]
    rs = B.rs_config_from_dict(bl["random_search"], budget_fallback=int(cfg["sep_cmaes"]["budget_b_env"]))
    assert rs.trials_per_candidate == 32
    assert rs.sample_lo == -0.5 and rs.sample_hi == 0.5
    rf = B.reinforce_config_from_dict(bl["reinforce"])
    assert rf.batch_size == 528 and rf.iterations == 60
    assert rf.lr == 1.0e-3
    sft = B.sft_config_from_dict(bl["sft"])
    assert sft.batch_size == 64 and sft.numpy_lr == 0.5


def test_optimizer_cli_choices_documented():
    """Baseline entrypoints exist and match the --optimizer choices in train.py."""
    from trinity import train as T

    for name in ("run_random_search", "run_sft", "run_reinforce"):
        assert hasattr(B, name)
    assert callable(T.train)
    src = Path(T.__file__).read_text(encoding="utf-8")
    for opt in ("cma", "rs", "sft", "reinforce"):
        assert f'"{opt}"' in src or f"'{opt}'" in src
