"""Unit tests for the supervised warm-start (IMPROVEMENTS.md #2).

Pure numpy, NO torch (the dev box has no torch). Covers: specialist routing is
learned, the InfoNCE/CE loss decreases, packing places weights in the correct
theta slots, and label loading from the oracle_matrix schema.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P  # noqa: E402
from trinity.coordinator import warmstart as W  # noqa: E402


def _synthetic_specialist_data(n_per=40, d_h=16, n_models=3, seed=0):
    """3 clusters; cluster c is solved ONLY by model c. Returns (enc, solve_prob, labels)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_models, d_h))
    enc, sp, lab = [], [], []
    for c in range(n_models):
        for _ in range(n_per):
            v = centers[c] + 0.15 * rng.normal(size=d_h)
            v = v / (np.linalg.norm(v) + 1e-9)          # L2-normalized like the SLM feature
            enc.append(v)
            row = np.zeros(n_models)
            row[c] = 1.0                                 # only the specialist solves it
            sp.append(row)
            lab.append(c)
    return np.array(enc), np.array(sp), np.array(lab)


def test_fit_learns_specialist_routing():
    enc, sp, lab = _synthetic_specialist_data()
    Wa = W.fit_agent_head(enc, sp, n_models=3, steps=400, lr=0.5, seed=0)
    assert Wa.shape == (3, enc.shape[1])
    pred = np.argmax(enc @ Wa.T, axis=1)
    acc = float((pred == lab).mean())
    assert acc > 0.9, f"routing accuracy {acc:.3f} should exceed 0.9 on disjoint specialists"


def test_loss_decreases():
    enc, sp, _ = _synthetic_specialist_data()
    _, losses = W.fit_agent_head(enc, sp, n_models=3, steps=300, lr=0.5,
                                 seed=0, return_history=True)
    assert losses[-1] < losses[0] - 1e-3, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    # roughly monotone: late-window mean below early-window mean
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_pack_places_weights_in_agent_rows_only():
    spec = P.make_spec(n_a=6, d_h=1024, n_svf=7168)
    Wa = np.full((3, 1024), 0.37)
    theta = W.pack_warmstart_theta(Wa, spec)
    assert theta.shape == (spec.n_total,)
    head_W, svf = P.unpack(theta, spec)
    # agent rows match
    assert np.allclose(head_W[:3], 0.37)
    # role rows untouched (uniform policy)
    assert np.allclose(head_W[3:], 0.0)
    # SVF identity
    assert np.allclose(svf, 1.0)


def test_warmstart_theta_differs_from_zero_init():
    spec = P.make_spec()
    enc, sp, _ = _synthetic_specialist_data(d_h=spec.head_shape[1])
    Wa = W.fit_agent_head(enc, sp, n_models=3, steps=100, seed=0)
    theta = W.pack_warmstart_theta(Wa, spec)
    assert not np.allclose(theta, P.initial_theta(spec)), "warm theta must differ from zero init"
    # but SVF half identical to the default identity init
    assert np.allclose(theta[spec.n_head:], P.initial_theta(spec)[spec.n_head:])


def test_prefer_disagree_downweights_unanimous():
    # Unanimous queries (all models solve or none) should not dominate the fit.
    enc, sp, lab = _synthetic_specialist_data()
    # add many "all-solve" queries (no routing signal)
    rng = np.random.default_rng(1)
    extra = rng.normal(size=(200, enc.shape[1]))
    extra = extra / (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-9)
    enc2 = np.vstack([enc, extra])
    sp2 = np.vstack([sp, np.ones((200, 3))])
    Wa = W.fit_agent_head(enc2, sp2, n_models=3, steps=400, lr=0.5,
                          prefer_disagree=True, seed=0)
    pred = np.argmax(enc @ Wa.T, axis=1)
    acc = float((pred == lab).mean())
    assert acc > 0.85, f"disagreement weighting should preserve specialist routing (acc={acc:.3f})"


def test_load_labels(tmp_path):
    matrix = {
        "benchmark": "toy", "k": 2,
        "tasks": [
            {"id": "q0", "answer": "a", "per_model": {"m_a": [1, 1], "m_b": [0, 0], "m_c": [0, 1]}},
            {"id": "q1", "answer": "b", "per_model": {"m_a": [0, 0], "m_b": [1, 1], "m_c": [0, 0]}},
        ],
    }
    p = tmp_path / "oracle_matrix_toy.json"
    p.write_text(json.dumps(matrix))
    qids, sp, models = W.load_labels(str(p))
    assert qids == ["q0", "q1"]
    assert models == ["m_a", "m_b", "m_c"]
    assert sp.shape == (2, 3)
    assert np.allclose(sp[0], [1.0, 0.0, 0.5])   # m_c solved 1 of 2
    assert np.allclose(sp[1], [0.0, 1.0, 0.0])
