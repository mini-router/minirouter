"""Offline unit tests for the linear coordinator head (SPEC §3.3, Eq. 5).

The head IS the router: it maps the SLM hidden state ``h`` to two independent
categoricals — which pool agent to call and which role it plays — via ``z = W·h``
with a separate softmax per group, argmax at eval and sampling at train (SPEC
§3.3 / §4.3). It carried **no dedicated test coverage**; these lock the decision
contract so a future edit cannot silently break routing.

Pure CPU + torch (no GPU, no model download, no network): every case drives
``forward`` / ``select`` with hand-built weights and hidden states. To make
``z = W·h`` equal a chosen logit vector we feed ``h = e_0`` (a one-hot) and set
the first column of ``W`` to the desired logits — so the tests assert on exact,
predictable outputs rather than on a loaded 0.6B model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

torch = pytest.importorskip("torch")  # skip cleanly on a torch-less box

from trinity.coordinator.head import LinearHead  # noqa: E402
from trinity.types import ROLE_ORDER, Role  # noqa: E402

D_H = 4  # small hidden size keeps the tests fast; head logic is dim-agnostic


def _head_with_column0(logits6) -> LinearHead:
    """A head whose ``W·e_0`` equals ``logits6`` (len 6): put logits in W[:,0]."""
    head = LinearHead(n_a=6, d_h=D_H, n_models=3)
    W = np.zeros((6, D_H), dtype=np.float64)
    W[:, 0] = np.asarray(logits6, dtype=np.float64)
    head.load_weight(W)
    return head


_E0 = None


def _e0() -> "torch.Tensor":
    return torch.tensor([1.0] + [0.0] * (D_H - 1))


# --------------------------------------------------------------------------- #
# Construction contract
# --------------------------------------------------------------------------- #
def test_default_shape_and_zero_init():
    head = LinearHead()
    assert tuple(head.weight.shape) == (6, 1024)
    assert head.n_models == 3 and head.n_roles == 3
    assert torch.count_nonzero(head.weight).item() == 0  # zero-init


def test_role_count_mismatch_raises():
    # n_a - n_models = 2 role logits, but ROLE_ORDER has 3 -> reject.
    with pytest.raises(ValueError):
        LinearHead(n_a=5, d_h=D_H, n_models=3)


# --------------------------------------------------------------------------- #
# forward: z = W·h and the agent/role split
# --------------------------------------------------------------------------- #
def test_forward_splits_agent_and_role_logits():
    head = _head_with_column0([1, 2, 3, 4, 5, 6])
    agent, role = head.forward(_e0())
    assert agent.tolist() == pytest.approx([1.0, 2.0, 3.0])   # z[:3]
    assert role.tolist() == pytest.approx([4.0, 5.0, 6.0])    # z[3:]


def test_forward_is_bias_free_linear():
    # zero weight -> zero logits (no bias term, SPEC §3.3).
    head = LinearHead(n_a=6, d_h=D_H, n_models=3)
    agent, role = head.forward(_e0())
    assert agent.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert role.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_forward_preserves_leading_batch_dim():
    head = _head_with_column0([1, 2, 3, 4, 5, 6])
    h = torch.stack([_e0(), _e0() * 2.0])  # (2, D_H)
    agent, role = head.forward(h)
    assert tuple(agent.shape) == (2, 3)
    assert tuple(role.shape) == (2, 3)
    assert agent[1].tolist() == pytest.approx([2.0, 4.0, 6.0])  # scales with h


# --------------------------------------------------------------------------- #
# select: argmax (eval) path
# --------------------------------------------------------------------------- #
def test_argmax_selects_max_logit_per_group():
    # agent argmax at idx 2 (val 3); role logits [6,5,4] -> role argmax idx 0.
    head = _head_with_column0([1, 2, 3, 6, 5, 4])
    agent_idx, role, dbg = head.select(_e0(), sample=False)
    assert agent_idx == 2
    assert role is ROLE_ORDER[0] is Role.THINKER
    assert dbg["sampled"] is False


@pytest.mark.parametrize("role_pos,expected", list(enumerate(ROLE_ORDER)))
def test_role_position_maps_through_role_order(role_pos, expected):
    role_logits = [0.0, 0.0, 0.0]
    role_logits[role_pos] = 5.0  # peak this role
    _, role, _ = _head_with_column0([0, 0, 0, *role_logits]).select(_e0(), sample=False)
    assert role is expected


def test_zero_init_is_uniform_policy():
    head = LinearHead(n_a=6, d_h=D_H, n_models=3)
    _, _, dbg = head.select(_e0(), sample=False)
    assert dbg["agent_probs"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert dbg["role_probs"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_debug_probs_are_valid_distributions():
    _, _, dbg = _head_with_column0([1, 2, 3, 4, 5, 6]).select(_e0(), sample=False)
    assert dbg["agent_probs"].sum() == pytest.approx(1.0)
    assert dbg["role_probs"].sum() == pytest.approx(1.0)
    assert dbg["agent_logits"].tolist() == pytest.approx([1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# select: sampling (train) path
# --------------------------------------------------------------------------- #
def test_sampling_is_reproducible_with_generator():
    head = _head_with_column0([2, 1, 0, 0, 1, 2])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    out1 = [head.select(_e0(), sample=True, rng=g1)[:2] for _ in range(20)]
    out2 = [head.select(_e0(), sample=True, rng=g2)[:2] for _ in range(20)]
    assert out1 == out2  # same seed -> identical stream


def test_sampling_respects_a_peaked_distribution():
    # Agent 1 and role 2 dominate; sampling should overwhelmingly pick them.
    head = _head_with_column0([-20, 20, -20, -20, -20, 20])
    g = torch.Generator().manual_seed(0)
    picks = [head.select(_e0(), sample=True, rng=g)[:2] for _ in range(50)]
    assert all(a == 1 and r is Role.VERIFIER for a, r in picks)


# --------------------------------------------------------------------------- #
# select: input-shape guards
# --------------------------------------------------------------------------- #
def test_select_squeezes_leading_batch_of_one():
    head = _head_with_column0([1, 2, 3, 6, 5, 4])
    agent_idx, role, _ = head.select(_e0().unsqueeze(0), sample=False)  # (1, D_H)
    assert agent_idx == 2 and role is Role.THINKER


@pytest.mark.parametrize("shape", [(2, D_H), (2, 3, D_H)])
def test_select_rejects_multi_row_batches(shape):
    head = LinearHead(n_a=6, d_h=D_H, n_models=3)
    with pytest.raises(ValueError):
        head.select(torch.zeros(*shape), sample=False)


# --------------------------------------------------------------------------- #
# load_weight validation
# --------------------------------------------------------------------------- #
def test_load_weight_rejects_wrong_shape():
    head = LinearHead(n_a=6, d_h=D_H, n_models=3)
    with pytest.raises(ValueError):
        head.load_weight(np.zeros((6, D_H + 1)))


def test_load_weight_accepts_numpy_and_tensor_equally():
    W = np.arange(6 * D_H, dtype=np.float64).reshape(6, D_H)
    h_np = LinearHead(n_a=6, d_h=D_H, n_models=3)
    h_np.load_weight(W)
    h_t = LinearHead(n_a=6, d_h=D_H, n_models=3)
    h_t.load_weight(torch.from_numpy(W))
    assert torch.equal(h_np.weight, h_t.weight)
