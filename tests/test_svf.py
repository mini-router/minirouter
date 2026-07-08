"""Offline unit tests for the SVF adapter (SPEC §3.4, smoke-test S2).

``coordinator/svf.py`` is the second half of the CMA-ES search vector θ: it
SVD-decomposes the 7 linear matrices of one transformer block, freezes the
orthogonal factors, and exposes only the singular-value *scales* as learnable
params — ``W' = U diag(s * scale) Vh``. SPEC S2 explicitly wants the identity
round-trip and the real scale count asserted before trusting training, but there
was **no test** (all coordinator tests need a loaded Qwen3-0.6B on a GPU).

These run offline: CPU + torch only, driven by a tiny module that mirrors the
Qwen3 layout (``model.model.layers[L].self_attn.{q,k,v,o}_proj`` +
``.mlp.{gate,up,down}_proj``) at small dims — no model download, no network.

Key SVD facts the tests pin:
  * ``set_scales(ones)`` reconstructs the original weight (identity round-trip);
  * a uniform scale ``c`` gives ``W' = c·W`` exactly (since ``U diag(c·s) Vh``);
  * ``reset()`` restores the pristine weight bit-for-bit.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from trinity.coordinator.svf import SVFAdapter  # noqa: E402

D = 8      # hidden size (small; SVF logic is dim-agnostic)
FF = 16    # MLP inner size
# Every targeted matrix has min(out, in) == D singular values, so with the full
# 7-matrix set num_scales == 7 * D.
FULL_SCALES = 7 * D


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(D, 2 * D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(2 * D, D, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(D, FF, bias=False)
        self.up_proj = nn.Linear(D, FF, bias=False)
        self.down_proj = nn.Linear(FF, D, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])


class _Model(nn.Module):
    """Minimal stand-in exposing the Qwen3 module paths SVFAdapter navigates."""

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _Inner(n_layers)


def _mk_model(n_layers: int = 4) -> _Model:
    torch.manual_seed(0)
    return _Model(n_layers)


def _weight(model: _Model, layer: int, parent: str, name: str) -> "torch.Tensor":
    return getattr(getattr(model.model.layers[layer], parent), name).weight


def _snapshot(model: _Model, layer: int) -> dict:
    paths = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
        ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ]
    return {(p, n): _weight(model, layer, p, n).detach().clone() for p, n in paths}


# --------------------------------------------------------------------------- #
# Layout: real scale count + contiguous slices
# --------------------------------------------------------------------------- #
def test_num_scales_and_contiguous_slices():
    svf = SVFAdapter(_mk_model(), target_layer=1)
    assert svf.num_scales == FULL_SCALES
    assert svf.matrix_names == (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    # Slices tile [0, num_scales) with no gaps/overlaps, each of length D.
    cursor = 0
    for name in svf.matrix_names:
        start, end = svf.scale_slices[name]
        assert (start, end) == (cursor, cursor + D)
        cursor = end
    assert cursor == svf.num_scales


def test_custom_subset_preserves_order_and_count():
    svf = SVFAdapter(_mk_model(), target_layer=0, matrices=["down_proj", "q_proj"])
    assert svf.matrix_names == ("down_proj", "q_proj")
    assert svf.num_scales == 2 * D
    assert svf.scale_slices == {"down_proj": (0, D), "q_proj": (D, 2 * D)}


def test_unknown_matrix_name_raises():
    with pytest.raises(KeyError):
        SVFAdapter(_mk_model(), matrices=["q_proj", "not_a_matrix"])


def test_identity_scales_and_describe_helpers():
    svf = SVFAdapter(_mk_model(), target_layer=1)
    ones = svf.identity_scales()
    assert ones.shape == (FULL_SCALES,) and np.all(ones == 1.0)
    d = svf.describe()
    assert d == svf.scale_slices
    d["q_proj"] = (99, 99)  # mutating the copy must not touch the adapter
    assert svf.scale_slices["q_proj"] == (0, D)


# --------------------------------------------------------------------------- #
# Reconstruction math
# --------------------------------------------------------------------------- #
def test_identity_scales_round_trip_to_original():
    """set_scales(ones) reconstructs each weight (S2 identity round-trip)."""
    model = _mk_model()
    orig = _snapshot(model, layer=1)
    svf = SVFAdapter(model, target_layer=1)
    svf.set_scales(np.ones(FULL_SCALES))
    for (parent, name), w0 in orig.items():
        torch.testing.assert_close(
            _weight(model, 1, parent, name), w0, atol=1e-4, rtol=1e-4
        )


def test_uniform_scale_multiplies_weight():
    """U diag(c·s) Vh == c·W, so a uniform scale c scales every weight by c."""
    model = _mk_model()
    orig = _snapshot(model, layer=1)
    svf = SVFAdapter(model, target_layer=1)
    svf.set_scales(np.full(FULL_SCALES, 2.0))
    for (parent, name), w0 in orig.items():
        torch.testing.assert_close(
            _weight(model, 1, parent, name), 2.0 * w0, atol=1e-4, rtol=1e-4
        )


def test_per_matrix_scale_isolation():
    """Scaling only q_proj's block leaves the other six matrices untouched."""
    model = _mk_model()
    orig = _snapshot(model, layer=1)
    svf = SVFAdapter(model, target_layer=1)
    scales = np.ones(FULL_SCALES)
    qs, qe = svf.scale_slices["q_proj"]
    scales[qs:qe] = 3.0
    svf.set_scales(scales)
    torch.testing.assert_close(
        _weight(model, 1, "self_attn", "q_proj"), 3.0 * orig[("self_attn", "q_proj")],
        atol=1e-4, rtol=1e-4,
    )
    for (parent, name), w0 in orig.items():
        if name == "q_proj":
            continue
        torch.testing.assert_close(
            _weight(model, 1, parent, name), w0, atol=1e-4, rtol=1e-4
        )


def test_reset_restores_pristine_weight():
    model = _mk_model()
    orig = _snapshot(model, layer=1)
    svf = SVFAdapter(model, target_layer=1)
    svf.set_scales(np.full(FULL_SCALES, 5.0))
    svf.reset()
    for (parent, name), w0 in orig.items():
        # reset() copies the cached pristine weight -> bit-identical.
        assert torch.equal(_weight(model, 1, parent, name), w0)


def test_set_scales_wrong_length_raises():
    svf = SVFAdapter(_mk_model(), target_layer=1)
    with pytest.raises(ValueError):
        svf.set_scales(np.ones(FULL_SCALES - 1))


def test_only_target_layer_is_modified():
    model = _mk_model(n_layers=4)
    others = {L: _snapshot(model, L) for L in (0, 2, 3)}
    svf = SVFAdapter(model, target_layer=1)
    svf.set_scales(np.full(FULL_SCALES, 2.0))
    for L, snap in others.items():
        for (parent, name), w0 in snap.items():
            assert torch.equal(_weight(model, L, parent, name), w0), (
                f"layer {L} {parent}.{name} changed but only layer 1 should"
            )
