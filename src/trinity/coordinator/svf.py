"""Singular Value Fine-tuning (SVF) adapter for the TRINITY coordinator.

SVF (Transformer^2 / Sun et al. 2025) is the second half of the CMA-ES search
vector ``theta``. We SVD-decompose a fixed set of linear weight matrices in the
**second-to-last** transformer block of the local Qwen3-0.6B coordinator
(``model.model.layers[target_layer]``), freeze the orthogonal factors ``U`` and
``Vh``, and expose only the singular-value *scales* as learnable parameters.

For each targeted matrix ``W`` with SVD ``W = U @ diag(s) @ Vh`` we reconstruct

    W' = U @ diag(s * scale_block) @ Vh

where ``scale_block`` is a slice of the flat ``scales`` vector, initialized to
all-ones (identity, so the SLM is unmodified at the start of training).

Verified facts for Qwen3-0.6B (docs/SPEC.md §0, authoritative):
  - ``hidden_size = 1024``, ``num_hidden_layers = 28`` -> second-to-last = 26.
  - Per-layer linear matrices and their SVD singular-value counts (= min dim):
      q_proj  1024x2048 -> 1024 SVs
      k_proj  1024x1024 -> 1024 SVs
      v_proj  1024x1024 -> 1024 SVs
      o_proj  2048x1024 -> 1024 SVs
      gate_proj 1024x3072 -> 1024 SVs
      up_proj   1024x3072 -> 1024 SVs
      down_proj 3072x1024 -> 1024 SVs
    All 7 matrices give 1024 SVs each -> 7,168 SVF scales.

This module DELIBERATELY does NOT hardcode 7,168 anywhere that matters:
``num_scales`` is computed from the real SVD shapes on the loaded checkpoint, so
smoke-test S2 can print/assert the genuine count.

Attn projections live under ``layer.self_attn.{q,k,v,o}_proj``; the MLP matrices
live under ``layer.mlp.{gate,up,down}_proj`` (standard Qwen3 module layout).

The ``torch`` import is intentionally kept inside this file (the local dev box
has no torch / no GPU); nothing here is import-tested locally.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch on dev box
    import torch
    import torch.nn as nn


# Canonical ordering of the SVF target matrices and the sub-module that owns each.
# (matrix_name, parent_attribute) where parent_attribute is relative to the layer.
_DEFAULT_TARGETS: tuple[tuple[str, str], ...] = (
    ("q_proj", "self_attn"),
    ("k_proj", "self_attn"),
    ("v_proj", "self_attn"),
    ("o_proj", "self_attn"),
    ("gate_proj", "mlp"),
    ("up_proj", "mlp"),
    ("down_proj", "mlp"),
)


class _Factor:
    """Frozen SVD factors + the live module reference for one targeted matrix.

    The SVD is computed in float32 (for numerical fidelity) and then cast to the
    model's working dtype/device. We retain the original weight ``w0`` only for a
    cheap ``reset`` path and for round-trip assertions.
    """

    __slots__ = ("name", "parent_attr", "module", "U", "s", "Vh", "w0", "n_sv")

    def __init__(
        self,
        name: str,
        parent_attr: str,
        module: "nn.Linear",
        U: "torch.Tensor",
        s: "torch.Tensor",
        Vh: "torch.Tensor",
        w0: "torch.Tensor",
    ) -> None:
        self.name = name
        self.parent_attr = parent_attr
        self.module = module
        self.U = U          # (out, r)  frozen
        self.s = s          # (r,)      frozen singular values
        self.Vh = Vh        # (r, in)   frozen
        self.w0 = w0        # (out, in) original weight (for reset / assertions)
        self.n_sv = int(s.shape[0])


class SVFAdapter:
    """SVD-based singular-value adapter over one transformer block.

    Parameters
    ----------
    model:
        A loaded HF Qwen3 causal-LM (e.g. ``AutoModelForCausalLM.from_pretrained``).
        We index ``model.model.layers[target_layer]`` and adapt the listed
        matrices in place.
    target_layer:
        0-indexed layer to adapt. Default 26 (second-to-last of 28).
    matrices:
        Ordered matrix names to SVF. Default = all 7 Qwen3 linear projections.
        Order defines the layout of the flat ``scales`` vector and ``scale_slices``.

    Attributes
    ----------
    num_scales:
        Total number of learnable singular-value scales = sum of singular-value
        counts across all targeted matrices. Computed from REAL shapes (expected
        7168 for the default 7-matrix Qwen3 set, but never hardcoded).
    scale_slices:
        ``dict[str, tuple[int, int]]`` mapping matrix name -> ``(start, end)``
        half-open index range into the flat ``scales`` vector.
    """

    def __init__(
        self,
        model: object,
        target_layer: int = 26,
        matrices: "list[str] | tuple[str, ...] | None" = None,
    ) -> None:
        import torch  # local import: no torch on the dev box

        self.model = model
        self.target_layer = int(target_layer)

        # Resolve the requested matrix names to (name, parent_attr) pairs while
        # preserving the caller's ordering.
        parent_of = {name: parent for name, parent in _DEFAULT_TARGETS}
        if matrices is None:
            targets = list(_DEFAULT_TARGETS)
        else:
            targets = []
            for name in matrices:
                if name not in parent_of:
                    raise KeyError(
                        f"Unknown SVF matrix '{name}'. "
                        f"Known: {sorted(parent_of)}"
                    )
                targets.append((name, parent_of[name]))
        self.matrix_names: tuple[str, ...] = tuple(name for name, _ in targets)

        layer = model.model.layers[self.target_layer]

        self._factors: list[_Factor] = []
        self.scale_slices: dict[str, tuple[int, int]] = {}

        cursor = 0
        for name, parent_attr in targets:
            parent = getattr(layer, parent_attr)
            module = getattr(parent, name)  # nn.Linear

            w = module.weight  # (out_features, in_features)
            device = w.device
            dtype = w.dtype

            # SVD in float32 on the weight's device for numerical fidelity, then
            # cast the frozen factors back to the model's working dtype.
            w32 = w.detach().to(torch.float32)
            U, s, Vh = torch.linalg.svd(w32, full_matrices=False)

            factor = _Factor(
                name=name,
                parent_attr=parent_attr,
                module=module,
                U=U.to(device=device, dtype=dtype).contiguous(),
                s=s.to(device=device, dtype=dtype).contiguous(),
                Vh=Vh.to(device=device, dtype=dtype).contiguous(),
                w0=w.detach().clone(),
            )
            self._factors.append(factor)

            start, end = cursor, cursor + factor.n_sv
            self.scale_slices[name] = (start, end)
            cursor = end

        self.num_scales: int = cursor

    # ------------------------------------------------------------------ #
    # Reconstruction
    # ------------------------------------------------------------------ #
    def _reconstruct(self, factor: _Factor, scale_block: "torch.Tensor") -> "torch.Tensor":
        """Reconstruct ``W' = U @ diag(s * scale_block) @ Vh`` in working dtype.

        The contraction is done in float32 then cast back, so a unit scale block
        round-trips to the original weight within bf16 tolerance.
        """
        import torch

        U = factor.U.to(torch.float32)
        s = factor.s.to(torch.float32)
        Vh = factor.Vh.to(torch.float32)
        scaled_s = s * scale_block.to(device=s.device, dtype=torch.float32)
        # (out, r) * (r,) broadcasts column-wise, equivalent to U @ diag(scaled_s).
        w_prime = (U * scaled_s.unsqueeze(0)) @ Vh
        return w_prime.to(dtype=factor.module.weight.dtype)

    def set_scales(self, scales: np.ndarray) -> None:
        """Apply a flat scale vector to every targeted matrix in place (no grad).

        Parameters
        ----------
        scales:
            1-D array of length ``num_scales``. Slice ``scale_slices[name]`` is
            applied (multiplicatively, ``s * scale_block``) to matrix ``name``.

        Each live ``module.weight`` is overwritten with the reconstructed
        ``W'`` under ``torch.no_grad()``.
        """
        import torch

        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        if scales.shape[0] != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scales, got {scales.shape[0]}."
            )

        with torch.no_grad():
            for factor in self._factors:
                start, end = self.scale_slices[factor.name]
                block = torch.as_tensor(
                    scales[start:end], dtype=torch.float32, device=factor.s.device
                )
                w_prime = self._reconstruct(factor, block)
                factor.module.weight.copy_(w_prime)

    def reset(self) -> None:
        """Restore every targeted matrix to its original weight (scales = 1.0).

        Uses the cached pristine ``w0`` to avoid SVD reconstruction drift, so the
        model is bit-identical to load time after a reset.
        """
        import torch

        with torch.no_grad():
            for factor in self._factors:
                factor.module.weight.copy_(factor.w0)

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def identity_scales(self) -> np.ndarray:
        """Return the all-ones scale vector (identity / unmodified SLM)."""
        return np.ones(self.num_scales, dtype=np.float64)

    def describe(self) -> dict[str, tuple[int, int]]:
        """Return a copy of the matrix-name -> (start, end) slice map."""
        return dict(self.scale_slices)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SVFAdapter(target_layer={self.target_layer}, "
            f"matrices={list(self.matrix_names)}, num_scales={self.num_scales})"
        )
