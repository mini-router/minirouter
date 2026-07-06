"""θ ↔ (head W, SVF singular-value scales) packing for sep-CMA-ES.

The trainable parameter vector ``θ`` that the optimizer searches over is the
concatenation of two blocks (SPEC §0.2 / §3.4 — Section 0 is authoritative):

    1. the linear head weight matrix  ``W ∈ R^{n_a × d_h}``   (6 × 1024 = 6,144)
    2. the SVF singular-value scales of layer 26's 7 linear matrices
       (7 × 1024 = 7,168)

giving a flat vector of length ``n = 6,144 + 7,168 = 13,312``.

Layout (contiguous, head block first):

    θ = [ W.flatten()  |  svf_scales ]
        └── 6,144 ──┘   └── 7,168 ──┘

The head block is stored in row-major (C) order so ``W.reshape(head_shape)``
round-trips exactly. All packing is done in ``float64`` because CMA-ES operates
in double precision; the SLM downcasts to bf16 only at load time.

These are pure, deterministic functions with no torch / GPU dependency so they
can be unit-tested on any machine (smoke test S3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults grounded in SPEC §0 (authoritative) and configs/trinity.yaml.
DEFAULT_N_A: int = 6          # head outputs: 3 agent logits + 3 role logits (L=3)
DEFAULT_D_H: int = 1024       # Qwen3-0.6B hidden_size (CONFIRMED)
DEFAULT_N_SVF: int = 7168     # 7 linear matrices of layer 26 × 1024 SVs each


@dataclass(frozen=True)
class ParamSpec:
    """Immutable description of the θ-vector layout for CMA-ES.

    Attributes
    ----------
    head_shape:
        Shape of the linear head weight ``W`` as ``(n_a, d_h)`` — e.g. ``(6, 1024)``.
    n_head:
        Number of head parameters = ``n_a * d_h`` = 6,144.
    n_svf:
        Number of SVF singular-value scales = 7,168.
    n_total:
        Total search dimension ``n`` = ``n_head + n_svf`` = 13,312.
    """

    head_shape: tuple[int, int]
    n_head: int
    n_svf: int
    n_total: int

    def __post_init__(self) -> None:
        n_a, d_h = self.head_shape
        expected_head = n_a * d_h
        if self.n_head != expected_head:
            raise ValueError(
                f"n_head ({self.n_head}) != prod(head_shape) ({expected_head})"
            )
        if self.n_total != self.n_head + self.n_svf:
            raise ValueError(
                f"n_total ({self.n_total}) != n_head + n_svf "
                f"({self.n_head} + {self.n_svf} = {self.n_head + self.n_svf})"
            )


def make_spec(
    n_a: int = DEFAULT_N_A,
    d_h: int = DEFAULT_D_H,
    n_svf: int = DEFAULT_N_SVF,
) -> ParamSpec:
    """Build the canonical :class:`ParamSpec`.

    Parameters
    ----------
    n_a:
        Head output width = ``L + 3`` (3 agent + 3 role logits). Default 6.
    d_h:
        SLM hidden size. Default 1024 (Qwen3-0.6B, verified against config).
    n_svf:
        SVF singular-value-scale count. Default 7,168 (7 × 1024). The smoke test
        S2 must confirm this against the loaded checkpoint before trusting it.

    Returns
    -------
    ParamSpec
        With ``head_shape=(n_a, d_h)``, ``n_head=n_a*d_h``, and
        ``n_total=n_head + n_svf``.
    """
    n_head = n_a * d_h
    return ParamSpec(
        head_shape=(n_a, d_h),
        n_head=n_head,
        n_svf=n_svf,
        n_total=n_head + n_svf,
    )


def pack(head_W: np.ndarray, svf_scales: np.ndarray) -> np.ndarray:
    """Flatten (head W, SVF scales) into a single ``float64`` θ-vector.

    Parameters
    ----------
    head_W:
        Head weight matrix, shape ``(n_a, d_h)`` (or any shape; it is flattened
        row-major). Must be 2-D.
    svf_scales:
        1-D array of SVF singular-value scales.

    Returns
    -------
    np.ndarray
        Flat ``float64`` vector of length ``head_W.size + svf_scales.size``,
        laid out as ``[W.flatten(), svf_scales]``.

    Notes
    -----
    A fresh contiguous array is always returned (no view aliasing of the inputs),
    so the optimizer may mutate θ freely.
    """
    head_W = np.asarray(head_W)
    svf_scales = np.asarray(svf_scales)
    if head_W.ndim != 2:
        raise ValueError(f"head_W must be 2-D, got shape {head_W.shape}")
    if svf_scales.ndim != 1:
        raise ValueError(f"svf_scales must be 1-D, got shape {svf_scales.shape}")
    head_flat = np.ascontiguousarray(head_W, dtype=np.float64).ravel(order="C")
    svf_flat = np.ascontiguousarray(svf_scales, dtype=np.float64).ravel(order="C")
    return np.concatenate([head_flat, svf_flat], axis=0)


def unpack(theta: np.ndarray, spec: ParamSpec) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat θ-vector back into (head W, SVF scales).

    Inverse of :func:`pack` given a matching :class:`ParamSpec`.

    Parameters
    ----------
    theta:
        Flat parameter vector of length ``spec.n_total``.
    spec:
        Layout produced by :func:`make_spec`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(head_W, svf_scales)`` where ``head_W`` has shape ``spec.head_shape``
        and ``svf_scales`` has shape ``(spec.n_svf,)``. Both are fresh
        ``float64`` arrays (copies), so mutating them does not touch ``theta``.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64).ravel(order="C")
    if theta.size != spec.n_total:
        raise ValueError(
            f"theta has {theta.size} elements, expected n_total={spec.n_total}"
        )
    head_flat = theta[: spec.n_head]
    svf_flat = theta[spec.n_head :]
    head_W = head_flat.reshape(spec.head_shape).copy()
    svf_scales = svf_flat.copy()
    return head_W, svf_scales


def initial_theta(spec: ParamSpec) -> np.ndarray:
    """Return the CMA-ES initial mean ``m_0`` (SPEC §0.2 / config init_mean).

    Head ``W = 0``  → uniform policy (every agent/role equally likely after the
    two softmaxes). SVF scales ``= 1.0`` → identity adaptation, so the SLM is
    unmodified at the start of training.

    Parameters
    ----------
    spec:
        Layout produced by :func:`make_spec`.

    Returns
    -------
    np.ndarray
        Flat ``float64`` θ-vector of length ``spec.n_total`` =
        ``[zeros(n_head), ones(n_svf)]``.
    """
    head_W = np.zeros(spec.head_shape, dtype=np.float64)
    svf_scales = np.ones(spec.n_svf, dtype=np.float64)
    return pack(head_W, svf_scales)
