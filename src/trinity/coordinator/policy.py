"""CoordinatorPolicy: ties the SLM encoder + SVF + linear head into one decision.

A policy is configured by a flat parameter vector θ (the thing sep-CMA-ES optimizes):
θ unpacks into (head weight W, SVF singular-value scales). `configure(θ)` writes both
into the live torch modules; `decide(transcript_text)` then runs the SLM forward, reads
the penultimate hidden state, and selects (agent_idx, role).

This module imports torch/transformers and only runs on the GPU box. The orchestration
session loop does NOT import torch — it just calls `policy.decide(...)`, so the loop can
be tested with a mock policy off-GPU (smoke test S4).
"""
from __future__ import annotations

import numpy as np

from ..types import Role
from . import params as _params


class CoordinatorPolicy:
    def __init__(self, encoder, svf, head, *, n_models: int = 3):
        self.encoder = encoder
        self.svf = svf
        self.head = head
        self.n_models = n_models
        self.spec: _params.ParamSpec | None = None

    @classmethod
    def build(
        cls,
        *,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        target_layer: int = 26,
        svf_matrices: list[str] | None = None,
        n_models: int = 3,
        n_roles: int = 3,
        l2_normalize: bool = True,
    ) -> tuple["CoordinatorPolicy", _params.ParamSpec]:
        """Load Qwen3-0.6B on the GPU, wrap SVF, build the head, return (policy, spec).

        Only call on the GPU box. The real SVF scale count is read from the loaded
        checkpoint (NOT hardcoded) — the spec is built from it.
        """
        from .head import LinearHead
        from .slm import CoordinatorEncoder
        from .svf import SVFAdapter

        encoder = CoordinatorEncoder(
            model_name=model_name, device=device, dtype=dtype, l2_normalize=l2_normalize
        )
        d_h = encoder.hidden_size
        assert d_h == 1024, f"expected d_h=1024, got {d_h}"

        svf = SVFAdapter(
            encoder.model,
            target_layer=target_layer,
            matrices=svf_matrices,
        )
        n_svf = int(svf.num_scales)  # real count from the checkpoint
        n_a = n_models + n_roles

        head = LinearHead(n_a=n_a, d_h=d_h, n_models=n_models).to(device)
        spec = _params.make_spec(n_a=n_a, d_h=d_h, n_svf=n_svf)
        policy = cls(encoder, svf, head, n_models=n_models)
        policy.spec = spec
        return policy, spec

    def configure(self, theta: np.ndarray, spec: _params.ParamSpec | None = None) -> None:
        """Load θ into the head weight + SVF scales (constant for a whole trajectory)."""
        spec = spec or self.spec
        if spec is None:
            raise RuntimeError("policy.spec is unset; pass spec or use build()")
        head_W, svf_scales = _params.unpack(theta, spec)
        self.head.load_weight(head_W)
        self.svf.set_scales(svf_scales)

    def decide(self, transcript_text: str, *, sample: bool = False, rng=None) -> tuple[int, Role]:
        import torch

        h = self.encoder.encode(transcript_text)  # np.float32 [d_h]
        h_t = torch.as_tensor(np.asarray(h, dtype=np.float32), device=self.head.weight.device)
        agent_idx, role, _dbg = self.head.select(h_t, sample=sample, rng=rng)
        return int(agent_idx), role
