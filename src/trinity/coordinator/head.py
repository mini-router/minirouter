"""Linear coordinator head (SPEC §3.3, Eq. 5).

Maps the SLM penultimate-token hidden state ``h ∈ R^{d_h}`` to ``n_a`` logits
through a single weight matrix with **no bias and no activation**::

    z = W · h ,   W ∈ R^{n_a × d_h}

The ``n_a = L + 3`` logits split into two independent groups:

    z[:n_models]   → agent logits   (which pool LLM to call)
    z[n_models:]   → role  logits   (Thinker / Worker / Verifier)

Each group is converted to a categorical via its **own** softmax (two separate
distributions, SPEC §3.3 / §10.8). At eval we take the argmax of each group; at
train-time fitness evaluation we sample, so the optimizer sees the stochastic
policy it is optimizing (SPEC §4.3).

torch is imported inside this module (allowed per the build constraints — the
LOCAL machine has no torch; this code is written but never executed here).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trinity.types import ROLE_ORDER, Role

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class LinearHead(torch.nn.Module):
    """Bias-free linear head ``z = W·h`` over the SLM hidden state.

    Parameters
    ----------
    n_a:
        Total output width = ``n_models + n_roles``. Default 6 (3 + 3).
    d_h:
        Hidden size of the SLM (input dimension). Default 1024 (Qwen3-0.6B).
    n_models:
        Pool size ``L`` = number of agent logits. Default 3. The remaining
        ``n_a - n_models`` logits are role logits and must equal
        ``len(ROLE_ORDER)`` (3).

    Notes
    -----
    The weight is a plain ``torch.nn.Parameter`` of shape ``(n_a, d_h)``; the
    CMA-ES θ-vector is loaded into it via :meth:`load_weight`. There is no bias
    term (Eq. 5). Weights initialize to zero → uniform policy at start (matches
    ``params.initial_theta``).
    """

    def __init__(self, n_a: int = 6, d_h: int = 1024, n_models: int = 3) -> None:
        super().__init__()
        n_roles = n_a - n_models
        if n_roles != len(ROLE_ORDER):
            raise ValueError(
                f"n_a - n_models = {n_roles} role logits, but ROLE_ORDER has "
                f"{len(ROLE_ORDER)} roles ({[r.value for r in ROLE_ORDER]})"
            )
        self.n_a = n_a
        self.d_h = d_h
        self.n_models = n_models
        self.n_roles = n_roles
        # No bias, no activation (SPEC §3.3). Zero-init = uniform policy.
        self.weight = torch.nn.Parameter(torch.zeros(n_a, d_h))

    @torch.no_grad()
    def load_weight(self, W: "np.ndarray | Tensor") -> None:
        """Copy a ``(n_a, d_h)`` weight matrix into this head (in place).

        Used by the coordinator to install a candidate θ's head block. Accepts
        either a NumPy array (as produced by ``params.unpack``) or a torch
        Tensor; it is cast to the parameter's dtype/device.

        Parameters
        ----------
        W:
            Weight matrix of shape ``(n_a, d_h)``.

        Raises
        ------
        ValueError
            If ``W``'s shape does not match ``(n_a, d_h)``.
        """
        if isinstance(W, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(W))
        else:
            t = W
        if tuple(t.shape) != (self.n_a, self.d_h):
            raise ValueError(
                f"weight shape {tuple(t.shape)} != expected {(self.n_a, self.d_h)}"
            )
        self.weight.copy_(t.to(dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the two logit groups ``z = W·h``.

        Parameters
        ----------
        h:
            Hidden state of shape ``(d_h,)`` or ``(..., d_h)``. Per SPEC §0.3.2
            ``h`` is expected to be L2-normalized upstream (in ``slm.py``) so the
            logit scale is independent of ``‖h‖``; this head does not normalize.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(agent_logits, role_logits)`` where ``agent_logits`` is
            ``z[..., :n_models]`` and ``role_logits`` is ``z[..., n_models:]``.
            Leading batch dimensions of ``h`` are preserved.
        """
        # z = h @ Wᵀ  → shape (..., n_a). Works for 1-D and batched h.
        z = torch.matmul(h, self.weight.t())
        agent_logits = z[..., : self.n_models]
        role_logits = z[..., self.n_models :]
        return agent_logits, role_logits

    @torch.no_grad()
    def select(
        self,
        h: Tensor,
        *,
        sample: bool,
        rng: "torch.Generator | None" = None,
    ) -> tuple[int, Role, dict[str, Any]]:
        """Pick (agent_idx, role) from a single hidden state.

        Applies a separate softmax to each logit group, then either samples a
        categorical (``sample=True``, training fitness) or takes the argmax
        (``sample=False``, deterministic eval) — SPEC §4.3.

        Parameters
        ----------
        h:
            Hidden state for ONE turn, shape ``(d_h,)`` (a leading batch dim of
            size 1 is squeezed; larger batches are rejected — selection is a
            per-turn scalar decision).
        sample:
            If ``True`` draw from the categoricals (train); if ``False`` take
            argmax of each group (eval).
        rng:
            Optional ``torch.Generator`` for reproducible sampling. Ignored when
            ``sample=False``.

        Returns
        -------
        tuple[int, Role, dict]
            ``(agent_idx, role, logits_debug)`` where ``agent_idx`` indexes the
            pool ``[0, n_models)``, ``role`` is the selected :class:`Role`
            (mapped through ``ROLE_ORDER``), and ``logits_debug`` carries the
            raw logits and probabilities for both groups (NumPy arrays) plus the
            ``"sampled"`` flag — for logging / smoke tests.
        """
        h = h.squeeze(0) if h.dim() == 2 and h.shape[0] == 1 else h
        if h.dim() != 1:
            raise ValueError(
                f"select expects a single hidden state of shape (d_h,), got {tuple(h.shape)}"
            )

        agent_logits, role_logits = self.forward(h)
        agent_probs = torch.softmax(agent_logits, dim=-1)
        role_probs = torch.softmax(role_logits, dim=-1)

        if sample:
            agent_idx = int(torch.multinomial(agent_probs, 1, generator=rng).item())
            role_pos = int(torch.multinomial(role_probs, 1, generator=rng).item())
        else:
            agent_idx = int(torch.argmax(agent_logits, dim=-1).item())
            role_pos = int(torch.argmax(role_logits, dim=-1).item())

        role = ROLE_ORDER[role_pos]

        logits_debug: dict[str, Any] = {
            "agent_logits": agent_logits.detach().to("cpu").float().numpy(),
            "role_logits": role_logits.detach().to("cpu").float().numpy(),
            "agent_probs": agent_probs.detach().to("cpu").float().numpy(),
            "role_probs": role_probs.detach().to("cpu").float().numpy(),
            "agent_idx": agent_idx,
            "role_pos": role_pos,
            "role": role,
            "sampled": sample,
        }
        return agent_idx, role, logits_debug
