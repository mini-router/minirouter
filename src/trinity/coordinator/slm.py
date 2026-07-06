"""Coordinator SLM: Qwen3-0.6B hidden-state extractor (SPEC §3.1-§3.2).

The coordinator reads the full conversation transcript with a frozen-ish
Qwen3-0.6B and exposes a single contextual feature vector ``h in R^1024`` to the
linear head. This module implements ONLY the encoder half: loading the model and
performing the canonical penultimate-token extraction. The SVF adaptation
(``svf.py``) wraps the ``.model`` exposed here; the head (``head.py``) consumes
the returned ``h``.

Canonical extraction (SPEC §3.2, [OUR CHOICE] mechanics, gate S1):
  1. Tokenize ``transcript_text`` (no special tokens added beyond what we control).
  2. Append the tokenizer's EOS so that a *penultimate* position exists and is
     well-defined (the last real content token sits at index -2).
  3. Single forward pass with ``output_hidden_states=True``.
  4. Take ``hidden_states[-1][0, -2, :]`` -- the FINAL hidden layer at the
     penultimate output token. NOT the last/EOS token: the paper's ablation shows
     using the last token collapses LiveCodeBench by >10 points.
  5. Cast to float32 numpy, shape (1024,), optionally L2-normalized.

L2-normalization is [OUR CHOICE] (SPEC §0.3.2): raw bf16 penultimate states can
have large norm and would saturate the softmax under sigma0=0.1 on a W=0 start,
so we normalize ``h <- h / ||h||`` to make the logit scale norm-independent.

The forward pass is deterministic: ``eval()`` mode, ``torch.no_grad()``, a fixed
input, and no sampling. Repeated calls on the same transcript return identical h
(gate S1). transformers/torch are imported lazily inside this file because the
local dev machine has no torch/GPU; only the remote H200 box loads the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "trinity.yaml"

# SPEC §0.1 / §9: Qwen3-0.6B verified config. We assert against the loaded
# checkpoint at runtime rather than trusting these blindly.
_EXPECTED_HIDDEN_SIZE = 1024
_EXPECTED_NUM_LAYERS = 28


class CoordinatorEncoder:
    """Qwen3-0.6B penultimate-token hidden-state extractor (SPEC §3.2).

    Loads the tokenizer + ``AutoModelForCausalLM`` once (with
    ``output_hidden_states=True``), puts the model in ``eval()`` mode, and
    extracts a single float32 feature vector per transcript. The loaded model and
    tokenizer are exposed as ``.model`` / ``.tokenizer`` so :mod:`trinity.coordinator.svf`
    can wrap the second-to-last layer's linear matrices in place.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to ``"Qwen/Qwen3-0.6B"`` (CONFIRMED
        ``hidden_size=1024``, 28 layers).
    device:
        Torch device string. Defaults to ``"cuda:0"`` which is physical GPU 5 via
        ``CUDA_VISIBLE_DEVICES=5`` (SPEC §0.1 / config).
    dtype:
        Torch dtype name (``"bfloat16"`` | ``"float16"`` | ``"float32"``).
    l2_normalize:
        If True (default, [OUR CHOICE] SPEC §0.3.2), L2-normalize ``h`` before
        returning so the head's logit scale is independent of ``||h||``.

    Attributes
    ----------
    hidden_size:
        ``config.hidden_size`` of the loaded model (asserted == 1024).
    num_layers:
        ``config.num_hidden_layers`` of the loaded model (expected 28).
    model:
        The loaded ``AutoModelForCausalLM`` (eval mode, requires_grad off).
    tokenizer:
        The loaded tokenizer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        l2_normalize: bool = True,
    ) -> None:
        # Lazy, file-local imports: the dev machine has no torch/GPU.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.l2_normalize = bool(l2_normalize)

        self._torch = torch
        self._dtype = self._resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            output_hidden_states=True,
        )
        self.model.to(device)
        self.model.eval()
        # The orthogonal SVD factors and head are not trained via autograd
        # (sep-CMA-ES is derivative-free); disable grads on the whole SLM.
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden_size: int = int(cfg.hidden_size)
        self.num_layers: int = int(getattr(cfg, "num_hidden_layers", _EXPECTED_NUM_LAYERS))

        # SPEC §3.1 / gate S1: verify against the actual checkpoint, do not trust.
        assert self.hidden_size == _EXPECTED_HIDDEN_SIZE, (
            f"Expected hidden_size={_EXPECTED_HIDDEN_SIZE} (d_h), "
            f"loaded model reports {self.hidden_size}. "
            "Recompute head dims and n if this differs (SPEC §10 #18)."
        )

        # A stable EOS id to append so a penultimate position always exists.
        self._eos_id: int = self._resolve_eos_id()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _resolve_dtype(self, dtype: str):
        """Map a dtype name to a torch dtype."""
        torch = self._torch
        table = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        key = str(dtype).lower()
        if key not in table:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(table)}")
        return table[key]

    def _resolve_eos_id(self) -> int:
        """Pick a stable EOS token id from tokenizer/model config.

        Falls back through tokenizer.eos_token_id -> model.config.eos_token_id
        -> pad_token_id so the appended terminator is always defined.
        """
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = getattr(self.model.config, "eos_token_id", None)
        # Qwen configs sometimes store a list of EOS ids; take the first.
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if eos is None:
            eos = self.tokenizer.pad_token_id
        if eos is None:
            raise RuntimeError(
                "Could not resolve an EOS/pad token id for the tokenizer; "
                "penultimate-token extraction needs an appendable terminator."
            )
        return int(eos)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def encode(self, transcript_text: str) -> np.ndarray:
        """Return the penultimate-token hidden state for ``transcript_text``.

        Implements the SPEC §3.2 canonical extraction: tokenize the transcript,
        append a single EOS token so a penultimate position exists, run one
        forward pass, and read ``hidden_states[-1][0, -2, :]`` (final layer,
        penultimate output token).

        Parameters
        ----------
        transcript_text:
            The full conversation transcript ``s = concat(C_{k-1})`` to encode.

        Returns
        -------
        numpy.ndarray
            float32 vector of shape ``(hidden_size,) == (1024,)``, L2-normalized
            iff ``self.l2_normalize`` is set. Deterministic across calls.
        """
        torch = self._torch

        # Tokenize WITHOUT auto special tokens so we control the sequence layout,
        # then append exactly one EOS as the final position. The last real
        # content token therefore sits at index -2 (the <Head Input> position).
        enc = self.tokenizer(
            transcript_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        eos = torch.tensor([[self._eos_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, eos], dim=1)

        attention_mask = torch.ones_like(input_ids)
        if input_ids.shape[1] < 2:
            raise ValueError(
                "Need at least 2 tokens (content + EOS) for a penultimate "
                f"position; got sequence length {input_ids.shape[1]}."
            )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states is a tuple of (num_layers + 1) tensors, each
        # (batch, seq_len, hidden_size); [-1] is the final layer. Index -2 along
        # the sequence is the penultimate (NOT EOS) output token.
        h = out.hidden_states[-1][0, -2, :]
        h = h.to(torch.float32)

        if self.l2_normalize:
            norm = torch.linalg.vector_norm(h)
            # Guard against a degenerate all-zero hidden state.
            h = h / norm if float(norm) > 0.0 else h

        return h.detach().cpu().numpy().astype(np.float32, copy=False)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = _DEFAULT_CONFIG,
    ) -> "CoordinatorEncoder":
        """Construct from ``configs/trinity.yaml`` ``coordinator`` block.

        Reads ``encoder_model``, ``device``, ``dtype`` and
        ``hidden_state.l2_normalize`` so callers stay in sync with the spec
        config without duplicating literals.
        """
        cfg = yaml.safe_load(Path(config_path).read_text())
        coord = cfg["coordinator"]
        hs = coord.get("hidden_state", {})
        return cls(
            model_name=coord.get("encoder_model", "Qwen/Qwen3-0.6B"),
            device=coord.get("device", "cuda:0"),
            dtype=coord.get("dtype", "bfloat16"),
            l2_normalize=bool(hs.get("l2_normalize", True)),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CoordinatorEncoder(model_name={self.model_name!r}, "
            f"device={self.device!r}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, l2_normalize={self.l2_normalize})"
        )
