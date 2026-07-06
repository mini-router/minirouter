"""Compact ~0.6B encoder: extract the hidden-state feature for a query/turn.

Runs locally on GPU 5 (CUDA_VISIBLE_DEVICES=5). Loads the encoder model once,
then for each turn returns the contextual feature vector that feeds the head.

TODO(SPEC §3): confirm exact encoder model, which layer, which token position
(penultimate per the abstract), and any pooling. Implement after docs/SPEC.md.
"""
from __future__ import annotations

# Placeholder — implemented once docs/SPEC.md pins the encoder details.
