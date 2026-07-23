"""RLPR benchmark facade.

This module keeps the config-facing benchmark entrypoint in one place while the
actual loading logic stays in ``trinity.orchestration.dataset``.
"""
from __future__ import annotations

from trinity.orchestration.dataset import load_tasks as _load_tasks

__all__ = ["load", "load_tasks"]


def load(
    split: str,
    *,
    max_items: int | None = None,
    seed: int = 0,
    allow_toy_fallback: bool = False,
):
    """Load RLPR tasks for the requested split."""
    return load_tasks(
        "rlpr",
        split,
        max_items=max_items,
        seed=seed,
    )


def load_tasks(
    split: str,
    *,
    max_items: int | None = None,
    seed: int = 0,
    allow_toy_fallback: bool = False,
):
    """Alias for config/codepaths that expect a ``load_tasks`` symbol."""
    return _load_tasks(
        "rlpr",
        split,
        max_items=max_items,
        seed=seed,
    )
