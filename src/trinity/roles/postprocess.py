"""Light post-processing of a model output ``M_k`` into ``O_k`` (SPEC §4.5).

Per SPEC §4.5 the default post-processing is **pass-through with light
truncation**: keep ``M_k`` verbatim but cap it at a fixed character budget to
bound transcript growth across the up-to-K turns. No extra summarizer LLM call
is made (this keeps atomic-eval cost predictable).

When an output exceeds the budget we keep a head and a tail and elide the middle,
so both the early framing (plans, problem restatement) and the late conclusion
(final answer, ``VERDICT:`` line for verifiers) survive truncation. Keeping the
tail matters: the Verifier's verdict and the Worker's final answer live at the
end, and a head-only truncation would silently drop them.

Pure / deterministic / no LLM calls.
"""
from __future__ import annotations

from trinity.types import Role

__all__ = ["postprocess", "ELISION_MARKER"]

# Inserted between the kept head and tail when an output is truncated.
ELISION_MARKER = "\n... [truncated] ...\n"


def postprocess(raw: str, role: Role, max_chars: int = 8000) -> str:
    """Post-process a raw model output into the transcript output ``O_k``.

    Pass-through with light head+tail truncation (SPEC §4.5). The ``role`` is
    accepted for API symmetry and future role-specific handling; the default
    policy is identical across roles (verifier verdicts are parsed separately by
    ``trinity.roles.verifier``).

    Args:
        raw: the raw model output ``M_k``.
        role: the role the producing agent played (THINKER / WORKER / VERIFIER).
        max_chars: character budget for ``O_k``. Non-positive values disable
            truncation (full pass-through).

    Returns:
        The post-processed output ``O_k``: ``raw`` stripped of surrounding
        whitespace, truncated to ``max_chars`` (head + tail) when over budget.
    """
    del role  # Unused today; kept for a stable, role-aware public signature.

    text = raw.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = ELISION_MARKER
    # If even the marker does not fit, hard-truncate from the head.
    if max_chars <= len(marker):
        return text[:max_chars]

    budget = max_chars - len(marker)
    # Bias slightly toward the tail so the final answer / verdict is preserved.
    head_len = budget // 2
    tail_len = budget - head_len
    head = text[:head_len]
    tail = text[len(text) - tail_len:]
    return f"{head}{marker}{tail}"
