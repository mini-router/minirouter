"""Role-specific prompt construction for the TRINITY coordination loop.

Implements the §4.4 system-prompt templates (THINKER / WORKER / VERIFIER) and a
readable transcript renderer. The coordinator's head picks the agent + role; this
module turns that choice plus the running transcript into OpenAI-style chat
messages that are sent to the selected Fireworks pool model.

Design notes (per SPEC §4.4 / §4.5):
- The system message carries the role contract; the user message carries the
  query and the rendered transcript so far. This keeps the role instruction in
  the highest-priority slot while the (potentially long) transcript lives in the
  user turn.
- The instruction text is preserved word-for-word from SPEC §4.4. The {Q} /
  {C_prev} blocks named in the spec are supplied via the user message, so the
  system prompt holds only the stable role contract.
- No LLM calls happen here; this is pure, deterministic string assembly.
"""
from __future__ import annotations

from trinity.types import Role, TurnRecord

__all__ = [
    "THINKER_SYSTEM",
    "WORKER_SYSTEM",
    "VERIFIER_SYSTEM",
    "render_transcript",
    "build_messages",
]


# --- §4.4 system prompts (role contracts) -----------------------------------
# Verbatim instruction text from SPEC §4.4. The {Q}/{C_prev} blocks named in the
# spec are supplied via the user message (see build_messages), so the system
# prompt holds only the stable role contract.

THINKER_SYSTEM = (
    "You are the THINKER. Do NOT solve the task end-to-end.\n"
    "Analyze the current state and produce meta-level guidance: a concise high-level\n"
    "plan, a decomposition into subgoals, or a critique of the partial solution so far.\n"
    "You may recommend which role should act next.\n"
    "Return only your plan/critique."
)

WORKER_SYSTEM = (
    "You are the WORKER. Make concrete progress toward the final answer.\n"
    "Follow any plan in the transcript. Produce actionable content: the derivation,\n"
    "the code, or the numerical/final result. Be explicit and complete.\n"
    "Return your solution work."
)

VERIFIER_SYSTEM = (
    "You are the VERIFIER. Check whether the accumulated solution is correct, complete,\n"
    "and responsive to the query.\n"
    "End your response with EXACTLY one line:\n"
    "VERDICT: ACCEPT      (if the current answer is correct and final)\n"
    "VERDICT: REVISE      (otherwise, with a one-line diagnosis above it)"
)

_SYSTEM_BY_ROLE: dict[Role, str] = {
    Role.THINKER: THINKER_SYSTEM,
    Role.WORKER: WORKER_SYSTEM,
    Role.VERIFIER: VERIFIER_SYSTEM,
}


def render_transcript(transcript: list[TurnRecord]) -> str:
    """Render the running transcript into a readable plain-text block.

    Each prior turn is shown with its 1-indexed turn number, the agent that
    produced it, the role it played, and the post-processed output (``O_k``).
    Verifier turns also surface their parsed verdict. The processed (not raw)
    output is used because that is what is appended to the transcript per the
    inner-loop contract (SPEC §4.2 step 5).

    Args:
        transcript: prior turns ``O_1 .. O_{k-1}`` in chronological order.

    Returns:
        A human/LLM-readable string. Returns a fixed sentinel when empty so the
        downstream model is never handed a blank transcript section.
    """
    if not transcript:
        return "(no prior turns yet)"

    blocks: list[str] = []
    for rec in transcript:
        header = (
            f"--- Turn {rec.turn} | agent={rec.agent_name} | "
            f"role={rec.role.value.upper()} ---"
        )
        body = rec.processed_output.strip()
        if rec.role is Role.VERIFIER and rec.verdict is not None:
            body = f"{body}\n[parsed verdict: {rec.verdict}]"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_messages(
    role: Role,
    query: str,
    transcript: list[TurnRecord],
) -> list[dict]:
    """Build OpenAI-style chat messages for the selected role.

    Produces the §4.4 message layout: a system message holding the role contract
    and a user message holding the query plus the rendered transcript so far.

    Args:
        role: the role the next agent will play (THINKER / WORKER / VERIFIER).
        query: the original user query ``Q``.
        transcript: prior turns ``O_1 .. O_{k-1}`` (may be empty on turn 1).

    Returns:
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``
        suitable for ``FireworksPool.chat``.

    Raises:
        KeyError: if ``role`` is not one of the three known roles.
    """
    system = _SYSTEM_BY_ROLE[role]
    rendered = render_transcript(transcript)
    user = f"QUERY:\n{query}\n\nTRANSCRIPT SO FAR:\n{rendered}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
