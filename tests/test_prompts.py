"""Offline unit tests for role prompt construction (SPEC §4.4).

Exercises ``trinity.roles.prompts`` only — no LLM calls, no torch.
"""
import pytest

from trinity.roles.prompts import (
    THINKER_SYSTEM,
    VERIFIER_SYSTEM,
    WORKER_SYSTEM,
    build_messages,
    render_transcript,
)
from trinity.types import Role, TurnRecord


def _turn(
    turn: int,
    role: Role,
    processed: str,
    *,
    agent: str = "test-model",
    verdict: str | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        agent_name=agent,
        role=role,
        raw_output=processed,
        processed_output=processed,
        verdict=verdict,
    )


def test_empty_transcript_returns_sentinel():
    assert render_transcript([]) == "(no prior turns yet)"


def test_render_single_turn():
    rec = _turn(1, Role.WORKER, "  partial solution  ")
    out = render_transcript([rec])
    assert "--- Turn 1 | agent=test-model | role=WORKER ---" in out
    assert "partial solution" in out


def test_render_verifier_includes_parsed_verdict():
    rec = _turn(
        2,
        Role.VERIFIER,
        "Looks good.\nVERDICT: ACCEPT",
        verdict="ACCEPT",
    )
    out = render_transcript([rec])
    assert "[parsed verdict: ACCEPT]" in out


def test_render_multiple_turns_chronological():
    transcript = [
        _turn(1, Role.THINKER, "plan A"),
        _turn(2, Role.WORKER, "work B", agent="other-model"),
    ]
    out = render_transcript(transcript)
    assert out.index("Turn 1") < out.index("Turn 2")
    assert "agent=other-model" in out
    assert "plan A" in out and "work B" in out


@pytest.mark.parametrize(
    ("role", "expected_system"),
    [
        (Role.THINKER, THINKER_SYSTEM),
        (Role.WORKER, WORKER_SYSTEM),
        (Role.VERIFIER, VERIFIER_SYSTEM),
    ],
)
def test_build_messages_system_prompt_by_role(role, expected_system):
    msgs = build_messages(role, "What is 2+2?", [])
    assert msgs == [
        {"role": "system", "content": expected_system},
        {
            "role": "user",
            "content": "QUERY:\nWhat is 2+2?\n\nTRANSCRIPT SO FAR:\n(no prior turns yet)",
        },
    ]


def test_build_messages_user_includes_rendered_transcript():
    transcript = [_turn(1, Role.WORKER, "42")]
    msgs = build_messages(Role.VERIFIER, "Compute answer", transcript)
    user = msgs[1]["content"]
    assert user.startswith("QUERY:\nCompute answer\n\nTRANSCRIPT SO FAR:\n")
    assert "role=WORKER" in user
    assert "42" in user


def test_build_messages_returns_openai_style_pair():
    msgs = build_messages(Role.THINKER, "Q", [])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
