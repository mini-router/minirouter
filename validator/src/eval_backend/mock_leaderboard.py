from __future__ import annotations

from datetime import datetime, timezone

from .schemas import LeaderboardEntry


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


MOCK_LEADERBOARD_ENTRIES: list[LeaderboardEntry] = [
    LeaderboardEntry(
        rank=1,
        submission_id="mock-tinyrouter",
        team="TinyRouter",
        miner_id="TinyRouter",
        accuracy=0.858,
        gsm8k=None,
        mmlu=0.925,
        math=0.792,
        humaneval=None,
        bbh=None,
        params=10240,
        submitted=_dt("2026-07-07T00:00:00Z"),
        report="#",
        status="mock",
    ),
    LeaderboardEntry(
        rank=2,
        submission_id="mock-deepseek-v4-pro",
        team="deepseek-v4-pro",
        miner_id="deepseek-v4-pro",
        accuracy=0.835,
        gsm8k=None,
        mmlu=0.922,
        math=0.747,
        humaneval=None,
        bbh=None,
        params=10240,
        submitted=_dt("2026-07-07T00:00:00Z"),
        report="#",
        status="mock",
    ),
    LeaderboardEntry(
        rank=3,
        submission_id="mock-random-routing",
        team="random routing",
        miner_id="random routing",
        accuracy=0.833,
        gsm8k=None,
        mmlu=0.875,
        math=0.792,
        humaneval=None,
        bbh=None,
        params=10240,
        submitted=_dt("2026-07-07T00:00:00Z"),
        report="#",
        status="mock",
    ),
    LeaderboardEntry(
        rank=4,
        submission_id="mock-glm-5p2",
        team="glm-5p2",
        miner_id="glm-5p2",
        accuracy=0.789,
        gsm8k=None,
        mmlu=0.783,
        math=0.794,
        humaneval=None,
        bbh=None,
        params=10240,
        submitted=_dt("2026-07-07T00:00:00Z"),
        report="#",
        status="mock",
    ),
    LeaderboardEntry(
        rank=5,
        submission_id="mock-kimi-k2p6",
        team="kimi-k2p6",
        miner_id="kimi-k2p6",
        accuracy=0.640,
        gsm8k=None,
        mmlu=0.539,
        math=0.742,
        humaneval=None,
        bbh=None,
        params=10240,
        submitted=_dt("2026-07-07T00:00:00Z"),
        report="#",
        status="mock",
    ),
]
