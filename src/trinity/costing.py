"""Shared cost reporting helpers for train/eval runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = [
    "COST_PRICES",
    "default_cost_ledger_path",
    "ledger_cost_report",
]

# $ / 1M tokens (prompt, completion).
# Keep this in sync with the model pools actually used in configs/.
COST_PRICES: dict[str, tuple[float, float]] = {
    # Legacy Fireworks pool.
    "fireworks:deepseek-v4-pro": (1.74, 3.48),
    "fireworks:glm-5p2": (1.40, 4.40),
    "fireworks:kimi-k2p6": (0.95, 4.00),
    # OpenRouter paid replacements.
    "openrouter:qwen/qwen3-coder-30b-a3b-instruct": (0.07, 0.27),
    "openrouter:openai/gpt-oss-120b": (0.036, 0.18),
    "openrouter:google/gemma-3-4b-it": (0.05, 0.10),
    "openrouter:google/gemma-3-27b-it": (0.08, 0.16),
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b": (0.50, 2.20),
    # Older OpenRouter models kept for backward-compatible ledgers.
    "openrouter:deepseek-v4-pro": (0.435, 0.87),
    "openrouter:kimi-k2p6": (0.66, 3.50),
    "openrouter:glm-5p2": (1.40, 4.40),
    "openrouter:nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    "openrouter:google/gemma-4-31b-it:free": (0.0, 0.0),
    "openrouter:openai/gpt-oss-120b:free": (0.0, 0.0),
    "openrouter:qwen/qwen3-coder:free": (0.0, 0.0),
    "openrouter:nvidia/nemotron-3-super-120b-a12b": (0.0, 0.0),
    "openrouter:google/gemma-4-31b-it": (0.12, 0.35),
    "openrouter:qwen/qwen3-32b": (0.08, 0.28),
    # Chutes pool.
    "chutes:deepseek-ai/DeepSeek-V3.2-TEE": (1.00, 1.00),
    "chutes:zai-org/GLM-5-TEE": (1.40, 4.40),
    "chutes:moonshotai/Kimi-K2.5-TEE": (0.66, 3.50),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": (0.15, 1.20),
    "chutes:google/gemma-4-31B-turbo-TEE": (0.12, 0.37),
    "chutes:Qwen/Qwen3-32B-TEE": (0.10, 0.42),
}


def default_cost_ledger_path(out_path: str | None) -> Path:
    if os.environ.get("TRINITY_COST_LEDGER"):
        return Path(os.environ["TRINITY_COST_LEDGER"]).expanduser()
    if out_path:
        return Path(out_path).expanduser().with_suffix(".cost_ledger.jsonl")
    return Path.cwd() / "cost_ledger.jsonl"


def ledger_cost_report(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {
            "cost_usd": 0.0,
            "cost_missing": True,
            "cost_ledger": str(ledger_path),
        }

    per_model: dict[str, dict[str, float | int]] = {}
    total = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            provider = str(row.get("provider", "")).strip()
            model = str(row.get("m", "")).strip()
            pt = int(row.get("p", 0) or 0)
            ct = int(row.get("c", 0) or 0)
            pin, pout = COST_PRICES.get(f"{provider}:{model}", (0.0, 0.0))
            usd = pt / 1e6 * pin + ct / 1e6 * pout
            total += usd
            prompt_tokens += pt
            completion_tokens += ct
            calls += 1
            bucket = per_model.setdefault(
                f"{provider}:{model}",
                {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usd": 0.0},
            )
            bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + pt
            bucket["completion_tokens"] = int(bucket["completion_tokens"]) + ct
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["usd"] = float(bucket["usd"]) + usd

    return {
        "cost_usd": round(total, 4),
        "cost_missing": False,
        "cost_ledger": str(ledger_path),
        "cost_calls": calls,
        "cost_prompt_tokens": prompt_tokens,
        "cost_completion_tokens": completion_tokens,
        "cost_per_model": {
            key: {
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "calls": int(row["calls"]),
                "usd": round(float(row["usd"]), 4),
            }
            for key, row in sorted(per_model.items())
        },
    }
