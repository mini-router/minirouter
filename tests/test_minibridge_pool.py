from __future__ import annotations

import json
import asyncio

import httpx
import pytest

from trinity.llm.minibridge_client import MiniBridgePool, _parse_minibridge_response
from trinity.llm.openai_compatible_pool import ChatResult
from trinity.llm.pool_factory import build_pool


def _config(tmp_path):
    path = tmp_path / "models.minibridge.yaml"
    path.write_text(
        """
minibridge:
  base_url: "https://minibridge.example"
  provider_id: "openrouter-prod"
  caller_id: "minirouter-maintainer"
  owner_id: "minirouter-miners"
  key_id: "minirouter-miners-openrouter-key"
  timeout_s: 10
  max_retries: 1
  max_concurrency: 2
  ttl_s: 60
  ledger_provider: "openrouter"
pool:
  - name: qwen3-coder-30b
    id: "qwen/qwen3-coder-30b-a3b-instruct"
decoding: {}
""",
        encoding="utf-8",
    )
    return path


def test_parse_minibridge_response_uses_input_output_tokens() -> None:
    data = {
        "response": {
            "content": "ok",
            "usage": {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15},
            "finish_reason": "stop",
        }
    }
    result = _parse_minibridge_response(data, "m")
    assert isinstance(result, ChatResult)
    assert result.text == "ok"
    assert result.prompt_tokens == 13
    assert result.completion_tokens == 2
    assert result.finish_reason == "stop"


def test_build_pool_accepts_minibridge(tmp_path) -> None:
    pool = build_pool("minibridge", _config(tmp_path))
    assert isinstance(pool, MiniBridgePool)
    assert pool.models == {"qwen3-coder-30b": "qwen/qwen3-coder-30b-a3b-instruct"}
    assert pool.describe_model("qwen3-coder-30b") == (
        "openrouter",
        "qwen/qwen3-coder-30b-a3b-instruct",
    )


def test_minibridge_http_400_includes_response_body(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False, "error": "policy denied"})

    async def run_chat():
        pool = MiniBridgePool(_config(tmp_path))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await pool.chat(
                "qwen3-coder-30b",
                [{"role": "user", "content": "Reply ok"}],
                client=client,
            )

    with pytest.raises(RuntimeError, match="policy denied"):
        asyncio.run(run_chat())


def test_minibridge_chat_sends_policy_identity_and_ledgers(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "ok": True,
                "response": {
                    "content": "ok",
                    "usage": {"input_tokens": 7, "output_tokens": 1},
                    "finish_reason": "stop",
                },
                "receipt": {
                    "provider_id": "openrouter-prod",
                    "computed_cost_usd": "0.000001",
                },
            },
        )

    async def run_chat():
        pool = MiniBridgePool(_config(tmp_path))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await pool.chat(
                "qwen3-coder-30b",
                [{"role": "user", "content": "Reply ok"}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=8,
                reasoning="minimal",
                client=client,
            )

    result = asyncio.run(run_chat())

    assert result.text == "ok"
    assert seen_payloads
    payload = seen_payloads[0]
    assert payload["provider_id"] == "openrouter-prod"
    assert payload["caller_id"] == "minirouter-maintainer"
    assert payload["owner_id"] == "minirouter-miners"
    assert payload["key_id"] == "minirouter-miners-openrouter-key"
    assert payload["model"] == "qwen/qwen3-coder-30b-a3b-instruct"
    assert payload["parameters"]["max_tokens"] == 8
    assert payload["parameters"]["reasoning_effort"] == "low"
    assert payload["nonce"]
    assert payload["expires_at"]

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "provider": "openrouter",
            "m": "qwen/qwen3-coder-30b-a3b-instruct",
            "p": 7,
            "c": 1,
        }
    ]
