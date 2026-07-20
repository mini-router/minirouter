"""MiniBridge client for the coordinated LLM pool.

MiniBridge is not an OpenAI-compatible endpoint. It wraps upstream provider calls
with caller/key policy checks and returns a signed receipt. This pool adapts that
request/response shape to the same small interface used by the router:
``models``, ``describe_model()``, and async ``chat()``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..envfile import load_project_env
from .openai_compatible_pool import ChatResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "models.minibridge.light.yaml"


@dataclass(frozen=True)
class MiniBridgeSpec:
    base_url: str
    provider_id: str
    caller_id: str
    owner_id: str
    key_id: str
    timeout_s: float = 60.0
    max_retries: int = 3
    max_concurrency: int = 8
    ttl_s: int = 300
    ledger_provider: str = "openrouter"
    reasoning_param: str | None = "reasoning_effort"


@dataclass(frozen=True)
class MiniBridgeModelSpec:
    name: str
    model_id: str


class _Retryable(Exception):
    """Wrap transient MiniBridge/provider failures for tenacity retries."""


def _ledger_append(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    path = os.environ.get("TRINITY_COST_LEDGER")
    if not path:
        return
    try:
        record = json.dumps(
            {
                "provider": provider,
                "m": model,
                "p": int(prompt_tokens),
                "c": int(completion_tokens),
            }
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(record + "\n")
    except Exception:
        pass


def _env_or_config(env_name: str, raw: dict, key: str, default: str = "") -> str:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value.strip()
    value = raw.get(key, default)
    return str(value).strip()


def _expires_at(ttl_s: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_s)))).isoformat()


def _parse_minibridge_response(data: dict, model: str) -> ChatResult:
    response = data.get("response") or {}
    usage = response.get("usage") or {}
    prompt_tokens = int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    content = response.get("content")
    return ChatResult(
        model=model,
        text="" if content is None else str(content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=response.get("finish_reason"),
        raw=data,
    )


class MiniBridgePool:
    """Async client over one MiniBridge runner and a logical model pool."""

    _REASONING_MAP = {
        "minimal": "low",
        "low": "low",
        "none": "none",
        "medium": "medium",
        "high": "high",
    }

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG):
        load_project_env(repo_root=_REPO_ROOT)
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.spec = self._load_spec(cfg)
        self.routes = self._load_routes(cfg)
        self.models: dict[str, str] = {name: route.model_id for name, route in self.routes.items()}
        self.decoding: dict = cfg.get("decoding", {})
        self._sem = asyncio.Semaphore(self.spec.max_concurrency)

    def _load_spec(self, cfg: dict) -> MiniBridgeSpec:
        raw = cfg.get("minibridge") or {}
        if not isinstance(raw, dict):
            raise ValueError("models.minibridge.yaml must define a minibridge mapping")
        base_url = _env_or_config("MINIBRIDGE_URL", raw, "base_url")
        if not base_url:
            raise RuntimeError("MINIBRIDGE_URL is not set and minibridge.base_url is empty")
        return MiniBridgeSpec(
            base_url=base_url.rstrip("/"),
            provider_id=_env_or_config("MINIBRIDGE_PROVIDER_ID", raw, "provider_id", "openrouter-prod"),
            caller_id=_env_or_config("MINIBRIDGE_CALLER_ID", raw, "caller_id", "minirouter-maintainer"),
            owner_id=_env_or_config("MINIBRIDGE_OWNER_ID", raw, "owner_id", "minirouter-miners"),
            key_id=_env_or_config("MINIBRIDGE_KEY_ID", raw, "key_id", "minirouter-miners-openrouter-key"),
            timeout_s=float(raw.get("timeout_s", 60)),
            max_retries=int(raw.get("max_retries", 3)),
            max_concurrency=int(raw.get("max_concurrency", 8)),
            ttl_s=int(raw.get("ttl_s", 300)),
            ledger_provider=str(raw.get("ledger_provider", "openrouter")),
            reasoning_param=raw.get("reasoning_param", "reasoning_effort"),
        )

    def _load_routes(self, cfg: dict) -> dict[str, MiniBridgeModelSpec]:
        pool = cfg.get("pool")
        if not isinstance(pool, list) or not pool:
            raise ValueError("models.minibridge.yaml must define a non-empty pool list")
        routes: dict[str, MiniBridgeModelSpec] = {}
        for item in pool:
            if not isinstance(item, dict):
                raise ValueError("each pool entry must be a mapping")
            name = str(item["name"])
            routes[name] = MiniBridgeModelSpec(name=name, model_id=str(item["id"]))
        return routes

    def model_id(self, name: str) -> str:
        if name in self.models:
            return self.models[name]
        if name in self.models.values():
            return name
        raise KeyError(f"Unknown model '{name}'. Known: {list(self.models)}")

    def _resolve_route(self, model: str) -> MiniBridgeModelSpec:
        if model in self.routes:
            return self.routes[model]
        for route in self.routes.values():
            if model == route.model_id:
                return route
        raise KeyError(f"Unknown model '{model}'. Known: {list(self.models)}")

    def describe_model(self, model: str) -> tuple[str, str]:
        route = self._resolve_route(model)
        return self.spec.ledger_provider, route.model_id

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        reasoning: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> ChatResult:
        route = self._resolve_route(model)
        spec = self.spec
        trace = os.environ.get("TRINITY_TRACE_LLM", "").strip() not in {"", "0", "false", "False"}
        parameters: dict[str, object] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if reasoning is not None and spec.reasoning_param:
            parameters[spec.reasoning_param] = self._REASONING_MAP.get(reasoning, reasoning)
        payload = {
            "request_id": f"minirouter-{uuid.uuid4().hex}",
            "provider_id": spec.provider_id,
            "caller_id": spec.caller_id,
            "owner_id": spec.owner_id,
            "key_id": spec.key_id,
            "model": route.model_id,
            "messages": messages,
            "parameters": parameters,
            "metadata": {
                "source": "minirouter",
                "logical_model": route.name,
            },
            "nonce": f"minirouter-{uuid.uuid4().hex}",
            "expires_at": _expires_at(spec.ttl_s),
        }
        if trace:
            print(
                f"[llm] -> provider=minibridge upstream={spec.provider_id} "
                f"model={route.model_id} caller={spec.caller_id} max_tokens={max_tokens}",
                flush=True,
            )

        @retry(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(spec.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )
        async def _do(cli: httpx.AsyncClient) -> ChatResult:
            async with self._sem:
                t0 = time.perf_counter()
                try:
                    resp = await cli.post(
                        f"{spec.base_url}/call",
                        json=payload,
                        timeout=spec.timeout_s,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if trace:
                        print(
                            f"[llm] !! provider=minibridge model={route.model_id} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    raise _Retryable(f"network: {type(exc).__name__}: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                if trace:
                    print(
                        f"[llm] !! provider=minibridge model={route.model_id} "
                        f"http={resp.status_code} retryable",
                        flush=True,
                    )
                raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                body = resp.text[:500]
                if trace:
                    print(
                        f"[llm] !! provider=minibridge model={route.model_id} "
                        f"http={resp.status_code} body={body}",
                        flush=True,
                    )
                raise RuntimeError(f"MiniBridge HTTP {resp.status_code}: {body}")
            data = resp.json()
            if not data.get("ok", False):
                raise RuntimeError(f"MiniBridge call failed: {data.get('error', data)}")
            result = _parse_minibridge_response(data, route.model_id)
            if trace:
                elapsed = time.perf_counter() - t0
                receipt = data.get("receipt") or {}
                print(
                    f"[llm] <- provider=minibridge upstream={receipt.get('provider_id', spec.provider_id)} "
                    f"model={route.model_id} status={resp.status_code} sec={elapsed:.1f} "
                    f"pt={result.prompt_tokens} ct={result.completion_tokens} "
                    f"cost={receipt.get('computed_cost_usd', '?')} "
                    f"finish={result.finish_reason} content_empty={result.text == ''}",
                    flush=True,
                )
            _ledger_append(
                spec.ledger_provider,
                route.model_id,
                result.prompt_tokens,
                result.completion_tokens,
            )
            return result

        if client is not None:
            return await _do(client)
        async with httpx.AsyncClient() as cli:
            return await _do(cli)


async def _selftest() -> int:
    pool = MiniBridgePool()
    print(f"Pool: {list(pool.models)}")
    async with httpx.AsyncClient() as cli:
        results = await asyncio.gather(
            *[
                pool.chat(
                    name,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=8,
                    temperature=0.0,
                    client=cli,
                )
                for name in pool.models
            ],
            return_exceptions=True,
        )
    ok = True
    for name, res in zip(pool.models, results):
        if isinstance(res, Exception):
            ok = False
            print(f"  [FAIL] {name}: {res!r}")
        else:
            print(f"  [ OK ] {name:20s} -> {res.text.strip()[:40]!r} ({res.completion_tokens} toks)")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniBridge pool client")
    ap.add_argument("--selftest", action="store_true", help="ping all pool models through MiniBridge")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(asyncio.run(_selftest()))
    ap.print_help()


if __name__ == "__main__":
    main()
