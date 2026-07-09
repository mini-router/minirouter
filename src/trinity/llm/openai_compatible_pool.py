"""OpenAI-compatible client for the coordinated LLM pool."""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..envfile import load_project_env

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ProviderSpec:
    """One OpenAI-compatible backend."""

    name: str
    base_url: str
    api_key_env: str
    timeout_s: float = 120.0
    max_retries: int = 4
    max_concurrency: int = 8
    reasoning_param: str | None = "reasoning_effort"
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """A logical pool model routed through a named provider."""

    name: str
    provider: str
    model_id: str


@dataclass
class ChatResult:
    """One completion plus the accounting we need for fitness/cost terms."""

    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class _Retryable(Exception):
    """Wraps transient HTTP failures so tenacity retries them."""


def _ledger_append(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Append one token-usage record to the cost ledger, if TRINITY_COST_LEDGER is set."""
    path = os.environ.get("TRINITY_COST_LEDGER")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(
                f'{{"provider":"{provider}","m":"{model}","p":{int(prompt_tokens)},'
                f'"c":{int(completion_tokens)}}}\n'
            )
    except Exception:
        pass


def _parse_completion(data: dict, model: str) -> ChatResult:
    """Build a :class:`ChatResult` from an OpenAI-compatible chat response.

    Fail-safe against valid-but-unexpected ``200`` payloads. Some providers
    return HTTP 200 with an **empty** ``choices`` list (a content-filter / safety
    block) or an ``{"error": {...}}`` envelope instead of a completion. Indexing
    ``data["choices"][0]`` on those raised ``IndexError`` / ``KeyError`` straight
    out of :meth:`OpenAICompatiblePool.chat` and aborted the whole eval run.

    We treat a missing/empty choice (or a missing ``message``) as an empty
    completion (``text=""``, ``finish_reason="error"``), mirroring the existing
    null-``content`` handling so a single odd reply degrades one turn instead of
    crashing the run. See JOURNAL 2026-07-08.
    """
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    choices = data.get("choices") or []
    if not choices:
        return ChatResult(
            model=model,
            text="",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="error",
            raw=data,
        )
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    return ChatResult(
        model=model,
        text="" if content is None else str(content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


class OpenAICompatiblePool:
    """Async-first client over one or more OpenAI-compatible chat endpoints."""

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG):
        load_project_env(repo_root=_REPO_ROOT)
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.providers = self._load_providers(cfg)
        self.routes: dict[str, ModelSpec] = self._load_routes(cfg)
        self.models: dict[str, str] = {name: route.model_id for name, route in self.routes.items()}
        self.decoding: dict = cfg.get("decoding", {})
        self._sems: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(spec.max_concurrency) for name, spec in self.providers.items()
        }
        self._headers: dict[str, dict[str, str]] = {
            name: self._build_headers(spec) for name, spec in self.providers.items()
        }

    def _load_providers(self, cfg: dict) -> dict[str, ProviderSpec]:
        providers_cfg = cfg.get("providers")
        if providers_cfg is None and "fireworks" in cfg:
            fw = cfg["fireworks"]
            providers_cfg = {
                "fireworks": {
                    "base_url": fw["base_url"],
                    "api_key_env": fw.get("api_key_env", "FIREWORKS_API_KEY"),
                    "timeout_s": fw.get("timeout_s", 120),
                    "max_retries": fw.get("max_retries", 4),
                    "max_concurrency": fw.get("max_concurrency", 8),
                    "reasoning_param": "reasoning_effort",
                    "extra_headers": fw.get("extra_headers", {}),
                }
            }
        if not isinstance(providers_cfg, dict) or not providers_cfg:
            raise ValueError("configs/models.yaml must define a providers block")

        providers: dict[str, ProviderSpec] = {}
        for name, raw in providers_cfg.items():
            if not isinstance(raw, dict):
                raise ValueError(f"provider {name!r} must be a mapping")
            providers[name] = ProviderSpec(
                name=name,
                base_url=str(raw["base_url"]).rstrip("/"),
                api_key_env=str(raw.get("api_key_env", "")),
                timeout_s=float(raw.get("timeout_s", 120)),
                max_retries=int(raw.get("max_retries", 4)),
                max_concurrency=int(raw.get("max_concurrency", 8)),
                reasoning_param=raw.get("reasoning_param", "reasoning_effort"),
                extra_headers=dict(raw.get("extra_headers", {}) or {}),
            )
        return providers

    def _load_routes(self, cfg: dict) -> dict[str, ModelSpec]:
        pool = cfg.get("pool")
        if not isinstance(pool, list) or not pool:
            raise ValueError("configs/models.yaml must define a non-empty pool list")
        routes: dict[str, ModelSpec] = {}
        for item in pool:
            if not isinstance(item, dict):
                raise ValueError("each pool entry must be a mapping")
            name = str(item["name"])
            provider = str(item.get("provider", "fireworks"))
            model_id = str(item["id"])
            if provider not in self.providers:
                raise ValueError(
                    f"pool entry {name!r} references unknown provider {provider!r}"
                )
            routes[name] = ModelSpec(name=name, provider=provider, model_id=model_id)
        return routes

    def _build_headers(self, provider: ProviderSpec) -> dict[str, str]:
        api_key = os.environ.get(provider.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{provider.api_key_env} is not set. Put it in {(_REPO_ROOT / 'secrets.env')} "
                f"or source ~/.config/trinity/secrets.env"
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(provider.extra_headers)
        return headers

    def model_id(self, name: str) -> str:
        if name in self.models:
            return self.models[name]
        if name in self.models.values():
            return name
        raise KeyError(f"Unknown model '{name}'. Known: {list(self.models)}")

    def _resolve_route(self, model: str) -> ModelSpec:
        if model in self.routes:
            return self.routes[model]
        for route in self.routes.values():
            if model == route.model_id:
                return route
        raise KeyError(f"Unknown model '{model}'. Known: {list(self.models)}")

    _REASONING_MAP = {
        "minimal": "low",
        "low": "low",
        "none": "none",
        "medium": "medium",
        "high": "high",
    }

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
        provider = self.providers[route.provider]
        headers = self._headers[provider.name]
        trace = os.environ.get("TRINITY_TRACE_LLM", "").strip() not in {"", "0", "false", "False"}
        payload = {
            "model": route.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if reasoning is not None and provider.reasoning_param:
            payload[provider.reasoning_param] = self._REASONING_MAP.get(reasoning, reasoning)
        if trace:
            print(
                f"[llm] -> provider={provider.name} model={payload['model']} "
                f"max_tokens={max_tokens} reasoning={reasoning}",
                flush=True,
            )

        @retry(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(provider.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        )
        async def _do(cli: httpx.AsyncClient) -> ChatResult:
            async with self._sems[provider.name]:
                t0 = time.perf_counter()
                try:
                    resp = await cli.post(
                        f"{provider.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=provider.timeout_s,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if trace:
                        print(
                            f"[llm] !! provider={provider.name} model={payload['model']} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    raise _Retryable(f"network: {type(exc).__name__}: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                if trace:
                    print(
                        f"[llm] !! provider={provider.name} model={payload['model']} "
                        f"http={resp.status_code} retryable",
                        flush=True,
                    )
                raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            result = _parse_completion(data, payload["model"])
            pt = result.prompt_tokens
            ct = result.completion_tokens
            if trace:
                elapsed = time.perf_counter() - t0
                print(
                    f"[llm] <- provider={provider.name} model={payload['model']} "
                    f"status={resp.status_code} sec={elapsed:.1f} pt={pt} ct={ct} "
                    f"finish={result.finish_reason} content_empty={result.text == ''}",
                    flush=True,
                )
            _ledger_append(provider.name, payload["model"], pt, ct)
            return result

        if client is not None:
            return await _do(client)
        async with httpx.AsyncClient() as cli:
            return await _do(cli)


async def _selftest() -> int:
    pool = OpenAICompatiblePool()
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
            print(f"  [ OK ] {name:16s} -> {res.text.strip()[:40]!r} "
                  f"({res.completion_tokens} toks)")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible pool client")
    ap.add_argument("--selftest", action="store_true", help="ping all pool models")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(asyncio.run(_selftest()))
    ap.print_help()


if __name__ == "__main__":
    main()
