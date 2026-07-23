"""Helper for selecting the pool backend at runtime."""
from __future__ import annotations

from pathlib import Path

from .chutes_client import ChutesPool
from .fireworks_client import FireworksPool
from .minibridge_client import MiniBridgePool
from .openai_compatible_pool import OpenAICompatiblePool
from .openrouter_client import OpenRouterPool

PoolName = str


def build_pool(provider: PoolName, config_path: str | Path | None = None):
    provider = (provider or "fireworks").strip().lower()
    path = Path(config_path) if config_path is not None else None
    if provider in {"compatible", "openai-compatible", "openai_compatible"}:
        return OpenAICompatiblePool(path) if path is not None else OpenAICompatiblePool()
    if provider == "fireworks":
        return FireworksPool(path) if path is not None else FireworksPool()
    if provider == "openrouter":
        if path is None or path.name == "models.yaml":
            return OpenRouterPool()
        return OpenRouterPool(path)
    if provider == "chutes":
        if path is None or path.name == "models.yaml":
            return ChutesPool()
        return ChutesPool(path)
    if provider == "minibridge":
        return MiniBridgePool(path) if path is not None else MiniBridgePool()
    raise ValueError(
        f"unknown provider {provider!r}; expected fireworks|openrouter|chutes|minibridge|compatible"
    )
