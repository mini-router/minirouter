"""Opt-in on-disk cache for LLM chat completions.

Training the coordinator with sep-CMA-ES re-queries the pool with *identical*
``(question, model, sampling params)`` tuples across generations, and eval reruns
replay the same held-out set. Every such call is a paid Fireworks/Chutes request.
This module memoizes completions to disk so a repeated request is served locally
for free, which both cuts spend (the repo tracks "every dollar of API spend") and
makes reruns deterministic/reproducible.

The cache is **disabled by default** and has zero effect on behavior unless a
caller opts in — either by constructing :class:`ResponseCache` explicitly or by
setting the ``TRINITY_LLM_CACHE`` environment variable to a directory path. When
disabled, :meth:`get`/:meth:`put` are cheap no-ops.

Design notes
------------
* Keying is content-addressed: a stable SHA-256 over the request fields that
  change the response (provider, model id, the full message list, and the
  decoding params). Two byte-identical requests share a cache entry; anything
  different (a changed prompt, temperature, or model) misses.
* Entries are individual JSON files under the cache directory, so the store is
  human-inspectable, append-only in practice, and safe to delete wholesale.
* A cache hit is *free*: it must not append to the cost ledger. Callers can read
  :attr:`ResponseCache.hits` / :attr:`misses` for accounting.
* Enabling the cache makes ``temperature > 0`` requests deterministic across
  runs (the first sampled completion is replayed). That is the intended
  behavior for cost/repro, but it means the cache should stay off during a
  sampling-noise study that relies on fresh draws each run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Bump when the on-disk entry format changes so stale entries miss cleanly.
CACHE_SCHEMA_VERSION = 1

_ENV_VAR = "TRINITY_LLM_CACHE"

# Fields persisted from a completion. Kept in sync with ChatResult so a hit can
# rebuild an equivalent object without importing that class here (avoids a cycle).
_RESULT_FIELDS = ("model", "text", "prompt_tokens", "completion_tokens", "finish_reason")


def _stable_json(obj: Any) -> str:
    """Serialize ``obj`` deterministically for hashing (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ResponseCache:
    """Content-addressed disk cache for chat completions.

    Args:
        directory: Where entries live. ``None`` (the default) yields a
            **disabled** cache whose methods are no-ops.
    """

    def __init__(self, directory: str | Path | None = None):
        self._dir: Path | None = Path(directory) if directory else None
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ResponseCache":
        """Build a cache from ``TRINITY_LLM_CACHE`` (a directory path).

        An unset or empty variable produces a disabled cache, so importing/using
        the pool without opting in costs nothing.
        """
        source = os.environ if env is None else env
        path = (source.get(_ENV_VAR) or "").strip()
        return cls(path or None)

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    # -- keying --------------------------------------------------------------
    def make_key(
        self,
        *,
        provider: str,
        model_id: str,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning: str | None,
    ) -> str:
        """Return the content-addressed key for one request.

        The key spans exactly the inputs that determine a completion; two calls
        that would receive the same server response hash to the same key.
        """
        payload = {
            "v": CACHE_SCHEMA_VERSION,
            "provider": provider,
            "model_id": model_id,
            "messages": messages,
            "temperature": round(float(temperature), 6),
            "top_p": round(float(top_p), 6),
            "max_tokens": int(max_tokens),
            "reasoning": reasoning,
        }
        digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
        return digest

    def _path_for(self, key: str) -> Path:
        assert self._dir is not None  # guarded by callers via `enabled`
        # Shard by the first two hex chars to avoid one giant directory.
        return self._dir / key[:2] / f"{key}.json"

    # -- read / write --------------------------------------------------------
    def get(self, key: str) -> dict | None:
        """Return the persisted result fields for ``key``, or ``None`` on miss.

        A corrupt or unreadable entry is treated as a miss (and left in place for
        inspection) rather than raising into the hot request path.
        """
        if self._dir is None:
            return None
        path = self._path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text())
            result = entry["result"]
            # Require every expected field so a schema drift misses cleanly.
            if not all(f in result for f in _RESULT_FIELDS):
                self.misses += 1
                return None
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return result

    def put(self, key: str, result: dict) -> None:
        """Persist ``result`` (the completion fields) under ``key``.

        Writes atomically via a temp file + rename so a crash mid-write cannot
        leave a half-written entry that would later be read as a hit.
        """
        if self._dir is None:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"version": CACHE_SCHEMA_VERSION, "key": key, "result": result}
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(_stable_json(entry))
            tmp.replace(path)
            self.writes += 1
        except OSError:
            # Best-effort cache; a write failure must not break the request.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def stats(self) -> dict[str, int | bool]:
        """Return a small accounting snapshot (for logs / reports)."""
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "lookups": total,
        }
