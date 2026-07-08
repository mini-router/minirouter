"""Offline unit tests for the opt-in LLM response cache (src/trinity/llm/cache.py).

Pure disk/hashing logic — no network, no GPU, and no dependency on the pool
client (so these run even where httpx/tenacity are absent).
"""
import asyncio

import pytest

from trinity.llm.cache import CACHE_SCHEMA_VERSION, ResponseCache

_REQ = dict(
    provider="fireworks",
    model_id="accounts/fireworks/models/glm-5p2",
    messages=[{"role": "user", "content": "2+2?"}],
    temperature=0.0,
    top_p=0.95,
    max_tokens=64,
    reasoning="minimal",
)
_RESULT = {
    "model": _REQ["model_id"],
    "text": "4",
    "prompt_tokens": 5,
    "completion_tokens": 1,
    "finish_reason": "stop",
}


def test_disabled_cache_is_noop():
    c = ResponseCache(None)
    assert c.enabled is False
    k = c.make_key(**_REQ)
    assert c.get(k) is None
    c.put(k, _RESULT)  # must not raise
    assert c.stats()["writes"] == 0


def test_from_env_disabled_when_unset():
    assert ResponseCache.from_env({}).enabled is False
    assert ResponseCache.from_env({"TRINITY_LLM_CACHE": "   "}).enabled is False


def test_from_env_enabled_with_path(tmp_path):
    c = ResponseCache.from_env({"TRINITY_LLM_CACHE": str(tmp_path / "cache")})
    assert c.enabled is True


def test_put_then_get_roundtrips(tmp_path):
    c = ResponseCache(tmp_path)
    k = c.make_key(**_REQ)
    assert c.get(k) is None  # miss
    c.put(k, _RESULT)
    got = c.get(k)  # hit
    assert got == _RESULT
    assert c.hits == 1 and c.misses == 1 and c.writes == 1


def test_key_is_deterministic_and_input_sensitive():
    c = ResponseCache(None)
    base = c.make_key(**_REQ)
    assert base == c.make_key(**_REQ)  # stable across calls
    # any material change flips the key
    assert base != c.make_key(**{**_REQ, "temperature": 0.7})
    assert base != c.make_key(**{**_REQ, "max_tokens": 65})
    assert base != c.make_key(**{**_REQ, "model_id": "other"})
    assert base != c.make_key(
        **{**_REQ, "messages": [{"role": "user", "content": "3+3?"}]}
    )
    # message ordering / content is part of the key
    assert base != c.make_key(
        **{**_REQ, "messages": _REQ["messages"] + [{"role": "assistant", "content": "x"}]}
    )


def test_two_caches_share_entries_on_disk(tmp_path):
    a = ResponseCache(tmp_path)
    k = a.make_key(**_REQ)
    a.put(k, _RESULT)
    # a fresh cache over the same dir sees the persisted entry
    b = ResponseCache(tmp_path)
    assert b.get(b.make_key(**_REQ)) == _RESULT


def test_corrupt_entry_is_treated_as_miss(tmp_path):
    c = ResponseCache(tmp_path)
    k = c.make_key(**_REQ)
    path = c._path_for(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json")
    assert c.get(k) is None
    assert c.misses == 1


def test_incomplete_entry_misses(tmp_path):
    import json

    c = ResponseCache(tmp_path)
    k = c.make_key(**_REQ)
    path = c._path_for(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    # missing 'text' / token fields -> must miss, not return a partial result
    path.write_text(json.dumps({"version": CACHE_SCHEMA_VERSION, "result": {"model": "x"}}))
    assert c.get(k) is None


def test_hit_fields_reconstruct_a_chat_result(tmp_path):
    """The pool builds ChatResult(cached=True, **hit); the hit must carry exactly
    the constructor fields. Verify the shape without importing the pool."""
    c = ResponseCache(tmp_path)
    k = c.make_key(**_REQ)
    c.put(k, _RESULT)
    hit = c.get(k)
    assert set(hit) == {"model", "text", "prompt_tokens", "completion_tokens", "finish_reason"}


def test_pool_chat_serves_cache_hit_without_network(tmp_path, monkeypatch):
    """End-to-end: with the cache enabled and pre-seeded, OpenAICompatiblePool.chat()
    returns the cached completion and never opens a network client."""
    pytest.importorskip("httpx")
    pytest.importorskip("yaml")
    pytest.importorskip("tenacity")
    import trinity.llm.openai_compatible_pool as P

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "providers:\n"
        "  fw:\n"
        "    base_url: http://localhost\n"
        "    api_key_env: FAKE_KEY\n"
        "pool:\n"
        "  - name: a\n"
        "    provider: fw\n"
        "    id: m1\n"
    )
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("TRINITY_LLM_CACHE", str(tmp_path / "cachedir"))

    pool = P.OpenAICompatiblePool(cfg)
    assert pool.cache.enabled

    # Any network use must blow up, proving the hit short-circuits the request.
    class _Boom:
        def __call__(self, *a, **k):
            raise AssertionError("network client created on a cache hit")

    monkeypatch.setattr(P.httpx, "AsyncClient", _Boom())

    msgs = [{"role": "user", "content": "hi"}]
    key = pool.cache.make_key(
        provider="fw", model_id="m1", messages=msgs,
        temperature=0.0, top_p=0.95, max_tokens=8, reasoning=None,
    )
    pool.cache.put(key, {
        "model": "m1", "text": "hello", "prompt_tokens": 1,
        "completion_tokens": 1, "finish_reason": "stop",
    })

    res = asyncio.run(pool.chat("a", msgs, temperature=0.0, top_p=0.95, max_tokens=8))
    assert res.cached is True
    assert res.text == "hello"
    assert pool.cache.hits == 1
