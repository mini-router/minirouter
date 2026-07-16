from __future__ import annotations

from trinity.llm.openai_compatible_pool import OpenAICompatiblePool


class _Pool(OpenAICompatiblePool):
    def __init__(self):
        # Bypass YAML/network init; only exercise decoding_for.
        self.decoding = {
            "thinker": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
            "worker": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096},
            "verifier": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
        }


def test_decoding_for_uses_role_block():
    pool = _Pool()
    d = pool.decoding_for("worker")
    assert d["temperature"] == 0.2
    assert d["max_tokens"] == 4096


def test_decoding_for_falls_back_when_role_missing():
    pool = _Pool()
    pool.decoding = {}
    d = pool.decoding_for("worker", temperature=0.0, top_p=1.0, max_tokens=4096)
    assert d == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096}
