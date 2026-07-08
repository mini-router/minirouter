"""Offline unit tests for the OpenAI-compatible pool CLI self-test entrypoint.

`main()` calls `sys.exit(asyncio.run(_selftest()))`, so the module must import
`sys`. These tests exercise the `--selftest` CLI path with `_selftest` stubbed,
so they make NO live API calls and need no GPU/network.
"""
import asyncio
import sys

import pytest

import trinity.llm.openai_compatible_pool as pool_mod


def test_module_imports_sys():
    """Regression guard: main() references sys.exit(...), so the module must
    have `sys` in scope. Before the fix this attribute was missing and the
    self-test crashed with NameError instead of exiting."""
    assert hasattr(pool_mod, "sys")


def test_main_selftest_exits_with_status_code(monkeypatch):
    """`--selftest` should propagate the self-test's return code via sys.exit,
    not raise NameError partway through."""
    async def fake_selftest():
        return 0

    monkeypatch.setattr(pool_mod, "_selftest", fake_selftest)
    monkeypatch.setattr(sys, "argv", ["openai_compatible_pool", "--selftest"])

    with pytest.raises(SystemExit) as exc_info:
        pool_mod.main()
    assert exc_info.value.code == 0


def test_main_selftest_propagates_failure_code(monkeypatch):
    """A failing self-test (return code 1) must exit(1), confirming the code
    threads through sys.exit rather than being swallowed or crashing."""
    async def failing_selftest():
        return 1

    monkeypatch.setattr(pool_mod, "_selftest", failing_selftest)
    monkeypatch.setattr(sys, "argv", ["openai_compatible_pool", "--selftest"])

    with pytest.raises(SystemExit) as exc_info:
        pool_mod.main()
    assert exc_info.value.code == 1
