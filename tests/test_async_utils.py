"""Offline unit tests for bounded async fan-out (``orchestration.async_utils``).

Uses ``asyncio.run`` — no pytest-asyncio plugin required.
"""
import asyncio

import pytest

from trinity.orchestration.async_utils import gather_in_batches


async def _identity(value):
    return value


async def _boom():
    raise ValueError("boom")


def test_empty_input_returns_empty_list():
    assert asyncio.run(gather_in_batches([], batch_size=4)) == []


def test_preserves_order_across_batches():
    coros = [_identity(i) for i in range(7)]
    out = asyncio.run(gather_in_batches(coros, batch_size=3))
    assert out == list(range(7))


def test_batch_size_clamped_to_at_least_one():
    coros = [_identity(i) for i in range(3)]
    out = asyncio.run(gather_in_batches(coros, batch_size=0))
    assert out == [0, 1, 2]


def test_single_batch_when_size_exceeds_length():
    coros = [_identity("a"), _identity("b")]
    out = asyncio.run(gather_in_batches(coros, batch_size=100))
    assert out == ["a", "b"]


def test_return_exceptions_false_propagates():
    with pytest.raises(ValueError, match="boom"):
        asyncio.run(gather_in_batches([_boom()], batch_size=1, return_exceptions=False))


def test_return_exceptions_true_collects_errors_in_order():
    coros = [_identity(1), _boom(), _identity(3)]
    out = asyncio.run(gather_in_batches(coros, batch_size=2, return_exceptions=True))
    assert out[0] == 1
    assert isinstance(out[1], ValueError)
    assert out[2] == 3
