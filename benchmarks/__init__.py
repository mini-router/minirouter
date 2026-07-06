"""Benchmark dataset loaders (LiveCodeBench, math, reasoning, domain knowledge).

Each loader exposes load(split, **kw) -> list[Task], where a Task carries the
prompt, the reference/answer or test harness, and a score(prediction) -> float.

TODO(SPEC §6): implement loaders for the exact datasets/splits in docs/SPEC.md.
"""
