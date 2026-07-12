"""Benchmark dataset loaders (LiveCodeBench, IFEval, RLPR, math, reasoning, domain knowledge).

Each loader exposes load(split, **kw) -> list[Task], where a Task carries the
prompt, the reference/answer or test harness, and a score(prediction) -> float.

TODO(SPEC §6): implement loaders for the exact datasets/splits in docs/SPEC.md.

The config-facing benchmark aliases currently map to canonical dataset names as follows:

- ``ifeval`` -> Google IFEval prompts
- ``rlpr`` -> OpenBMB RLPR-Evaluation suite
"""
