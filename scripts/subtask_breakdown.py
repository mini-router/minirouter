#!/usr/bin/env python3
"""Aggregate a per-item eval breakdown into a per-subtask accuracy table.

Some benchmarks pool many distinct subtasks into a single score -- BBH pools
27 (`boolean_expressions`, `causal_judgement`, `word_sorting`, ...). That
pooled average can hide large per-subtask swings: a coordinator might ace
multi-step arithmetic and fail completely at temporal reasoning while still
posting a middling overall number. `trinity.eval --breakdown-out PATH` writes
one JSON record per item (`task_id`, `benchmark`, `subtask`, `score`); this
script re-groups those records by `subtask` (falling back to `benchmark` for
items with no subtask, e.g. math500/mmlu) and reports mean accuracy + item
count per group, so the pooled number can be inspected instead of trusted
blindly.

Run:
    python -m trinity.eval --benchmark bbh --theta best_theta.npy \
        --breakdown-out experiments/bbh/run/breakdown.json
    python scripts/subtask_breakdown.py experiments/bbh/run/breakdown.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_records(path: str) -> list[dict]:
    """Load and validate a breakdown JSON file (a list of per-item records)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of per-item records, got {type(data)}")
    return data


def group_by_subtask(records: list[dict]) -> dict[str, list[float]]:
    """Group numeric scores by `subtask`, falling back to `benchmark`.

    Records with a non-numeric/missing score are silently skipped (the same
    "unscoreable" convention the reward checkers use elsewhere in this repo).
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        key = rec.get("subtask") or rec.get("benchmark") or "unknown"
        score = rec.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            groups[key].append(float(score))
    return dict(groups)


def render(groups: dict[str, list[float]]) -> str:
    """Render a Markdown table: one row per subtask, plus a pooled total."""
    out = ["# Per-subtask breakdown\n"]
    out.append("| subtask | n | accuracy |")
    out.append("|---|---|---|")
    overall: list[float] = []
    for key in sorted(groups):
        scores = groups[key]
        overall.extend(scores)
        acc = sum(scores) / len(scores) if scores else 0.0
        out.append(f"| {key} | {len(scores)} | {acc:.3f} |")
    if overall:
        pooled = sum(overall) / len(overall)
        out.append(f"| **overall (pooled)** | **{len(overall)}** | **{pooled:.3f}** |")
    return "\n".join(out) + "\n"


def summarize(groups: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    """JSON-serializable per-subtask summary: `{subtask: {n, accuracy}}`."""
    return {
        key: {"n": len(scores), "accuracy": (sum(scores) / len(scores)) if scores else 0.0}
        for key, scores in groups.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("breakdown_json", help="path written by `trinity.eval --breakdown-out`")
    ap.add_argument("--json", action="store_true", help="also print the raw per-subtask JSON summary")
    args = ap.parse_args()

    records = load_records(args.breakdown_json)
    groups = group_by_subtask(records)
    print(render(groups))
    if args.json:
        print(json.dumps(summarize(groups), indent=2))


if __name__ == "__main__":
    main()
