#!/usr/bin/env python3
"""TRINITY Fireworks cost tracker.

Two modes:
  --ledger PATH   Exact: sum token usage recorded by FireworksPool (set
                  TRINITY_COST_LEDGER=PATH when running train/eval).
  --estimate      Approximate: estimate calls/tokens/cost from run configs, for runs
                  that happened BEFORE the ledger existed (the early pilots + the
                  currently in-flight parallel runs).

Fireworks has no usable billing API for this key (we probed: /v1/usage 404,
/v1/accounts/usage 403). But every chat response carries exact token counts, so we
price from tokens. PRICES below are ASSUMPTIONS — replace with the real per-model
Fireworks rates to get exact dollars; pass --in/--out to override the blended rate.

    python scripts/cost_report.py --estimate
    TRINITY_COST_LEDGER=~/trinity/cost_ledger.jsonl python -m trinity.train ...
    python scripts/cost_report.py --ledger ~/trinity/cost_ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

# ---- Fireworks serverless prices ($ per 1M tokens), (input, output). ----
# Real rates from Fireworks pricing (web search, June 2026):
#   deepseek-v4-pro $1.74 in / $3.48 out ($0.145 cached in)
#   kimi-k2p6 (K2.6) $0.95 in / $4.00 out ($0.16 cached in)
#   glm-5p2: GLM 5.1 listed at $1.40 / $4.40 — used as proxy for GLM 5.2 (5.2 not separately listed).
# Cached input tokens get ~50% off; we ignore caching here (slight over-estimate).
PRICES = {
    "deepseek-v4-pro": (1.74, 3.48),
    "glm-5p2":         (1.40, 4.40),
    "kimi-k2p6":       (0.95, 4.00),
}
_DEFAULT_BLENDED_IN = sum(p[0] for p in PRICES.values()) / len(PRICES)
_DEFAULT_BLENDED_OUT = sum(p[1] for p in PRICES.values()) / len(PRICES)


def cost(prompt_tok: int, completion_tok: int, in_rate: float, out_rate: float) -> float:
    return prompt_tok / 1e6 * in_rate + completion_tok / 1e6 * out_rate


def report_ledger(path: str) -> None:
    per = {}  # model -> [prompt, completion, calls]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("m", "?")
            acc = per.setdefault(m, [0, 0, 0])
            acc[0] += r.get("p", 0)
            acc[1] += r.get("c", 0)
            acc[2] += 1
    total = 0.0
    print(f"{'model':18s} {'calls':>8s} {'prompt_tok':>12s} {'compl_tok':>12s} {'$':>9s}")
    print("-" * 64)
    for m, (p, c, n) in sorted(per.items()):
        ir, orr = PRICES.get(m, (_DEFAULT_BLENDED_IN, _DEFAULT_BLENDED_OUT))
        d = cost(p, c, ir, orr)
        total += d
        print(f"{m:18s} {n:8d} {p:12d} {c:12d} {d:9.3f}")
    print("-" * 64)
    print(f"{'TOTAL (exact tokens, ASSUMED prices)':40s} ${total:.2f}")


# Per-run config estimates. avg_turns < max_turns due to early Verifier-ACCEPT.
# prompt grows with transcript; completion ~ fills max_tokens for reasoning models.
RUNS = [
    # name, generations, popsize, m_cma, max_turns, max_tokens, status
    ("pilot#1 (math)",       5, 6, 4, 3, 1024, "done"),
    ("pilot_crn (math)",     3, 8, 8, 3,  640, "done"),
    ("full_pilot (math)",   12, 8, 8, 3,  640, "done"),
    ("eval math500",         1, 0, 0, 0,    0, "done(special)"),
    ("eval mmlu",            1, 0, 0, 0,    0, "done(special)"),
    ("mmlu_pilot",          12, 8, 8, 3,  640, "running"),
    ("math_s0",             14, 8,10, 4,  768, "running"),
    ("math_s1",             14, 8,10, 4,  768, "running"),
    ("mmlu_s0",             14, 8,10, 4,  768, "running"),
    ("mmlu_s1",             14, 8,10, 4,  768, "running"),
]
_EVAL_CALLS = 40 * (3 * 1 + 2.5 + 2.5)  # 40 items: 3 single + TRINITY + random


def report_estimate(in_rate: float, out_rate: float, avg_turn_frac: float = 0.7,
                     avg_prompt: int = 650) -> None:
    print(f"Estimate @ blended ${in_rate:.2f}/1M in, ${out_rate:.2f}/1M out "
          f"(avg_turns={avg_turn_frac:g}*max, avg_prompt~{avg_prompt} tok, "
          f"completion~max_tokens):\n")
    print(f"{'run':22s} {'calls':>7s} {'Mtok':>7s} {'$':>8s}  status")
    print("-" * 60)
    grand_calls = grand_tok = grand_cost = 0.0
    for name, g, p, m, t, mt, status in RUNS:
        if "eval" in name:
            calls = _EVAL_CALLS
            compl = mt or 640
            ptok = calls * avg_prompt
            ctok = calls * 640
        else:
            avg_turns = max(1.0, avg_turn_frac * t)
            calls = g * p * m * avg_turns
            ptok = calls * avg_prompt
            ctok = calls * mt  # reasoning fills the budget
        d = cost(ptok, ctok, in_rate, out_rate)
        tok = (ptok + ctok) / 1e6
        grand_calls += calls; grand_tok += tok; grand_cost += d
        print(f"{name:22s} {calls:7.0f} {tok:7.2f} {d:8.2f}  {status}")
    print("-" * 60)
    print(f"{'TOTAL (when all finish)':22s} {grand_calls:7.0f} {grand_tok:7.2f} ${grand_cost:8.2f}")
    print("\nNOTE: real Fireworks rates (web, Jun 2026); glm-5p2 uses GLM-5.1 as proxy. Token/turn "
          "counts are estimated (in-flight runs predate the ledger); caching discount ignored.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", help="path to cost_ledger.jsonl (exact)")
    ap.add_argument("--estimate", action="store_true", help="estimate from run configs")
    ap.add_argument("--in", dest="in_rate", type=float, default=_DEFAULT_BLENDED_IN)
    ap.add_argument("--out", dest="out_rate", type=float, default=_DEFAULT_BLENDED_OUT)
    args = ap.parse_args()
    if args.ledger:
        report_ledger(args.ledger)
    elif args.estimate:
        report_estimate(args.in_rate, args.out_rate)
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
