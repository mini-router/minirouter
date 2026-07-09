"""Offline unit tests for the multi-task summary in scripts/results_table.py.

These tests exercise ONLY the pure ``render()`` aggregation. They make NO live
API calls and need no GPU/network.

The multi-task summary compares two aggregates and prints the paper's R1/R2
verdict (``TRINITY per-task-best avg`` vs ``best fixed single model avg``). For
that comparison to be apples-to-apples, the fixed single baseline must take its
per-benchmark *best* (max across that bench's evals), exactly as TRINITY does
(``max(r["trinity"] ...)``) and as the code comment on line 74 states. Taking the
mean instead understates the single baseline and can flip R1/R2 into a false win.
"""
import importlib.util
import sys
from pathlib import Path

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results_table.py"
_spec = importlib.util.spec_from_file_location("results_table", _SCRIPT)
rt = importlib.util.module_from_spec(_spec)
sys.modules["results_table"] = rt
_spec.loader.exec_module(rt)


def _row(benchmark, coordinator, trinity, random, single):
    """Build a row shaped like results_table.load_rows() output."""
    singles = {"deepseek": single}
    return {
        "file": f"experiments/{coordinator}/{benchmark}/eval.json",
        "benchmark": benchmark,
        "coordinator": coordinator,
        "variant": "eval",
        "trinity": trinity,
        "random": random,
        "best_single": single,
        "best_model": "deepseek",
        "singles": singles,
    }


# Two benchmarks, two coordinator evals each. The single model 'deepseek' has a
# high and a low eval on each bench, so max vs mean diverge sharply.
#   math500: TRINITY {0.85, 0.84}; deepseek {0.70, 0.90}
#   mmlu:    TRINITY {0.85, 0.83}; deepseek {0.80, 0.86}
# TRINITY per-task-best avg = mean(0.85, 0.85)               = 0.850
# best fixed single (max)   = mean(max .90, max .86)         = 0.880  -> R1/R2 FALSE
# best fixed single (mean)  = mean(mean .80, mean .83)       = 0.815  -> R1/R2 would falsely HOLD
_ROWS = [
    _row("math500", "c1", trinity=0.85, random=0.50, single=0.70),
    _row("math500", "c2", trinity=0.84, random=0.50, single=0.90),
    _row("mmlu", "c1", trinity=0.85, random=0.50, single=0.80),
    _row("mmlu", "c2", trinity=0.83, random=0.50, single=0.86),
]


def test_single_baseline_uses_per_bench_max_not_mean():
    """The fixed single baseline must aggregate each bench by max, not mean."""
    md = rt.render(_ROWS)
    # per-bench max: mean(0.90, 0.86) = 0.880
    assert "single: deepseek (fixed) | 0.880" in md
    # the buggy mean value (0.815) must NOT appear
    assert "0.815" not in md


def test_r1_r2_verdict_is_not_a_false_win():
    """With a fair max-vs-max comparison, TRINITY (0.850) does NOT beat the best
    fixed single (0.880), so R1/R2 must report ❌ — not a false HOLDS."""
    md = rt.render(_ROWS)
    r1_line = next(line for line in md.splitlines() if line.startswith("**R1/R2**"))
    assert "❌" in r1_line
    assert "HOLDS" not in r1_line
    assert "0.850 vs 0.880" in r1_line


def test_r1_r2_true_win_still_holds():
    """Sanity check the other direction: when TRINITY genuinely beats the best
    per-bench single, R1/R2 still reports a win."""
    rows = [
        _row("math500", "c1", trinity=0.95, random=0.50, single=0.70),
        _row("math500", "c2", trinity=0.96, random=0.50, single=0.72),
        _row("mmlu", "c1", trinity=0.95, random=0.50, single=0.80),
        _row("mmlu", "c2", trinity=0.94, random=0.50, single=0.82),
    ]
    md = rt.render(rows)
    r1_line = next(line for line in md.splitlines() if line.startswith("**R1/R2**"))
    assert "✅ HOLDS" in r1_line
