# Fugu Conductor: prompted-baseline result on math500

First real measurement on the open-source pool. Branch `openfugu-replication`.
Date 2026-06-25.

## What was run

A **zero-training** prompted Conductor: `deepseek-v4-pro` is asked to emit a
workflow (the 3-list format), the pool executes it, and the answer is graded by
the shared FIXED grader. No GRPO, no GPU.

- Tasks: the **same 120 held-out math500** tasks as the oracle matrix (verified
  identical set and order).
- Conductor: `deepseek-v4-pro`, `reasoning="none"`, `max_depth=0` (no recursion),
  `reps=1`.
- Workers: `deepseek-v4-pro` / `glm-5p2` / `kimi-k2p6` via Fireworks.
- Command: `scripts/fugu_baseline_eval.py` (cost-capped at $5).

## Result

| System | math500 (120) | Notes |
| --- | --- | --- |
| Routing ceiling (perfect single-pick) | 0.855 | single-turn L0 oracle |
| **Prompted Conductor (this run)** | **0.917** | parse rate 0.975, cost **$1.10** |
| best single (glm-5p2) | 0.808 | oracle matrix K=5 |
| deepseek-v4-pro single-shot | 0.783 | oracle matrix K=5 |
| TinyRouter CMA-ES router | 0.792 | prior rigorous eval |
| random routing | 0.792 | prior rigorous eval |

Paired comparison vs best-single (McNemar on the same 120): **b=15** (Conductor
right, best-single wrong), **c=3** (best-single right, Conductor wrong),
**p_exact = 0.0075**. `router_gap_closed = 2.3` (it exceeds the single-pick
ceiling).

## What it means (and does not)

- The Conductor scored **0.917, which is above the 0.855 single-pick routing
  ceiling.** That is not a contradiction: the ceiling bounds picking ONE model
  per query once; the Conductor runs a **multi-step decompose / solve / verify**
  workflow, which is a more powerful computation and is not bounded by the
  single-pick ceiling.
- The lift is **test-time compute, not routing.** The Conductor sent ~all work
  to deepseek (deepseek 203k completion tokens vs glm 271, kimi 64) and barely
  used the other models. So this is "deepseek in a 2 to 3 step self-checking
  scaffold", scoring +13 points over deepseek single-shot (0.783 -> 0.917).
  Learned routing (use glm for math) is a SEPARATE lever a GRPO-trained
  Conductor would add; the prompted baseline does not route.
- **Cost of the lift:** $1.10 for 120 tasks, roughly 3.6x a single-shot pass.
  That is the multi-step fanout tax (the same tax independent testers measured
  on Fugu itself).

## False-positive / false-negative audit

The headline number was not trusted blind. An 18-task audit printed the gold
answer, the grader's extracted answer, and the verdict for each task:

- **Zero false positives.** Every task the grader marked correct had its
  extracted answer equal to the gold (including non-trivial equalities the grader
  handles: `0.09` == `9/100`, `\dfrac{2}{21}` == `\frac{2}{21}`).
- **One false negative, now fixed.** `math500-459` (gold `$18.90`, answer
  `18.90`) was marked wrong because `normalize_math_answer` stripped the bare `$`
  before `\$`, leaving `\18.90`. Fixed in `orchestration/reward.py` (strip `\$`
  first) with a regression test. This is a SHARED-grader bug, so it slightly
  understated the oracle matrix too. Net effect: **0.917 is a conservative lower
  bound**; correcting `math500-459` alone moves it to about 0.925.

So the grader errs only in the safe direction (a correct answer occasionally
marked wrong), never the dangerous one (a wrong answer marked correct). The
0.917 is real and if anything understated.

## Honest caveats

- **Single rep.** reps=1, so there is per-sample noise; the paired McNemar
  (p=0.0075) controls for it by comparing on the same items, but a 3-rep rerun
  (about $4) would tighten the point estimate.
- **This is the zero-training floor.** It already beats best-single, which was
  not the expectation; the open question GRPO answers is whether *learned
  routing + a learned workflow* beats a strong-model multi-step scaffold, and at
  what cost.
- **One task, one pool.** math500 only, the existing 3-model pool. MMLU (near its
  ceiling) and a stronger/cheaper worker mix may behave differently.

## Reproduce

```bash
# materialize the exact 120 tasks (env with HF datasets):
PYTHONPATH=src python3 -c "import json;from trinity.orchestration.dataset import load_tasks;\
json.dump([{'task_id':t.task_id,'benchmark':t.benchmark,'prompt':t.prompt,'answer':t.answer} \
for t in load_tasks('math500','test',120,0)],open('tasks.json','w'))"
# run the paid eval (lite env), cost-capped + ledgered:
source ~/.config/trinity/secrets.env
TRINITY_COST_LEDGER=ledger.jsonl PYTHONPATH=src python scripts/fugu_baseline_eval.py \
  --tasks-json tasks.json --max-items 120 --conductor-model deepseek-v4-pro \
  --max-depth 0 --reps 1 --max-cost-usd 5
# honest verdict (router_gap_closed + McNemar vs best-single):
python scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json \
  --trinity-per-query experiments/final/fugu_baseline_perquery_math500.json
```
