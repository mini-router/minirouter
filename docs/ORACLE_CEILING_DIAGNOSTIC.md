# Plan: Oracle-Ceiling Diagnostic + Pool-Complementarity Audit

Status: PLAN (not yet implemented). Branch: `oracle-ceiling-diagnostic`.
This is recommendation #1 from [`IMPROVEMENTS.md`](IMPROVEMENTS.md). It is a measurement, not a model
change. Its single job is to answer one question **without lying**:

> On our current 3-model pool, is there enough routable complementarity that improving the router could
> beat the best single model? Or is the ceiling so close to best-single that the only real lever is the
> model pool itself?

Every downstream decision (warm-start the head, shape the reward, swap a model) hinges on this verdict,
so the diagnostic must be **proof against both a false "routing can help" (false positive) and a false
"routing is hopeless" (false negative).** The bulk of this document is how we guarantee that.

## 1. What we measure

For a held-out query set, with `p[q,m]` = probability model `m` solves query `q` (estimated, see §3):

- `best_single = max over m of (mean over q of p[q,m])` — the best fixed single model.
- `routing_oracle = mean over q of (max over m of p[q,m])` — the best a **perfect query-conditional
  deterministic router** can achieve (pick the model most likely to solve each query). THIS is the honest
  ceiling for routing.
- `clairvoyant_any = mean over q of (1 - product over m of (1 - p[q,m]))` — probability that *some*
  model would have solved it. An optimistic UPPER bound that a single-pick router cannot reach.
- `routing_headroom = routing_oracle - best_single` — the number that decides the verdict.
- `unroutable_noise = clairvoyant_any - routing_oracle` — headroom that exists only because of luck /
  nondeterminism and is NOT capturable by routing.
- `router_gap_closed = (trinity - best_single) / (routing_oracle - best_single)` — how much of the real
  headroom our trained router actually captures.

## 2. The two ways a naive version lies (and the fixes)

### 2.1 False positive: the naive "solved by any model" oracle overcounts

The common diagnostic is `oracle = fraction of queries solved by ANY model`. With stochastic models and a
single sample each, this **inflates** the ceiling: a query where each of 3 models has a 40% chance gets
solved by *someone* ~78% of the time, but a router that commits to ONE model per query can only ever get
~40% on it. Reporting the 78% as "routing headroom" is a false positive ("routing can help a lot") for
something a router can never realize.

Two compounding sub-effects:
- **Union-over-noise inflation:** `clairvoyant_any` counts independent lucky draws across models.
- **Winner's-curse on the max:** even `routing_oracle = mean_q max_m p̂[q,m]` is upward-biased when `p̂`
  is a noisy estimate, because `max` of noisy numbers is `>=` max of true numbers (Jensen).

**Fixes:**
1. Report `routing_oracle` (achievable) as the headline ceiling, and show `clairvoyant_any` separately and
   explicitly labelled "not routing-achievable, shows noise not opportunity."
2. **Cross-fit the max** to kill the winner's-curse bias: split each `(q,m)`'s K samples into halves A and
   B. Use half A to pick the argmax model per query, evaluate THAT model's solve rate on half B. The
   selected-model accuracy is then an unbiased estimate of what a router that picked the same way would
   get. Average over multiple A/B splits. This is the single most important false-positive guard.
3. Estimate `p[q,m]` from K independent samples (§3), not one, so the noise that drives both sub-effects
   shrinks.

### 2.2 False negative: a single-turn, single-role oracle undercounts

TRINITY is not just "pick one model." It is `(model, role)` decisions over up to 5 turns
(Thinker -> Worker -> Verifier). A query unsolvable by any model in one Worker turn may be solvable by a
multi-turn sequence. If we compute the oracle from single-model single-turn Worker correctness only, we
get a **lower bound** on the true reachable ceiling, so a thin gap could wrongly read as "routing is
hopeless" when multi-turn collaboration would have helped.

**Fixes:**
1. Compute the oracle at three reachability levels and report all three:
   - **L0 single-turn Worker** (cheapest; a strict lower bound on the ceiling).
   - **L1 best-role-per-model** (each model as Thinker/Worker/Verifier-style single answerer; captures
     role sensitivity).
   - **L2 short multi-turn probe** (a small sample of `(model, role)` sequences within the 5-turn budget),
     reported with its own CI because it is sampled, not exhaustive.
2. State explicitly that L0 is a lower bound: a thin L0 headroom only rules out routing if L1 and L2 are
   also thin. The verdict "routing hopeless" requires the WIDEST reachability level we measured to still
   be thin.

## 3. Denoising the correctness matrix (the input everything depends on)

- For each `(query, model)` (and per role/turn-config at L1/L2), draw **K independent samples** and record
  per-sample binary correctness, giving `solves[q,m] in {0..K}` and `p̂[q,m] = solves/K`.
- Default **K = 5** (matches and extends the rigorous eval's 3 reps). The aggregate ceilings are averages
  over ~120 queries, so per-query noise concentrates; K=5 is enough for the aggregate CIs. MMLU is near
  deterministic (observed single-model std ~0.004 to 0.01), so K=3 suffices there; math is noisier, use
  K=5 or more.
- Sampling temperature must match how the router is actually evaluated (temperature 0 / `reasoning=minimal`
  for the Worker path) so the matrix reflects deployment behaviour, not a different decoding regime.

## 4. Statistics (no verdict from a point estimate)

- **Bootstrap CIs** (resample queries with replacement, >=2000 draws) for `best_single`, `routing_oracle`,
  `routing_headroom`, `unroutable_noise`, and `router_gap_closed`. The verdict is read off the CIs, never
  the point estimates.
- **Paired tests:** all systems are scored on the SAME queries, so use McNemar / paired bootstrap for
  TRINITY-vs-best-single and oracle-vs-best-single, not unpaired comparisons.
- Report per-model accuracies (not just the max) plus the per-query disagreement rate (fraction of queries
  where the 3 models do not all agree) as the raw complementarity signal.

## 5. Integrity guards (so a bug in the diagnostic does not produce a confident wrong answer)

1. **Reuse the FIXED grader** (`reward.score_text` / `reward._committed_answer`). The brittle-extraction
   bug already cost us a phantom MMLU 0.95 -> 0.55 swing; the diagnostic must not reintroduce it.
2. **Grader audit:** manually (or with a second independent check) review a random sample of ~30 grading
   decisions per benchmark, especially boundary cases, and report an estimated grader error rate. Grader
   false-negatives deflate the oracle; false-positives inflate it. The audit bounds this confound.
3. **Cross-check against the rigorous aggregates:** the per-query matrix, averaged per model, must
   reproduce the rigorous eval numbers (glm 0.794, deepseek 0.747, kimi 0.742 on math; etc.) within CI. If
   it does not, the matrix collection is buggy and the verdict is void. This is a cheap, decisive
   collection-correctness check.
4. **Threshold sensitivity:** report the oracle under the probabilistic definition (expected accuracy) AND
   under a hard `solve if p >= 0.5` definition, and show the verdict is stable across both. A verdict that
   flips on the threshold choice is not trustworthy.
5. **Held-out only:** run on the held-out eval split (the same n=120), never training queries, so the
   verdict reflects generalization.

## 6. Decision rule (CI-gated)

Let `H = routing_headroom` measured at the widest reachability level (L2 if run, else L1).

- **Pool-bound (routing cannot help here):** the UPPER CI bound of `H` is small (<= ~0.02) AND `H`'s CI
  includes 0. Action: do NOT tune the head; the lever is the **pool**. Confirm by reporting per-query
  disagreement and `unroutable_noise`; recommend swapping the most redundant model (highest pairwise
  agreement, fewest unique solves) for one correct on a disjoint slice.
- **Router-bound (headroom exists, router leaves it on the table):** lower CI bound of `H` > 0 AND
  `router_gap_closed` < ~0.5. Action: pursue IMPROVEMENTS.md #2 (warm-start), #3 (shaped fitness),
  #4 (LRA), #5 (feature).
- **Near-ceiling (router already good):** `H` real AND `router_gap_closed` high. Action: gains require a
  better pool, not more router tuning.

The rule is deliberately conservative against false positives: we only claim "routing can help" when the
headroom CI excludes 0, and only claim "routing is hopeless" when even the widest reachability level is
thin.

## 7. Implementation

New script `scripts/oracle_ceiling.py` (plus a small extension to the eval path to dump per-query results).

1. **Matrix collection** (`scripts/oracle_ceiling.py --benchmark math500 --k 5 --level L0`):
   - Load the held-out tasks (reuse `dataset.load_tasks(..., split="test", seed=...)` with the SAME seed as
     the rigorous eval so the query set matches).
   - For each task, each model, each of K reps: call the model in the answering role, grade with
     `reward.score_text`, record `correct in {0,1}`.
   - L1 adds a loop over roles; L2 samples a few `(model, role)` sequences via `session.run_trajectory`.
   - Write `experiments/final/oracle_matrix_<bench>.json`:
     `{ "benchmark", "k", "level", "seed", "tasks": [ {"id","answer","per_model": {m: [0/1,...K]}} ] }`.
2. **Analysis** (`scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json`):
   - Compute best_single, routing_oracle (cross-fit), clairvoyant_any, headroom, unroutable_noise.
   - Pull TRINITY's per-query correctness (extend `eval.py` to dump per-query, or reuse if cached) for
     `router_gap_closed`.
   - Bootstrap CIs; McNemar; threshold-sensitivity; rigorous-aggregate cross-check.
   - Emit a Markdown section appended to `RESULTS.md` and a machine-readable
     `experiments/final/oracle_report_<bench>.json`.
3. **`scripts/results_table.py`** gains an oracle column (oracle, headroom, gap_closed with CIs).

### Self-validation (before trusting any real run)
- **Synthetic unit tests:** feed hand-built matrices with a known oracle (e.g., 3 disjoint specialists ->
  routing_oracle = 1.0, headroom = 0.67; 3 identical models -> headroom = 0) and assert the computed values
  and the cross-fit debiasing match. Include a pure-noise matrix (all p=0.5) and assert headroom CI
  includes 0 (no false positive on noise).
- **Determinism check:** re-run matrix collection on a 10-query subset and confirm `p̂` estimates agree
  within sampling error (catches nondeterministic collection bugs).

## 8. Cost and runtime

- Core L0 matrix: 3 models x 120 queries x K=5 = 1,800 calls per benchmark. Math (long reasoning) is the
  expensive side; MMLU is cheap and near-deterministic (K=3 -> 1,080 calls).
- Estimate ~$8 to $12 total for L0 on both benchmarks at current Fireworks rates; L1 (x roles) and L2
  (multi-turn probe) add more and are optional second steps gated on the L0 verdict.
- We can partly reuse the rigorous eval's already-paid compute if we re-run with per-query logging; the
  matrix is what was missing, not the calls.
- Runs detached on the GPU box exactly like the rigorous eval, with the concurrency/retry settings already
  hardened (concurrency 8, retries 10, 60s backoff).

## 9. False-positive / false-negative defense summary

| Failure of the diagnostic | Direction | Defense |
|---|---|---|
| "Solved by any" union over noise | false positive (routing looks good) | report routing_oracle, not clairvoyant_any |
| Winner's-curse bias in max_m p̂ | false positive | cross-fit the argmax (split-half select/evaluate) |
| Single-sample label noise | both | K-sample solve probabilities |
| Single-turn/role oracle | false negative (routing looks hopeless) | L0/L1/L2 reachability; L0 is a lower bound |
| Small-n point estimate | both | bootstrap CIs + paired McNemar; verdict from CIs |
| Grader extraction errors | both | reuse fixed grader + manual audit of ~30 decisions |
| Buggy matrix collection | both | cross-check per-model means vs rigorous aggregates |
| Threshold knife-edge | both | report expected-accuracy AND p>=0.5 oracle; require stable verdict |

## 10. Deliverables

- `scripts/oracle_ceiling.py` (collect + analyze) with synthetic unit tests.
- Per-query matrices and oracle reports under `experiments/final/`.
- An "Oracle ceiling" section appended to `RESULTS.md` with the CI-gated verdict.
- A `JOURNAL.md` entry recording the verdict and which IMPROVEMENTS.md path it unlocks.
- All committed on the `oracle-ceiling-diagnostic` branch.
