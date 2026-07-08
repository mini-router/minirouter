# Improving the TinyRouter classifier — 2025/2026 research notes

How this was produced: a multi-agent web search restricted to **2025–2026 work only**, where every
technique was adversarially re-checked for recency and for whether it actually applies to *our* router
(frozen Qwen3-0.6B penultimate-token feature, ~10K-param linear head over which-LLM × which-role,
sep-CMA-ES against a sparse binary terminal reward, ~31.7k-eval budget, 3-model pool). 17 techniques
survived verification; the ranking below is by expected impact × feasibility for our setup.

Citations are gathered by the automated search and worth a quick manual skim before implementation.

## The four failure modes we are trying to fix

1. **Thin complementarity** — three same-tier models, so little routing headroom (our math result: a wash).
2. **Sparse, noisy binary reward** — sep-CMA-ES sees one 0/1 bit, so some seeds collapse to bad policies.
3. **Small training n.**
4. **Weak feature** — a single penultimate-token 0.6B hidden state may under-represent the query.

## Headline

Three of the four (sparse reward, small n, weak feature) are fixable cheaply by feeding the binary signal
we already collect into a supervised warm-start and a shaped fitness. **Thin complementarity is a
pool/diagnostic problem no head change can fix**, so measure the oracle ceiling first to know whether any
of this is worth running on the current 3-model pool.

## Do this first (the honest gate)

**1. Oracle-ceiling diagnostic + pool-complementarity audit** · impact high · effort low
Before tuning anything, measure how much routing *can* help. Build a per-query correctness matrix over the
3 models, compute `oracle_acc` (solved by ANY model), `oracle_gap = oracle_acc - best_single`, and
`router_gap_closed = (trinity - best_single) / (oracle - best_single)`. If `oracle_gap` is thin, no head or
optimizer change will move the needle and the fix is **pool curation** (swap the most redundant model for
one that is correct on a disjoint slice), not more router tuning. Add to `scripts/results_table.py` and the
eval path; make it multi-turn-aware so the role axis is not undercounted.
Sources: RouterEval (arXiv 2503.10657, 2025); LLMRouterBench (arXiv 2601.07206, Findings@ACL 2026).

## Ranked recommendations

| # | Technique | Impact | Effort | What to change |
|---|---|---|---|---|
| 1 | Oracle-ceiling diagnostic + pool audit | high | low | see above |
| 2 | Supervised warm-start of the head | high | medium | pretrain the head, then evolve |
| 3 | Shaped CMA-ES fitness (dense + binary anchor) | medium | medium | denser reward, variance reweighting |
| 4 | LRA-CMA-ES (learning-rate adaptation) | medium | low | SNR-based step damping in the optimizer |
| 5 | Multi-layer / pooled encoder feature | medium | medium | better query representation |
| 6 | NeuralUCB online bandit head | medium | high | sample-efficient online learner |
| 7 | UCCI calibrated cascade (early-stop) | medium | high | confidence-based turn stopping |
| 8 | Encoder-is-not-the-bottleneck (prioritization) | medium | low | spend budget on reward/head, add anti-collapse term |
| 9 | LRA-CMA-ES-LED (per-coordinate LRA) | low | high | refines #4, diagonal approx only |
| 10 | Surrogate-assisted CMA-ES prescreening | low | high | stretch eval budget if binding |

### Detail on the high-leverage ones

**2. Supervised warm-start of the head** (RouteLLM golden-label augmentation; CSCR InfoNCE pretraining)
Today the head is zero-init, so CMA-ES starts from a uniform policy and must rediscover routing from one
noisy bit. Instead: one offline pass runs all 3 models in each role once per training query and caches a
`(query, model, role, correct)` matrix (inference cost, not per-ES-eval). Then pretrain the agent-selection
rows of the head (~3K params) with InfoNCE over cached positives (a few hundred Adam steps), pack that into
`x0` instead of zeros, and let sep-CMA-ES fine-tune roles + SVF + multi-turn credit. Label preferentially on
queries where the models *disagree* (ties carry no signal in our thin pool). Attacks modes 2, 3, and the
cold start at once.
Sources: RouteLLM (ICLR 2025, arXiv 2406.18665); Cost-Spectrum InfoNCE Contrastive Routing (2025, arXiv 2508.12491).

**3. Shaped CMA-ES fitness with a binary anchor** (HERO; Router-R1)
`fitness.py` returns the mean of a 0/1 vector, so most candidate minibatches return near-identical flat means
and CMA-ES gets little ranking signal. Ship in two flagged stages. Stage B (low risk, no quality model):
variance-aware prompt reweighting `w = 0.5 + 1.5*sigmoid(5*(sigma_task - sigma_mean))` plus small shaped
terms on TRAINING fitness only (`+0.05*format_ok - 0.05*turn_penalty`); keep eval pure binary. Stage A
(optional): a HERO dense quality proxy in [0,1] (self-consistency vote fraction across turns, or Verifier
accept-confidence), min-max normalized *within* the correct and incorrect buckets so a correct answer always
outranks a wrong one. A/B on the held-out seeds; success = lower cross-seed variance, no worse mean.
Sources: HERO (arXiv 2510.07242, 2025); Router-R1 (NeurIPS 2025, arXiv 2506.09033).

**4. LRA-CMA-ES** (Nomura et al.) — **IMPLEMENTED (default-off), offline-validated**
Our `SepCMAES` wraps stock pycma with no learning-rate adaptation, so it updates at full rate even when
generation-to-generation ranking is pure reward noise — exactly the regime that collapses bad seeds. LRA
shrinks the step when SNR is low and restores it when signal returns, at default popsize (no extra evals).
Shipped in `src/trinity/optim/lra.py` + `sep_cmaes.py` (`sep_cmaes.lra.enabled` / `--lra`). NOTE: the
paper's SNR of the *parameter update* did **not** discriminate our regime (the separable mean displacement
decorrelates even on a clean objective — see JOURNAL 2026-07-09); the shipped controller instead keys off
the **fitness-estimation SNR** `Var(candidate fitnesses)` vs the `p(1-p)/m_cma` sampling noise, which is
neutral on a clean signal by construction and damps only when the ranking is noise-dominated. Validated on
the S7 synthetic objective with injected noise (`utility/lra_ablation.py`): neutral on clean, −1.8/−5.8/−9.6%
final distance at noise sd 0.75/1.5/3.0. The held-out-math A/B (blank-init vs `--lra`) is still unrun.
Source: LRA-CMA-ES (ACM TELO 5(1), March 2025, DOI 10.1145/3698203).

**5. Multi-layer / pooled encoder feature**
`encode()` already runs with `output_hidden_states=True`, so capturing mid/upper-layer states or mean-pooling
is nearly free. A/B mean-pool over query tokens vs the current penultimate token, and concatenating upper-half
layers, with offline Fisher/double-fault layer selection then PCA back to 1024-d so the ES vector stays flat.
Tempered: at 0.6B scale the evidence is mixed and encoder swaps barely move accuracy elsewhere, so this is an
A/B, ranked below the reward/warm-start levers.
Sources: Prefill-Activations encoder recipe (arXiv 2603.20895, 2026); Value Aggregation (arXiv 2602.01572, 2026); ICL-Router (AAAI 2026).

## Quick wins (low-effort, reversible)

- **Oracle-ceiling diagnostic** in `results_table.py` (a few hours, cannot regress anything, tells you whether
  further tuning is worthwhile on this pool).
- **Instruction prefix in `encode()`**: prepend `"Instruct: Select the best solver model and role for the
  following query.\nQuery: "` before tokenization (zero new params; needs a short re-run since old head
  checkpoints are invalid under shifted features). Evidence: Qwen3-Embedding (arXiv 2506.05176).
- **HERO variance reweighting** in `fitness.py` (reversible flag, no quality model, concentrates the tiny
  budget on outcome-flipping queries).
- **Turn penalty + format-OK term** on TRAINING fitness only, reusing the existing `has_answer()` helper.

## Demoted (kept, not worth it now)

- **RouterEval / LLMRouterBench** are diagnostic, no importable algorithm (folded into #1).
- **NeuralUCB** (arXiv 2603.30035, 2026) is mapped but a full ES-loop rewrite, and its own limitation is our
  mode 1 — it finds best-single faster but cannot beat it.
- **LRA-CMA-ES-LED** (GECCO 2025) needs full covariance; we run diagonal/separable CMA, so a faithful port is
  infeasible at n=13,312.
- **Surrogate-assisted CMA-ES** (GECCO 2025) only buys sample-efficiency, not a ceiling lift, and its
  near-noiseless assumption is violated by our binary noisy reward.

## Bottom-of-stack truth

On our current 3 same-tier models, the ceiling is the binding constraint (this is the same lesson the sibling
`project_harness` reached independently). The single most informative next step is the **oracle-ceiling
diagnostic**; if the gap is thin, the real lever is the **model pool**, and the warm-start + shaped-fitness +
LRA changes are what to do if and when we add a genuinely complementary model.

## Related effort: the Fugu / Conductor direction (orthogonal to this doc)

Everything above stays inside the **routing** paradigm (make the CMA-ES head better). A separate 2026
research effort, in [`docs/fugu/FUGU_REPLICATION_RESEARCH.md`](./fugu/FUGU_REPLICATION_RESEARCH.md) (sources
indexed in [`docs/fugu/REFERENCE_INDEX.md`](./fugu/REFERENCE_INDEX.md)), looks at replicating Sakana **Fugu**,
which is TRINITY **plus** the Conductor (arXiv:2512.04388): an RL-trained natural-language orchestrator that
emits workflows (subtask + worker + access-list), recurses into itself, and synthesizes. That is a second
model (GRPO), not a head tune, so it does not overlap with the techniques here. Note that Fugu's *base* tier
recipe (SFT on soft per-model performance distributions, then sep-CMA-ES) independently validates #2 and #3
above; the likely missing ingredient was soft performance targets for the SFT stage.
