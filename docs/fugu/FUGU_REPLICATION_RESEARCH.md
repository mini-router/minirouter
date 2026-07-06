# Replicating Sakana Fugu with Open-Source Models

> Research dossier for the **TinyRouter** project (open-source TRINITY replication).
> Produced 2026-06-25 by a multi-agent web sweep (6 angles, per-angle adversarial
> recency + measured-vs-marketed audit). **Scope: 2026 sources only.** Pre-2026 work
> appears only as labelled lineage. Every source is date-stamped; the full source list
> with dedup status lives in [`REFERENCE_INDEX.md`](./REFERENCE_INDEX.md).

---

## 0. TL;DR

- **Fugu = TRINITY + the Conductor, productized.** Sakana shipped Fugu on 2026-06-22 as a
  single OpenAI-compatible model that internally routes, delegates, verifies, and synthesizes
  across a pool of frontier LLMs (and itself, recursively). It is built on **two** ICLR 2026
  papers: TRINITY (Xu et al.) which TinyRouter already replicates, and **the Conductor**
  (Nielsen et al.) which TinyRouter does **not** have yet.
- **The gap is a second model, not a tune of the existing router.** TinyRouter's CMA-ES head
  (~13K params) corresponds to the cheap **base Fugu** tier. The missing half is the
  **Conductor**: a ~7B model RL-trained (GRPO) to emit a natural-language workflow (which
  workers, what each does, who reads whose output) with recursive self-call. That is the
  **Fugu-Ultra** tier.
- **Every Fugu number is first-party.** No independent third-party reproduction of any
  Fugu / Conductor / TRINITY benchmark existed as of 2026-06-25. Treat all benchmark cells as
  vendor claims, not established results. Independent practitioner tests found Fugu-Ultra slow
  and not clearly better than a single frontier model in real use.
- **Everything needed to build an open clone exists in 2026 open source.** Conductor-class
  base models (Qwen3.5 family), the worker pool TinyRouter already serves
  (deepseek-v4-pro, glm-5p2, kimi-k2p6), GRPO/RLVR trainers (slime, verl, prime-rl + verifiers),
  and serving stacks (vLLM, SGLang). The closest open template for the Conductor is a
  non-Sakana paper, **Uno-Orchestra** (arXiv:2605.05007).
- **Closest existing open replication: OpenFugu** (trotsky1997, Apache-2.0). It already pairs a
  Qwen3-0.6B CMA-ES router with a GRPO-trained Conductor (a Llama-3.2-3B fine-tune on
  `nvidia/ToolScale`, weights on Hugging Face). Mine it; do not trust its headline number.

---

## 1. Top-line evidentiary caveat (read before any benchmark table below)

As of 2026-06-25 there is **no independent, third-party, reproducible benchmark** of Fugu,
the Conductor, or TRINITY. Specifically:

1. Every Fugu and Fugu-Ultra benchmark cell is **Sakana first-party**. Sakana's own note states
   that "all scores other than Fugu's are reported by the model providers", so the baselines are
   provider-reported reference numbers, not a controlled head-to-head in one shared harness.
2. The headline "shoulder-to-shoulder with Fable 5 and Mythos Preview" claim is **not a
   measurement**. Those two Anthropic models are export-controlled, were never in Fugu's pool,
   and were never run head-to-head; the comparison uses provider-reported reference scores.
3. Sakana has released **no Fugu weights and no training code**. `github.com/SakanaAI/fugu` is a
   documentation and CLI repo that points to the commercial API.

Where this dossier quotes a number, read it as "the authors report X", never "X is true". The
Conductor paper's own lower, more-detailed figures (LiveCodeBench 83.93, GPQA-Diamond 87.5) are
the safer replication targets, not the product page's full-stack ~93.

---

## 2. What Fugu is

**Primary source:** Sakana Fugu Technical Report, arXiv:2606.21228 (v1 2026-06-19, v2 2026-06-23),
plus the product pages [sakana.ai/fugu](https://sakana.ai/fugu/),
[sakana.ai/fugu-release](https://sakana.ai/fugu-release/) (2026-06-22), and the beta page
[sakana.ai/fugu-beta](https://sakana.ai/fugu-beta/) (2026-04-24).

Fugu is an **orchestrator language model sold as a single model**. It is itself an LLM trained to
call a swappable pool of frontier worker LLMs, including instances of itself recursively, then
return one synthesized answer behind one OpenAI-compatible endpoint (Chat Completions plus
Responses; CLI `codex-fugu`).

### 2.1 Two tiers, two methods, one API

| Tier | What it does | Training (per the technical report) | Lineage |
| --- | --- | --- | --- |
| **Fugu** | Latency/cost-balanced. A lightweight selection head routes each query to **one** worker. User-configurable pool (you can opt providers out). | SFT on single-step tasks against **soft performance distributions**, then **separable CMA-ES** on end-to-end multi-turn tasks. | TRINITY |
| **Fugu Ultra** | Max quality. Emits a **full natural-language agentic workflow**: per-step subtasks, an assigned worker per step, and an access-list indexing which prior outputs feed each step. Fixed pool. | **GRPO with no KL-divergence penalty.** | The Conductor |

The two tiers are **not fused into one network**. They are two framework lineages served behind
one API. This is the single most important architectural fact for replication: closing the Fugu
gap means standing up a second orchestrator alongside the existing router, then serving both
behind one endpoint.

### 2.2 Deployed worker pool and mechanics

- **Pool (named in the report):** Gemini-3.1-Pro, Claude-Opus-4.8, GPT-5.5. All closed frontier
  models. Per-query selection and routing are proprietary and never disclosed.
- **Workflows:** up to **5 steps**, with persistent shared memory across the workflow while each
  agent's function-calling trajectory stays isolated.
- **Recursive self-call:** Fugu/Conductor can select itself as a worker; recursion depth is tunable
  at inference time, giving a test-time compute axis that activates when the model recognizes a
  failed attempt and spins up a corrective workflow.
- **Orchestrator size:** **not disclosed** in the report. Only the lineage sizes are public
  (Conductor 7B, TRINITY ~0.6B).

### 2.3 Benchmark table (all first-party; baselines provider-reported)

Columns are **Fugu / Fugu-Ultra**. Treat as vendor claims.

| Benchmark | Fugu | Fugu-Ultra | Note |
| --- | --- | --- | --- |
| SWE-Bench Pro | 59.0 | 73.7 | Opus-4.8 reference 69.2 |
| Terminal-Bench 2.1 | 80.2 | 82.1 | |
| LiveCodeBench v6 | 92.9 | 93.2 | report HTML also shows 90.3/92.0; minor cross-source drift |
| LiveCodeBench Pro | 87.8 | 90.8 | |
| Humanity's Last Exam | 47.2 | 50.0 | |
| CharXiv Reasoning | 85.1 | 86.6 | |
| GPQA-Diamond | 95.5 | 95.5 | |
| SciCode | 60.1 | 58.7 | **cheaper Fugu beats Ultra** |
| tau^3 Banking | 21.7 | 20.6 | **cheaper Fugu beats Ultra** |
| Long-Context Reasoning | 74.7 | 73.3 | |
| MRCRv2 | 86.6 | 93.6 | |

Sakana's claim is "top 10 of 11". The two cells where base Fugu beats Fugu-Ultra (SciCode,
tau^3 Banking) undercut the clean tiering story. Beta numbers (2026-04) differed from the final
report, confirming the benchmarks moved between beta and release.

### 2.4 Product facts

- **Pricing:** subscriptions $20 (Standard) / $100 (Pro) / $200 (Max). Fugu-Ultra PAYG
  $5 input / $30 output / $0.50 cached per 1M tokens ($10 / $45 / $1.00 above 272K context).
  "Never stack model fees" single top-tier rate.
- **Availability:** not in EU/EEA. No open weights. Ultra model id `fugu-ultra-20260615`;
  roughly 500 beta users at launch (press-reported).
- **Token fanout:** roughly 4 to 6x tokens per request versus a single-model call, the cost and
  latency tax of orchestration (independent aggregation).

### 2.5 Why it launched when it did

Fugu's "frontier capability without export-control risk" framing responds to a concrete event:
on **2026-06-12**, a US BIS "Is Informed" letter under ECRA forced Anthropic to disable Fable 5
and Mythos 5 worldwide, three days after their 2026-06-09 launch (a Pentagon "supply chain risk"
designation preceded it in March 2026). Fugu's pitch is vendor-lock-in insurance. The framing is
contested (see Section 6).

---

## 3. The two ICLR 2026 papers

### 3.1 TRINITY (already replicated, context only)

**TRINITY: An Evolved LLM Coordinator**, arXiv:2512.04695 (Xu, Sun, Schwendeman, Nielsen, Cetin,
Tang; ICLR 2026). Dates: v1 2025-12-04, v3 2026-04-27.

This is what TinyRouter implements: a frozen ~0.6B coordinator plus a ~10K-param head plus SVF
scales, optimized by separable CMA-ES against a sparse binary reward, assigning one of three roles
(Thinker / Worker / Verifier) to one selected LLM per turn over up to 5 turns. Reported 86.2% on
LiveCodeBench (first-party). The method is routing only; it does not decompose tasks, build
workflows, recurse, or synthesize. **It is the baseline, not new headroom.**

### 3.2 The Conductor (THE replication target)

**Learning to Orchestrate Agents in Natural Language with the Conductor**, arXiv:2512.04388
(Nielsen, Cetin, Schwendeman, Sun, Xu, Tang; Sakana AI plus University of Michigan; ICLR 2026).
Dates: v1 2025-12-04, v5 2026-05-06; Sakana blog
[sakana.ai/learning-to-orchestrate](https://sakana.ai/learning-to-orchestrate/) 2026-04-27.
**The 2026 anchor is the v5 revision / ICLR 2026 / the April blog**; the Dec-2025 v1 predates the
2026-only scope and is disclosed here for honesty.

This is the exact half TinyRouter lacks. Verified mechanics:

- **Base model:** Qwen2.5-7B, RL fine-tuned (the "7B Conductor").
- **Output format:** three parallel Python lists, capped at 5 steps:
  1. `model_id`: integer worker selection per step.
  2. `subtasks`: a natural-language instruction per step (this is the learned prompt engineering).
  3. `access_list`: `[]`, `'all'`, or specific prior-step indices, defining the **communication
     topology** (which earlier outputs each step can see). It emits the workflow in natural
     language; it does not execute code.
- **RL algorithm:** **GRPO.** Group size G = 64 rollouts per question; advantage is
  group-normalized `A_i = (r_i - mean) / std`; **KL penalty beta = 0** (no reference model);
  200 iterations; batch 256 (4 questions per iteration x 64 rollouts); AdamW, lr 1e-6, cosine
  schedule, 0.03 warmup; train temperature 1.0; worker inference temperature 0.2.
- **Reward (two-stage, verifiable):** `r = 0` if the lists fail to parse (format gate); otherwise
  `r = 1` if the final output matches ground truth, `r = 0.5` if it does not. Pure end-to-end
  reward; no process reward.
- **Pool generalization (the swappability trick):** for each question, restrict the Conductor to a
  randomly sampled k-of-n subset of the worker pool and adjust its instructions accordingly, so one
  Conductor adapts to arbitrary open or closed pools. Pool size n = 7.
- **Recursive self-call (test-time scaling):** the Conductor can select itself as a worker; each
  inner call receives its own parent output plus the previous agent's response, letting it detect
  team failure and build a corrective workflow. Fine-tuned 20 extra iterations on a 350-sample
  filtered subset, 64 rollouts/sample, reward scaled 0.25 for non-recursive rounds; max depth set
  at inference.
- **Worker pool (7 agents):** closed GPT-5, Gemini-2.5-Pro, Claude-Sonnet-4; open
  DeepSeek-R1-Distill-Qwen-32B, Gemma3-27B-instruct, Qwen3-32B (workers capped at 4096 completion
  tokens, temp 0.2).
- **Training data:** roughly 960 problems across MATH500, an MMLU subset, LiveCodeBench V1, and
  RLPR. Held-out eval on AIME25 (30), GPQA-Diamond (198), BigCodeBench.
- **Results (first-party):** LiveCodeBench 83.93 (vs GPT-5 82.90), GPQA-Diamond 87.5
  (vs Gemini-2.5 84.8), AIME25 93.3, MMLU 94.1; beats every single worker. Against multi-agent
  baselines (constrained, average): Conductor 72.35 > MoA 62.13 > MASRouter 56.89 > RouterDC 52.41.

Audit note: the abstract describes "reinforcement learning / end-to-end reward maximization"; the
"GRPO, 200 iters, batch 256, no KL, G=64" recipe is from the v5 full text. Re-read
`arxiv.org/html/2512.04388v5` and the OpenReview camera-ready before locking the exact recipe.

---

## 4. What Fugu adds beyond TinyRouter (the four deltas)

| # | Delta | TinyRouter today | Fugu / Conductor | Build difficulty |
| --- | --- | --- | --- | --- |
| 1 | **Natural-language orchestration** | fixed Thinker/Worker/Verifier turn loop; head picks (model, role) | a 7B policy writes a task-specific DAG: subtask + worker-id + access-list per step | **Large.** RL on a ~7B model, not evolving 13K params. The center of the gap. |
| 2 | **Recursive self-invocation** | flat multi-turn loop, capped at 5, cannot re-enter itself | orchestrator selects itself as a worker, fed its own parent output; bounded depth = test-time compute knob | Small once Delta 1 exists. The lever behind Fugu's test-time scaling. |
| 3 | **Answer synthesis** | Verifier accept-and-stop returns one worker's output | implicit: the terminal DAG node reads prior outputs via the access-list | **Open research.** No published recipe across any of the three Sakana papers. |
| 4 | **Productization** | Fireworks pool, eval harness | swappable pool behind one OpenAI-compatible API, two tiers | Engineering, not research. |

Delta 3 deserves emphasis. Synthesis is the **least-specified component in all three Sakana
papers**. The Conductor's only documented "synthesis" is the access-list terminal node reading
prior outputs; Fugu's product page claims internal "verification and synthesis" but the report
gives no recipe. Plan to design this, not port it.

---

## 5. The 2026 source and paper catalogue

Full dedup manifest with links and recency status: [`REFERENCE_INDEX.md`](./REFERENCE_INDEX.md).
Grouped summary here.

### 5.1 Sakana primary (lineage and target)

| Work | ID | 2026 anchor | Role |
| --- | --- | --- | --- |
| Sakana Fugu Technical Report | arXiv:2606.21228 | 2026-06 | the product target |
| The Conductor | arXiv:2512.04388 | v5 2026-05; ICLR 2026 | **the method to replicate** |
| TRINITY | arXiv:2512.04695 | v3 2026-04; ICLR 2026 | already replicated baseline |

### 5.2 Non-Sakana 2026 academic literature (the open templates)

| Work | ID | Date | Transferable idea |
| --- | --- | --- | --- |
| **Uno-Orchestra** | arXiv:2605.05007 | 2026-05 | **Closest open Conductor analog.** One causal LM decompose-then-route in a forward pass; Agentic-GRPO (turn-level credit); blind-worker protocol; verifier-gated curriculum; bounded shaping rewards. CC BY 4.0. |
| Graph-GRPO | arXiv:2603.02701 | 2026-03 | edge-level group-relative advantages for low-variance topology-learning credit assignment |
| R2-Router | arXiv:2602.02823 | 2026-02 (ICML 2026) | make compute/output-length budget a jointly-routed variable; SOTA at 4-5x lower cost (first-party) |
| ACRouter (Agent-as-a-Router) | arXiv:2606.22902 | 2026-06 | Orchestrator + Verifier + Memory in a Context-Action-Feedback loop; ships CodeRouterBench |
| AgentRouter | aclanthology 2026.acl-long.33 | ACL 2026 | non-RL GNN with soft supervision and **weighted-aggregation synthesis**; preprint 2510.05445 is 2025 lineage |
| EvolveRouter | arXiv:2604.05149 | 2026-04 | co-evolve routing **and** prompts; maps onto TinyRouter's existing CMA-ES loop |
| DeepVerifier | arXiv:2601.15808 | 2026-01 | verification-as-decomposition; rubric reward denser than binary right/wrong |
| RL over orchestration traces (survey) | arXiv:2605.02801 | 2026-05 | taxonomy of reward families and sub-decisions; flags the **STOP decision** as nearly unaddressed in RL |
| AdaptOrch | arXiv:2602.16873 | 2026-02 | thesis: as models converge, orchestration is the differentiator (motivation) |
| LLMRouterBench | arXiv:2601.07206 | 2026-01 (Findings ACL 2026) | ready-made eval harness: 400K instances, 21 datasets, 33 models, 10 baselines |
| FusionRoute | arXiv:2601.05106 | 2026-01 | token-level fusion; needs decoding control (open-backbone variant only) |
| Can Heterogeneous LMs Be Fused? | arXiv:2604.01674 | 2026-04 | cross-architecture weight fusion; open-backbone variant only |
| Hyperagents | arXiv:2603.19461 | 2026-03 | recursive self-improving agents (adjacent) |
| SEVerA | arXiv:2603.25111 | 2026-03 | verified synthesis of self-evolving agents (formal-methods flavored) |

Pre-2026 lineage, out of scope, listed so nobody re-pulls them: xRouter (arXiv:2510.08439,
Oct 2025), Recursive Self-Aggregation (arXiv:2509.26626, Sep 2025), original SVF / Transformer-squared
(2025).

### 5.3 Open-source ingredients (2026)

See Section 7 tables.

### 5.4 Open replications already in the wild

| Repo | What it is | Use to us |
| --- | --- | --- |
| [trotsky1997/OpenFugu](https://github.com/trotsky1997/OpenFugu) | Apache-2.0. Qwen3-0.6B CMA-ES router + GRPO Conductor (a Llama-3.2-3B fine-tune on `nvidia/ToolScale`, weights on HF `di-zhang-fdu/openfugu-conductor-3b`); serves an OpenAI-compatible endpoint. | **Closest sibling to TinyRouter.** Mine `train/` and `verify/`. Do not trust its "+107%" banner (its own `results/` walks it back to query-level routing on a complementary pool; recursion eval tied). The "7B Conductor" prose is aspirational; shipped weights are 3B. |
| [nshkrdotcom/trinity_coordinator](https://github.com/nshkrdotcom/trinity_coordinator) | Elixir/Nx TRINITY router; consumes fixed artifacts; CMA-ES training lane removed; explicitly does not reproduce paper scores. ~2026-05-21. | Reference only; closest non-Python TRINITY port. |
| [BicaMindLabs/open-sakanafugu](https://github.com/BicaMindLabs/open-sakanafugu) | v1.0.0 2026-06-21. A hand-designed scaffold (9 implementer LLMs + Codex reviewer + bounded review-fix loop). No trained orchestrator, no recursion, no learned synthesis. | What **not** to copy: it is a coordination harness, not a learned orchestrator. |

Caution: a repo under the org `Sakana-AI-labs` (hyphenated) is **not** the official `SakanaAI`
org; treat it as a possible mirror or impersonation.

---

## 6. Independent assessment (measured vs marketed)

The benchmark narrative and the practitioner experience diverge sharply.

- **Ethan Mollick** (within 24h of launch): Fugu-Ultra "incredibly slow", roughly 30 minutes per
  coding test, and does not match Fable in real use.
- **Peter Steinberger:** burned a full 5-hour quota in one prompt; the Three.js result was
  "notably worse than GPT-5.5" and needed 7 to 8 fix rounds.
- **Mark Santos:** measured Fugu-Ultra at roughly $7.32/task (fast) versus Opus-4.8 at $37.85
  (slower, higher quality), suggesting the win is speed and cost, not quality.
- **Paddo.dev** (sharpest skeptic): "a benchmark you can't reproduce, against baselines you can't
  test"; an independent breakdown puts Fable 5 near 86.0 on SWE-Bench Pro versus Ultra's 73.7.
- **Elie Bakouch:** a closed orchestrator over closed models "is not AI sovereignty", and adding a
  new LLM likely needs classifier retraining. (This swappability-needs-retraining limitation is
  exactly TinyRouter's frozen-head constraint; the Conductor's randomized-pool training is the
  documented answer to it.)
- **Stella Biderman** publicly challenged Fugu-Ultra's claims and the export-control framing,
  citing Sakana's history of overstated results.
- **Sakana eval-integrity precedent** that fuels the skepticism: the AI CUDA Engineer (Feb 2025)
  claimed large kernel speedups but exploited a sandbox memory loophole (one case ran 3x slower,
  which Sakana acknowledged); the AI Scientist had a high experimental failure rate.

These independent findings actually **agree with TinyRouter's own honest result**: orchestration's
win is cross-task routing and verification, not a within-task quality miracle, and it carries a
real 4 to 6x token and latency tax. TinyRouter's measured "the win is across tasks, math is a wash"
matches the practitioner reality better than Sakana's marketing does. The methodological lesson:
report reproducible head-to-heads on a published open pool and report the oracle ceiling, which is
the stance the existing oracle-ceiling diagnostic already takes.

Caveat on the counter-evidence too: the "Fable 5 ~86.0" counter-figure is one blog's
non-reproducible breakdown and inherits the same un-runnable-baseline weakness as Sakana's claim;
the cost anecdotes are single-tester numbers, not systematic benchmarks.

---

## 7. How to replicate Fugu with open-source models

The goal: stand up the **Fugu-Ultra (Conductor) tier** that TinyRouter lacks, on the open pool
TinyRouter already serves, and present both tiers behind one endpoint. The architecture mirrors
Sakana's own split: keep the cheap CMA-ES router as the fast path, add a GRPO orchestrator as the
heavy path.

### 7.1 Architecture target

```
                         one OpenAI-compatible endpoint (vLLM/SGLang + thin gateway)
                                          |
                     cheap path  <----  difficulty gate  ---->  heavy path
                          |                                         |
                  TinyRouter CMA-ES router                 open Conductor (GRPO)
                  (= base Fugu, exists today)              emits NL workflow:
                  pick 1 worker + role                     [model_id, subtasks, access_list]
                          |                                  + recursive self-call
                          +----------------> worker pool (Fireworks) <----------+
                              deepseek-v4-pro | glm-5p2 | kimi-k2p6
                                          |
                                  synthesis / verify step
                                          |
                                    final answer
```

### 7.2 Capability → open ingredient → TinyRouter change

| Fugu capability | 2026 open ingredient | Concrete change in TinyRouter |
| --- | --- | --- |
| Conductor base model | **Qwen3.5-4B or 9B** (Apache-2.0, 2026-03-02, 262K ctx). Existing **Qwen3-0.6B** as a throwaway loop-validation seed. | New module `src/trinity/conductor/` holding the policy model + the 3-list decode/parse. |
| NL workflow output (model_id / subtasks / access_list, max 5 steps) | Conductor paper format; **Uno-Orchestra** (arXiv:2605.05007) as the open template | New workflow schema + parser; generalizes the current single (model, role) pick into a DAG. Reuse `roles/` prompts as subtask seeds. |
| GRPO training (no KL, two-stage reward) | **slime** (THUDM, trained GLM-5) or **verl** (most complete multi-turn agentic + tool calling); **prime-rl + verifiers** to wrap the reward | New trainer alongside `optim/`. Reuse TinyRouter's cached per-(query, model, role, correct) oracle as the RLVR reward; upgrade to Fugu's parse-gate + 1.0/0.5/0 shape (this subsumes IMPROVEMENTS.md #2 warm-start and #3 shaped fitness). |
| Worker pool (swappable) | already served: **deepseek-v4-pro** (MIT), **glm-5p2** (MIT), **kimi-k2p6** (Modified MIT) | No new integrations. Add the Conductor's **randomized k-of-n pool** sampling per question so one orchestrator stays pool-agnostic. |
| Recursive self-call | Conductor mechanism; Graph-GRPO (arXiv:2603.02701) for topology credit | Let the Conductor emit its own id as a worker; feed parent output + prior response back in; bound depth as an inference flag. |
| Synthesis / verify | **DeepVerifier** (arXiv:2601.15808) rubric reward; **AgentRouter** (ACL 2026) weighted aggregation as a non-RL alternative | Design a synthesis node (open sub-problem). Start with the access-list terminal-node read; optionally a verifier-scored aggregate. |
| Serve as one model | **vLLM / SGLang** (OpenAI-compatible); thin gateway (LiteLLM-style; `llmhop` is a candidate but its 2026 recency is unverified) | New `serve/` exposing `/v1/chat/completions` with a difficulty gate choosing router vs Conductor. |
| Eval, field-comparable | **LLMRouterBench** (arXiv:2601.07206) | Add as an eval target next to the existing oracle-ceiling diagnostic. |

### 7.3 Reward design (the part TinyRouter already half-has)

Fugu's verifiable reward is essentially TinyRouter's binary oracle with a format gate and a partial
credit tier:

```
r = 0.0                      if the workflow lists fail to parse        (format gate)
r = 1.0                      if final answer == ground truth           (correctness)
r = 0.5                      otherwise                                  (partial)
```

Because the Conductor is the policy producing text, **gradients flow** (unlike the frozen-encoder
CMA-ES setup), so GRPO is the natural optimizer. Two upgrades from the non-Sakana literature attack
TinyRouter's documented failure modes directly:

- **Agentic-GRPO turn-level credit** (Uno-Orchestra) fixes the sparse trajectory-level signal that
  made the shaped-fitness retrain "sampling-noise-dominated".
- **Blind-worker protocol** (Uno-Orchestra): hide and reshuffle worker identities per episode so the
  policy profiles capability through interaction rather than memorizing "glm → math". This is the
  mechanism most likely to capture the ~4.9 points of math headroom the oracle-ceiling diagnostic
  flagged as ROUTER_BOUND.

### 7.4 Staged build plan

1. **Phase 0 (validate the loop, cheap).** Wrap the existing binary oracle as a prime-rl/verifiers
   RLVR environment. Train a Qwen3-0.6B "mini Conductor" on the 3-model pool to emit the 3-list
   format for math + MMLU. Success = it produces parseable workflows and beats random routing on the
   held-out 120. This proves the trainer + reward + pool plumbing on one H200 before spending on a
   bigger base.
2. **Phase 1 (real Conductor).** Swap in Qwen3.5-4B (or 9B if the H200 budget allows), add the
   randomized k-of-n pool sampling and the parse-gate + 1.0/0.5/0 reward. Target the Conductor
   paper's posture (beat every single worker on the in-distribution suite), not the product page's
   ~93.
3. **Phase 2 (recursive self-call + synthesis).** Add self-as-worker with bounded depth; prototype a
   synthesis node (access-list read first, then a DeepVerifier-style rubric-scored aggregate).
   Measure cost/latency multiplier against accuracy, since recursion is where the 4 to 6x fanout tax
   lives.
4. **Phase 3 (serve two tiers).** Difficulty gate: cheap queries to the CMA-ES router (base Fugu),
   hard/multi-step to the Conductor (Fugu-Ultra), behind one OpenAI-compatible endpoint.
5. **Throughout:** log to `docs/JOURNAL.md`; report reproducible head-to-heads and the oracle
   ceiling, never marketing-style single-pool wins.

### 7.5 Reusable vs new

- **Reuse:** the Fireworks pool and `llm/` client; the binary correctness oracle and label cache;
  `roles/` prompt templates (as subtask seeds); the oracle-ceiling diagnostic; the eval harness; the
  H200 + HTTP-pool topology.
- **New:** the Conductor policy model and 3-list decode/parse; the GRPO/RLVR trainer; the
  randomized-pool sampler; recursive self-call; the synthesis node; the two-tier serving gateway.

### 7.6 Feasibility note

The binding cost is **not** GPU; it is **rollout latency**. Each GRPO rollout fans out to several
paid Fireworks calls, and 64 rollouts/question x 200 iters is a lot of HTTP. slime/verl async
rollout helps, but budget the API spend and consider a smaller base, fewer rollouts, or a cheaper
worker subset during training. This is the main open risk for replicating the Conductor on a single
box.

---

## 8. What is genuinely new here (beyond `docs/IMPROVEMENTS.md`)

`IMPROVEMENTS.md` is about making the **existing CMA-ES router** better (warm-start, shaped fitness,
LRA-CMA-ES, better encoder feature, bandit heads). All of that stays inside the routing paradigm.
This dossier is about a **different model**: an RL-trained natural-language orchestrator. The
non-overlapping, genuinely new work is:

1. **A generative NL orchestrator (the Conductor)**, trained by GRPO, replacing the discrete head
   for the heavy path. Not in IMPROVEMENTS.md at all.
2. **Recursive self-call** as a test-time compute axis. New.
3. **Answer synthesis** as a learned step. New, and an open sub-problem with no published recipe.
4. **Randomized-pool training** for swappability without per-model retraining (answers the Bakouch
   critique). New.
5. **GRPO/RLVR plumbing** (slime/verl/prime-rl + verifiers) and a two-tier serving gateway. New
   infrastructure.

Overlap worth noting: Fugu's **base** tier recipe (SFT on soft performance distributions, then
sep-CMA-ES) **validates** IMPROVEMENTS.md #2 and #3. The likely missing ingredient there was **soft
per-model performance targets** for the SFT stage, not just shaped CMA-ES. So IMPROVEMENTS.md is not
wrong; it was the right idea under-powered, and Fugu confirms the two-stage direction.

---

## 9. Risks, unknowns, and the first experiment

### 9.1 Open questions (verify before committing)

- **Synthesis mechanism:** no Sakana source gives a recipe. Design it; treat as research.
- **Orchestrator size:** the Fugu report omits it. Conductor is 7B; OpenFugu shipped 3B. Sweep size
  vs coordination quality on the open 3-model pool, which is smaller and lower-ceiling than Sakana's
  7-agent frontier pool.
- **Recipe pinning:** re-read `arxiv.org/html/2512.04388v5` (and the OpenReview camera-ready) to
  confirm GRPO hyperparameters before locking the trainer config.
- **Rollout economics:** quantify GRPO rollout cost over paid Fireworks APIs on one H200.
- **Small-Conductor viability:** can a 0.6B to 4B open Conductor learn useful topologies on a
  3-model pool, or is 7B a floor?
- **"Soft performance distributions":** what they concretely are (per-model accuracy soft labels?
  distillation?) is load-bearing for the base-tier warm-start.

### 9.2 Suggested first experiment

Phase 0 above: wrap the existing oracle as an RLVR environment and train a Qwen3-0.6B mini-Conductor
to emit parseable 3-list workflows over the current pool on math + MMLU. It is cheap, reuses
everything TinyRouter already has, and answers the one question that gates the whole effort: can an
open small model learn to write a useful workflow over this pool at all? If yes, scale the base to
Qwen3.5-4B. If no, the Conductor path needs a stronger pool or a bigger base before it is worth the
API spend.

---

## 10. Provenance

Produced 2026-06-25 by a 6-angle multi-agent web sweep with per-angle adversarial recency and
measured-vs-marketed auditing, then synthesized and grounded against the TinyRouter codebase. Every
benchmark number above is first-party (Sakana or paper authors) unless explicitly attributed to an
independent tester; no independent third-party reproduction of any Fugu / Conductor / TRINITY result
existed as of this date. The complete, deduplicated source list and recency notes are in
[`REFERENCE_INDEX.md`](./REFERENCE_INDEX.md); consult it before starting any new research on this
topic so we do not re-spend tokens on sources already covered.
