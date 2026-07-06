# Fugu Replication, Research Reference Index (DO NOT RE-RESEARCH)

This file is the **canonical record of every source already consulted** for the "replicate Sakana
Fugu with open-source models" effort (research run 2026-06-25). Before doing any new research on
Fugu, the Conductor, TRINITY, LLM orchestration, or the open-source ingredients below, **consult
this index first and do not re-fetch listed sources.** It exists so we never spend tokens twice on
the same paper.

Scope rule used: **2026 only.** Sources marked `N` in the in-scope column are pre-2026 lineage,
recorded so they are not re-pulled. Dates are `YYYY-MM`. Findings narrative lives in
[`FUGU_REPLICATION_RESEARCH.md`](./FUGU_REPLICATION_RESEARCH.md); section refs (§) point there.

Every benchmark number in any of these sources is **first-party** (Sakana or the paper's own
authors) unless the source is an independent tester. No independent third-party reproduction of any
Fugu / Conductor / TRINITY benchmark existed as of 2026-06-25.

---

## A. Official Sakana (product + blog + repo)

| # | Source | Type | Date | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [sakana.ai/fugu](https://sakana.ai/fugu/) | official | 2026-06 | Y | §2 | Product/pricing page; full 11-benchmark table; two tiers |
| 2 | [sakana.ai/fugu-release](https://sakana.ai/fugu-release/) | official | 2026-06 | Y | §2 | "One Model to Command Them All" (2026-06-22); recursive self-call; provider-reported baselines note |
| 3 | [sakana.ai/fugu-beta](https://sakana.ai/fugu-beta/) | official | 2026-04 | Y | §2 | Beta (2026-04-24); "Fugu Mini"/"Fugu Ultra"; beta numbers differ from final |
| 4 | [sakana.ai/learning-to-orchestrate](https://sakana.ai/learning-to-orchestrate/) | official | 2026-04 | Y | §3.2 | Conductor blog (2026-04-27); 7B, RL, LCB 83.9, GPQA-D 87.5, recursive scaling |
| 5 | [github.com/SakanaAI/fugu](https://github.com/SakanaAI/fugu) | github | 2026-06 | Y | §1 | Official repo is docs/CLI only (100% shell), NO weights, points to commercial API |

## B. Sakana papers (lineage and target)

| # | Paper | Authors | Date / venue | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | [Sakana Fugu Technical Report](https://arxiv.org/abs/2606.21228) (arXiv:2606.21228) | Tang et al. (Sakana, 14 authors) | 2026-06 (v1 06-19, v2 06-23) | Y | §2 | The product target; training recipe; benchmark tables |
| 7 | [The Conductor](https://arxiv.org/abs/2512.04388) (arXiv:2512.04388) | Nielsen, Cetin, Schwendeman, Sun, Xu, Tang (Sakana + UMich) | ICLR 2026; v1 2025-12, **v5 2026-05** | Y (2026 anchor) | §3.2 | **THE replication target.** 7B Qwen2.5, GRPO no-KL, 3-list NL workflow, recursive self-call |
| 8 | [TRINITY: An Evolved LLM Coordinator](https://arxiv.org/abs/2512.04695) (arXiv:2512.04695) | Xu, Sun, Schwendeman, Nielsen, Cetin, Tang | ICLR 2026; v1 2025-12, **v3 2026-04** | Y (2026 anchor) | §3.1 | Already-replicated baseline; ~0.6B + ~10K head + SVF + CMA-ES; 86.2% LCB |

## C. Non-Sakana 2026 academic literature (the open templates)

| # | Paper | Date / venue | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- |
| 9 | [Uno-Orchestra](https://arxiv.org/abs/2605.05007) (arXiv:2605.05007) | 2026-05 | Y | §5.2, §7 | **Closest open Conductor analog.** decompose-then-route in one LM; Agentic-GRPO; blind-worker; verifier-gated curriculum. CC BY 4.0 |
| 10 | [Graph-GRPO](https://arxiv.org/abs/2603.02701) (arXiv:2603.02701) | 2026-03 | Y | §5.2, §7 | Edge-level group-relative advantages; low-variance topology credit |
| 11 | [R2-Router](https://arxiv.org/abs/2602.02823) (arXiv:2602.02823) | 2026-02 (ICML 2026) | Y | §5.2 | Compute/length budget as a jointly-routed variable; SOTA at 4-5x lower cost |
| 12 | [Agent-as-a-Router / ACRouter](https://arxiv.org/abs/2606.22902) (arXiv:2606.22902) | 2026-06 | Y | §5.2 | Orchestrator+Verifier+Memory C-A-F loop; CodeRouterBench (~10K tasks) |
| 13 | [AgentRouter](https://aclanthology.org/2026.acl-long.33/) (ACL 2026; arXiv:2510.05445) | ACL 2026 (preprint 2025-10) | Y via venue | §5.2 | GNN, soft supervision, **weighted-aggregation synthesis** (non-RL) |
| 14 | [EvolveRouter](https://arxiv.org/abs/2604.05149) (arXiv:2604.05149) | 2026-04 | Y | §5.2 | Co-evolves routing AND prompts; maps onto existing CMA-ES loop |
| 15 | [DeepVerifier](https://arxiv.org/abs/2601.15808) (arXiv:2601.15808) | 2026-01 (v2 2026-04) | Y | §5.2, §7 | Verification-as-decomposition; rubric reward denser than binary |
| 16 | [RL over orchestration traces (survey)](https://arxiv.org/abs/2605.02801) (arXiv:2605.02801) | 2026-05 | Y | §5.2, §7 | Reward/credit taxonomy; flags the STOP decision as nearly unaddressed |
| 17 | [AdaptOrch](https://arxiv.org/abs/2602.16873) (arXiv:2602.16873) | 2026-02 | Y | §5.2 | Orchestration > single-model as models converge (motivation) |
| 18 | [LLMRouterBench](https://arxiv.org/abs/2601.07206) (arXiv:2601.07206) | 2026-01 (Findings ACL 2026) | Y | §5.2, §7 | Eval harness: 400K instances, 21 datasets, 33 models, 10 baselines |
| 19 | [FusionRoute](https://arxiv.org/abs/2601.05106) (arXiv:2601.05106) | 2026-01 | Y | §5.2 | Token-level fusion; needs decoding control (open-backbone variant only) |
| 20 | [Can Heterogeneous LMs Be Fused?](https://arxiv.org/abs/2604.01674) (arXiv:2604.01674) | 2026-04 (v2 2026-05) | Y | §5.2 | Cross-architecture weight fusion (open-backbone variant only) |
| 21 | [Hyperagents](https://arxiv.org/abs/2603.19461) (arXiv:2603.19461) | 2026-03 | Y | §5.2 | Recursive self-improving agents (adjacent; thin mechanism) |
| 22 | [SEVerA](https://arxiv.org/abs/2603.25111) (arXiv:2603.25111) | 2026-03 | Y | §5.2 | Verified synthesis of self-evolving agents (formal-methods flavored) |

## D. Open-source ingredients, worker-pool + Conductor-base models

| # | Source | Org | Date | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | [DeepSeek V4-Pro/Flash (Artificial Analysis)](https://artificialanalysis.ai/articles/deepseek-is-back-among-the-leading-open-weights-models-with-v4-pro-and-v4-flash) | DeepSeek / AA | 2026-04 | Y | §7 | `deepseek-v4-pro` (already in pool); 1.6T/49B active, MIT, AA index 52 |
| 24 | [zai-org/GLM-5](https://github.com/zai-org/GLM-5) | Zhipu/Z.ai | 2026-04 (5.2 2026-06) | Y | §7 | `glm-5p2` (already in pool); MIT; trained with slime; AA index ~51 |
| 25 | [Kimi K2.6 (RITS coverage)](https://rits.shanghai.nyu.edu/ai/moonshot-ai-releases-kimi-k2-6-with-256k-context-and-300-agent-swarms/) | Moonshot AI | 2026-04 | Y | §7 | `kimi-k2p6` (already in pool); 1T/32B active, Modified MIT; AA index ~54 |
| 26 | [Qwen3.5 small models (AA)](https://artificialanalysis.ai/articles/qwen3-5-small-models) | Qwen / AA | 2026-03 | Y | §7 | **Conductor base candidates** Qwen3.5-0.8B/2B/4B/9B; Apache-2.0; 262K ctx |

## E. Open-source ingredients, RL trainers + serving

| # | Source | Org | Date | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- | --- |
| 27 | [Keep the Tokens Flowing (16 RL libs)](https://huggingface.co/blog/async-rl-training-landscape) | Hugging Face | 2026-03 | Y | §7 | Trainer decision matrix (verl, slime, AReaL, ROLL, PRIME-RL, OpenRLHF...) |
| 28 | [verl](https://github.com/verl-project/verl) | ByteDance | 2026 | Y | §7 | Most complete multi-turn agentic GRPO/DAPO trainer; vLLM/SGLang rollout |
| 29 | [slime](https://github.com/THUDM/slime) | THUDM | 2026 | Y | §7 | Async SGLang-native MoE RL; the framework that trained GLM-5 |
| 30 | [verifiers + Environments Hub + prime-rl](https://www.primeintellect.ai/blog/environments) | Prime Intellect | 2026-01 | Y | §7 | RLVR against any OpenAI-compatible endpoint; wraps TinyRouter's oracle as reward |
| 31 | [INTELLECT-3](https://arxiv.org/abs/2512.16144) (arXiv:2512.16144) | Prime Intellect | **2025-12** | **N (pre-2026)** | §7 | Open RLVR proof-point; FLAGGED: Dec-2025, borderline scope |
| 32 | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | OpenRLHF | 2026-04 (v0.10) | Y | §7 | Lighter Ray-based alt trainer; multi-turn RL |
| 33 | [SGLang](https://github.com/sgl-project/sglang) | SGLang | 2026-03 (v0.5.9) | Y | §7 | OpenAI-compatible serving; native multi-API; DeepSeek sparse-attn kernels |
| 34 | [llmhop](https://github.com/mirkolenz/llmhop) | Mirko Lenz | recency UNVERIFIED | ? | §7 | Thin OpenAI-compatible gateway candidate; **verify 2026 recency before use** |

## F. Press / independent / critical

| # | Source | Type | Date | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- | --- |
| 35 | [MarkTechPost launch](https://www.marktechpost.com/2026/06/22/sakana-ai-launches-sakana-fugu-an-orchestration-model-that-routes-tasks-across-a-swappable-pool-of-frontier-llms/) | press | 2026-06 | Y | §2, §6 | Benchmark table repro; ~500 beta users; model id `fugu-ultra-20260615` |
| 36 | [VentureBeat (Fugu launch)](https://venturebeat.com/orchestration/no-claude-fable-5-no-problem-sakana-achieves-frontier-performance-with-new-fugu-multi-model-auto-synthesis-system) | press | 2026-06 | Y | §2 | "auto synthesis" framing |
| 37 | [VentureBeat (Conductor deep-dive)](https://venturebeat.com/orchestration/how-sakana-trained-a-7b-model-to-orchestrate-gpt-5-claude-sonnet-4-and-gemini-2-5-pro) | press | 2026-06 | Y (date unverified, 403) | §3.2 | "How Sakana trained a 7B model to orchestrate..." |
| 38 | [Nikkei Asia](https://asia.nikkei.com/business/technology/artificial-intelligence/japan-s-sakana-fugu-multiagent-ai-scores-well-against-fable-5-gpt-5.5) | press | 2026-06 | Y | §6 | Independent coverage vs Fable 5 / GPT 5.5 |
| 39 | [The Decoder](https://the-decoder.com/sakana-ais-fugu-orchestrates-multiple-llms-to-match-anthropics-fable-and-mythos-benchmarks/) | press | 2026-06 | Y | §6 | States baselines are provider-reported; code review cited as a strength |
| 40 | [explainx.ai](https://www.explainx.ai/blog/sakana-fugu-multi-agent-orchestration-model-2026) | blog | 2026-06 | Y | §6 | Aggregates Mollick/Steinberger tests; 4-6x token fanout |
| 41 | [Paddo.dev](https://paddo.dev/blog/sakana-fugu-orchestration-model/) | blog | 2026-06 | Y | §6 | Sharpest skeptic; Fable 5 ~86 SWE; tiering anomaly; AI CUDA Engineer recall |
| 42 | [Digg (Bakouch critique)](https://digg.com/tech/2nguhqug) | press | 2026-06 | Y | §6 | "closed orchestrator over closed models is not sovereignty"; retraining critique |
| 43 | [Digg (Biderman challenge)](https://digg.com/tech/hm73dcfr) | press | 2026-06 | Y | §6 | Stella Biderman challenges Fugu Ultra claims + export-control framing |
| 44 | [TechTimes](https://www.techtimes.com/articles/318968/20260624/ai-orchestrator-sakana-fugu-claims-fable-5-parity-real-world-tests-reveal-30-minute-waits.htm) | press | 2026-06 | Y (body 403) | §6 | "30-minute waits" independent tests |
| 45 | [The New Stack](https://thenewstack.io/sakana-fugu-ai-sovereignty/) | press | 2026-06 | Y (body unretrievable) | §6 | "more than a router but not the blueprint for AI sovereignty" (framing only) |
| 46 | [Hacker News thread](https://news.ycombinator.com/item?id=48624782) | discussion | 2026-06 | Y | §6 | "premium model router with a very good marketing story" |
| 47 | [Fortune (export ban)](https://fortune.com/2026/06/13/anthropic-disables-fable-mythos-export-controls-national-security-threat/) | press | 2026-06 | Y | §2.5 | BIS "Is Informed" under ECRA disabled Fable 5/Mythos 5 on 2026-06-12 |
| 48 | [Anthropic statement](https://www.anthropic.com/news/fable-mythos-access) | official | 2026-06 | Y | §2.5 | Primary source for the export directive Fugu's pitch responds to |
| 49 | [paper_notes #5363 (Conductor)](https://github.com/AkihikoWatanabe/paper_notes/issues/5363) | github | 2026-04 | Y | §3.2 | Third-party paper note; venue corroboration (correct date 2026-04-26) |

## G. Open replications (study, do not trust headline numbers)

| # | Source | Date | In-scope | § | One-line |
| --- | --- | --- | --- | --- | --- |
| 50 | [trotsky1997/OpenFugu](https://github.com/trotsky1997/OpenFugu) | 2026-06 (creation unverified) | Y (caveat) | §5.4, §7 | **Closest sibling.** Qwen3-0.6B CMA-ES router + GRPO Conductor; Apache-2.0; mine `train/`,`verify/` |
| 51 | [HF: openfugu-conductor-3b](https://huggingface.co/di-zhang-fdu/openfugu-conductor-3b) | 2026-06 | Y | §5.4 | OpenFugu's published Conductor: Llama-3.2-3B GRPO'd on `nvidia/ToolScale` (real BF16 weights) |
| 52 | [nshkrdotcom/trinity_coordinator](https://github.com/nshkrdotcom/trinity_coordinator) | 2026-05 | Y | §5.4 | Elixir/Nx TRINITY router; consumes fixed artifacts; does not reproduce scores |
| 53 | [BicaMindLabs/open-sakanafugu](https://github.com/BicaMindLabs/open-sakanafugu) | 2026-06 (recency partial) | Y (caveat) | §5.4 | Hand-designed scaffold (9 LLMs + Codex reviewer); no trained orchestrator |
| 54 | [nvidia/ToolScale (HF dataset)](https://huggingface.co/datasets/nvidia/ToolScale) | 2026 | Y | §5.4 | Tool-use/orchestration dataset OpenFugu used to GRPO its Conductor |

## H. Pre-2026 lineage (OUT OF SCOPE, do not re-pull)

| Source | Date | Why excluded |
| --- | --- | --- |
| [xRouter](https://arxiv.org/abs/2510.08439) (arXiv:2510.08439) | 2025-10 | Salesforce RL cost-aware tool router; closest pre-Uno-Orchestra precedent; no 2026 venue found |
| [AgentRouter preprint](https://arxiv.org/abs/2510.05445) (arXiv:2510.05445) | 2025-10 | The 2025 preprint; the in-scope artifact is the ACL 2026 version (row 13) |
| Recursive Self-Aggregation (arXiv:2509.26626) | 2025-09 | recursive aggregation precedent |
| SVF / Transformer-squared | 2025 | the SVF lineage TinyRouter already uses |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (arXiv:2512.16144) | 2025-12 | open RLVR proof-point but Dec-2025; see row 31 |

---

## Recency and accuracy corrections (from the adversarial audit)

Carry these forward so the metadata stays honest:

- **Conductor and TRINITY originate December 2025** (arXiv `2512.*` prefix = Dec 2025). They are
  in-scope for 2026 only via their 2026 revisions (Conductor v5 2026-05-06; TRINITY v3 2026-04-27)
  and the ICLR 2026 venue. Cite the 2026 revision or the April 2026 Sakana blog as the anchor, not
  the Dec-2025 v1.
- **The Conductor's correct title** is "Learning to Orchestrate Agents in Natural Language with the
  Conductor" (one early draft note mislabeled it "Learning Natural-Language Coordination...").
- **AgentRouter's technical content is October 2025**; it is in-scope only because of the ACL 2026
  publication. The one related-lit item whose substance genuinely predates 2026.
- **INTELLECT-3 (2512.16144) is Dec-2025**, not 2026-01. Treat as borderline lineage.
- Date fixes: paper_notes #5363 is 2026-04-26 (not 2026-01); DeepVerifier v1 is 2026-01-22;
  "Can Heterogeneous LMs Be Fused?" v1 is 2026-04-02, v2 2026-05-16.
- Unverified recency to confirm before relying on: `llmhop`, and the creation/commit dates of
  OpenFugu and BicaMindLabs/open-sakanafugu.
- Unresolved product detail: the $200 "Max" tier is listed as 30x usage (one FAQ says 20x).
- `Sakana-AI-labs/Sakana-Fugu` (hyphenated org) is **not** the official `SakanaAI` org; treat as a
  possible mirror or impersonation, not a primary source.

## Angles already swept on 2026-06-25 (do not repeat)

1. Fugu product + architecture (official Sakana sources, GitHub).
2. The Conductor paper (the new replication target), full method extraction.
3. What Fugu adds on top of TRINITY (the four deltas).
4. Independent benchmarks + skeptical analysis (measured vs marketed; export-control context).
5. Related non-Sakana 2026 orchestration / RL-routing / mixture-of-agents literature.
6. Open-source replication ingredients (conductor bases, worker pool, RL frameworks, serving).

If a future research need falls outside these six angles (for example: a deep read of the Conductor
v5 full text to pin GRPO hyperparameters, a hands-on audit of OpenFugu's `train/`, or 2026 work
published after 2026-06-25), that is new ground. Everything above is covered.
