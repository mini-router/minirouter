<img width="1672" height="941" alt="ChatGPT Image Jul 7, 2026, 06_04_14 AM" src="https://github.com/user-attachments/assets/b9184d3e-a3f8-41bb-acca-068dd573f752" />

# MiniRouter

MiniRouter is the **SN74** miner workspace for the **Gittensor** LLM routing competition. This repo was
adapted from the original TinyRouter project at https://github.com/harrrshall/tinyrouter/ and then
expanded into the competition workspace used here. The goal is not to build one giant model, but to
learn a better router: for each question, decide **which** model should answer and **what role** it
should play. Miners can improve the trainer, the evaluation pipeline, the model pool, or the web app
that publishes results.

The core router is deliberately tiny and cheap. A frozen **0.6B** encoder reads the question into a
single vector, and a **~10K-parameter** head turns that vector into the routing decision. It is
trained by **separable CMA-ES**, a derivative-free evolution strategy, against a simple right/wrong
reward. The coordinator never solves the question itself; it only learns who to ask.

The method follows TRINITY (Xu et al., ICLR 2026, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
rebuilt from scratch with an open model pool and the miner-facing competition tooling in this repo.

## What we did

- Implemented the full coordinator: the 0.6B encoder feature, the ~10K routing head, the three roles,
  the multi-turn loop (up to 5 turns, terminated by a Verifier accept), and the sep-CMA-ES trainer.
- Wired a 3-model open-source pool plus an automatic grader (exact-match for math, letter-match for
  MMLU, pass@1 code execution for LiveCodeBench) that produces the binary reward.
- Trained per-task coordinators by evolution: breed thousands of candidate heads, keep the ones that
  route best, repeat.
- Evaluated rigorously on 120 held-out questions, with every single-model baseline averaged over 3 runs
  to remove run-to-run noise, against each model alone and against random routing.
- Built an **oracle-ceiling diagnostic** to ask whether the pool even leaves room for routing to help,
  and used it to decide where improvement effort was worth spending.
- Implemented and tested two upgrades from that diagnostic (supervised warm-start of the head, shaped
  training fitness) and measured them on the task with real headroom.
- Tracked every dollar of API spend and logged each result.

## Repository map

- `src/trinity/` - training, evaluation, routing, and reward code
- `configs/` - benchmark and model-pool configuration
- `benchmarks/` - benchmark loaders and task definitions
- `scripts/` - local and remote training/eval entry points
- `experiments/` - run outputs and saved training artifacts
- `submissions/` - submit-ready bundles for the evaluation backend
- `validator/` - competition backend that ingests submissions and runs eval
- `web/` - public competition site and leaderboard frontend
- `docs/` - research notes, results, and implementation notes

## Miner workflow

Miners should work in their own branch, keep changes scoped, and open PRs for review. The branch
prefix rule is documented in `CONTRIBUTOIN.md`, and the submit-ready model bundle should stay in
`submissions/final_model/` when a run is ready to evaluate or publish.

The repository also includes a GitHub Actions PR automation workflow. It labels PRs by path,
tags miner submission PRs, packages `submissions/final_model/`, uploads the bundle through the
public `/submit` endpoint, waits for the worker to advance the submission status, comments back the
result, and then merges the PR when the run completes.
No separate GitHub bot is required for that flow.

The validator backend stores submissions and evaluation runs in Postgres. Set
`DATABASE_URL` in the repo-root `secrets.env` before starting the API or worker.
Use `PIPELINE_MODE=submission_eval` for the current checkpoint-evaluation flow, or
`PIPELINE_MODE=train_eval` to switch the validator into the server-side
train-then-evaluate flow for PR code submissions.

## Model pool

| Slot | Model            | Strong at        |
| ---- | ---------------- | ---------------- |
| A    | `deepseek-v4-pro` | knowledge (MMLU) |
| B    | `glm-5p2`         | math             |
| C    | `kimi-k2p6`       | general          |

The 0.6B encoder and the evolution loop run on a single NVIDIA H200; the three LLMs are called over HTTP.

## How it works

1. The frozen 0.6B encoder turns the question into one 1024-dim vector.
2. The ~10K head reads that vector and picks a model and a role.
3. The chosen model answers in that role; its output is appended to the transcript.
4. Steps 1 to 3 repeat for up to 5 turns; a Verifier turn can accept and stop early.
5. The final answer is graded right/wrong, and that reward drives the evolutionary training.

## Results

Rigorous eval: 120 held-out questions per task; single-model baselines are the mean over 3 runs.
Scores are fraction correct (0.792 = 79.2%).

**Math**

| system | score |
| --- | --- |
| glm-5p2 | 0.794 (best single) |
| **TinyRouter** | **0.792** |
| random routing | 0.792 |
| deepseek-v4-pro | 0.747 |
| kimi-k2p6 | 0.742 |

**MMLU**

| system | score |
| --- | --- |
| **TinyRouter** | **0.925** |
| deepseek-v4-pro | 0.922 (best single) |
| random routing | 0.875 |
| glm-5p2 | 0.783 |
| kimi-k2p6 | 0.539 |

**Both tasks together**

| system | math | MMLU | average |
| --- | --- | --- | --- |
| **TinyRouter** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 | 0.922 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | 0.794 | 0.783 | 0.789 |
| kimi-k2p6 | 0.742 | 0.539 | 0.640 |

### What the numbers say

The tiny router scores **0.858 on average, higher than any single model**. No single model is good at
both tasks: deepseek is the knowledge specialist, glm is the math specialist. The router wins the
average by sending each task to the right specialist.

Reading it straight: the win is **across tasks, not within a task**. On MMLU, where the models differ a
lot (0.54 to 0.92), routing clearly helps and the router beats random (0.925 vs 0.875). On math, where
all three models sit around 0.79, there is nothing to route around, so the router ties both the best
model and random routing. Routing pays off when the models genuinely differ.

## Can routing do better? (oracle-ceiling diagnostic)

A tie on math could mean two very different things: either the pool has no headroom (every model is
equally good or bad on the same questions), or the headroom exists but our router fails to capture it.
To tell them apart we built a diagnostic that estimates the best score a **perfect** query-conditional
router could reach, debiased for the winner's-curse with split-half cross-fit, and read the verdict off
bootstrap confidence intervals rather than point estimates.

| benchmark | best single | perfect router | real headroom (95% CI) | verdict |
| --- | --- | --- | --- | --- |
| math500 | 0.808 | **0.856** | **+0.049 [0.005, 0.085]** | ROUTER_BOUND |
| MMLU | 0.939 | ≥0.939 | +0.025 [0.000, 0.058] | inconclusive (near-ceiling) |

This overturned the easy reading of math as "no benefit." There **is** about 4.9 points of real,
achievable headroom on math; our trained router just captures none of it. So the math limit is the
router, not the pool. MMLU sits near its ceiling, where deepseek already dominates and the router
already matches it.

## Trying to capture it: warm-start + shaped fitness

The diagnostic pointed effort at math, so we tried two upgrades: warm-starting the head with a
supervised fit against per-(question, model) correctness labels (instead of starting the evolution from
a blank head), and shaping the training reward (format bonus, turn penalty, variance reweighting) while
keeping the **eval pure right/wrong**.

| system | math (held-out 120) |
| --- | --- |
| best single (glm-5p2) | 0.817 |
| **TinyRouter (warm-start + shaped)** | **0.808** |
| prior router, same test | 0.792 |
| random routing | 0.733 |

The retrained router scored 0.808 vs the prior 0.792, but we read this as **inconclusive, not a win**.
The eval samples each model once per question, and that sampling noise is large: random routing alone
swung from 0.792 to 0.733 between runs with nothing changed. A swing that size swamps a 1.6-point router
delta. We did not run the clean control (blank-init, pure-binary, same settings), so there is no causal
claim that warm-start or shaping moved the number. The result is still below the best single model
(0.817) and below the 0.856 ceiling, so the headroom the diagnostic found remains on the table. The two
upgrades are implemented and covered by 54 offline tests; whether they move the held-out score is
unproven.

## Submitting a final model

Use `submissions/final_model/` as the submit-ready bundle for your final checkpoint and metadata.
Keep the trained checkpoint and the JSON files together in that folder, then open a PR from your
required `sn74-<miner-name>` branch.

Expected contents:

- `best_theta.npy` - the trained routing head
- `summary.json` - training summary for the run
- `history.json` - optional per-generation history
- `eval.json` - optional local evaluation output

Typical workflow:

```bash
git checkout main
git pull upstream main
git checkout -b sn74-your-github-username

mkdir -p submissions/final_model
cp experiments/math500/<run-name>/best_theta.npy submissions/final_model/
cp experiments/math500/<run-name>/summary.json submissions/final_model/
cp experiments/math500/<run-name>/history.json submissions/final_model/  # optional

python utility/validate_submission.py --dir submissions/final_model

python -m trinity.eval \
  --benchmark math500 \
  --theta submissions/final_model/best_theta.npy \
  --provider chutes \
  --models configs/models.chutes.yaml \
  --device cpu \
  --dtype float32 \
  --out submissions/final_model/eval.json

git add submissions/final_model
git commit -m "Add final model bundle"
git push origin sn74-your-github-username
```

Open a pull request from your branch. The validator and maintainer workflow will pick up the bundle
from the PR, evaluate it, and record the result.

## Continuous integration

The CI workflow lives in `.github/workflows/ci.yml` and runs on every pull request and on push to
`main`. It does not require any secrets or a database.

- root package: `pip install -e ".[dev]"` then `pytest tests -q`
- `validator/`: `pip install -e ".[dev]"` then `pytest -q`
- `web/`: `npm ci`, `npm run build` (type-checks via `tsc -b`), `npm run lint`

This catches a broken test suite or web build on the PR itself, instead of only after merge when
`deploy-web.yml` runs against `main`.

## PR automation

The PR automation workflow lives in `.github/workflows/pr-automation.yml`.

It:

- adds labels such as `web`, `validator`, `train`, `eval`, `benchmark`, `docs`, `miner`, and `submission`
- queues miner submission PRs in the validator backend
- leaves evaluation and merge handling to the backend service

Repository setup:

- set `GITHUB_WEBHOOK_SECRET` in the validator `secrets.env`
- set `MINIROUTER_WEBHOOK_SECRET` as a GitHub Actions secret with the same value
- optionally set `BACKEND_BASE_URL` as a repository variable if the backend URL changes
- the workflow posts submission archives to `POST /submit`
- set `EVAL_PROVIDER=chutes` and `EVAL_MODELS_CONFIG=configs/models.chutes.yaml` for default
  validator evals using the Chutes pool
