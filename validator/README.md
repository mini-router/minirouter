# minirouter-validator

Backend service for PR intake, checkpoint evaluation, and leaderboard storage for the
MiniRouter competition. This subtree is copied into the main `minirouter` repo so the
validator, worker, and API live alongside the router code. Use the repo-root
`secrets.env` for both the router and the validator. The validator backend
requires Postgres, so `DATABASE_URL` must be a PostgreSQL connection string.

## What it does

- accepts PR webhooks from the miner repo
- accepts direct archive uploads from the frontend `/submit` form
- stores submission metadata and artifact hashes in a database
- queues each submission in Postgres and lets the worker evaluate it
- runs a configurable submission-only evaluation command against a checkpoint
- records evaluation runs and exposes leaderboard data for the public site
- posts a formatted PR summary comment after evaluation
- can auto-merge evaluated miner submission PRs when enabled

## Layout

- `src/eval_backend/main.py` - FastAPI app
- `src/eval_backend/worker.py` - polling worker that evaluates queued submissions
- `src/eval_backend/models.py` - SQLAlchemy tables
- `src/eval_backend/services/eval_runner.py` - clone/extract/run-eval orchestration
- `src/eval_backend/api/routes.py` - HTTP endpoints used by the website and webhooks

## Default contract

The backend expects trained checkpoints like `best_theta.npy` and evaluates them in
two stages by default:

1. remote GPU over SSH on the `TRINITY_GPU_HOST` box
2. local CPU fallback on this machine if the remote path fails

The command templates can be overridden, but the default setup assumes the remote box
already has the `minirouter` repo synced and its own `.venv` prepared. The local
fallback uses the local `MINIROUTER_REPO_DIR` checkout.

Example command templates:

```bash
cd {repo_dir} && source .venv/bin/activate && \
PYTHONPATH=src python -m trinity.eval --submission-only \
  --benchmark {benchmark} \
  --provider {provider} \
  --models {models_config} \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-items {max_items} \
  --batch-size {eval_batch_size} \
  --theta {checkpoint_path} \
  --out {results_path}
```

Set `EVAL_RESULT_POINTER=results.TRINITY` if the command writes JSON like the current
`trinity.eval` harness.

For local fallback, point `MINIROUTER_REPO_DIR` at a local checkout of the miner repo.
For remote execution, set `TRINITY_GPU_HOST`, `TRINITY_REMOTE_DIR`, and
`TRINITY_REMOTE_WORKSPACE_ROOT` to the SSH host alias, remote repo checkout, and
remote temp workspace root. The remote box should have its own repo-root `secrets.env` too.

The runner does not duplicate evaluation logic. It shells out to the submission-only
`python -m trinity.eval` path from the copied `minirouter` repository, so the benchmark
code stays in one place.

## Local run

```bash
python -m venv .venv
.venv/bin/pip install -e .
uvicorn eval_backend.main:app --reload
python -m eval_backend.worker --loop
```

Run the API and the worker as separate processes. The API only stores queued
submissions; the worker claims them and advances the progress fields.
The submission API exposes the current phase, message, and item counters so the
GitHub workflow can show live status while it waits.
Set `EVAL_BATCH_SIZE` to control how many benchmark items the evaluator runs
concurrently. The worker passes it through to both remote GPU and local fallback
commands.

Example `DATABASE_URL`:

```bash
DATABASE_URL=postgresql+psycopg://minirouter:minirouter@127.0.0.1:5432/minirouter
```

For workflow smoke checks, set `EVAL_MAX_ITEMS=1` in the repo-root `secrets.env`.
Set `PIPELINE_MODE=submission_eval` to keep the current "submit a trained
checkpoint and evaluate it" path, or `PIPELINE_MODE=train_eval` to switch the
backend into the PR-code training path. In train mode, the uploaded archive is
treated as source code, the worker trains a router on the server, and the
resulting checkpoint is evaluated afterward. The automation should upload
`submissions/final_model/` in submission mode and the PR source bundle in train
mode.

GitHub PR automation uses the repo-root `GITHUB_WEBHOOK_SECRET` plus the optional
`GITHUB_ACCESS_TOKEN`. Set `PUBLIC_SITE_URL` if the public frontend moves to a different domain.
The webhook intake is idempotent for repeated events on the same PR, and the worker evaluates one
queued submission at a time.

The PR workflow now uploads `submissions/final_model/` as a tarball to
`POST /submit`, which stores the archive, extracts `best_theta.npy`, and queues the submission for
evaluation.

To enable automatic merge after evaluation, set `GITHUB_AUTO_MERGE_SUBMISSIONS=true`.
To keep only the PR comment and skip auto-merge, leave it `false`.

The public website can read `GET /api/leaderboard` and `GET /api/submissions/{id}`.
