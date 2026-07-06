# minirouter-validator

Backend service for PR intake, checkpoint evaluation, and leaderboard storage for the
MiniRouter competition. This subtree is copied into the main `minirouter` repo so the
validator, worker, and API live alongside the router code.

## What it does

- accepts PR webhooks from the miner repo
- accepts direct archive uploads from the frontend `/submit` form
- stores submission metadata and artifact hashes in a database
- runs a configurable evaluation command against a checkpoint
- records evaluation runs and exposes leaderboard data for the public site

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
PYTHONPATH=src python -m trinity.eval \
  --benchmark {benchmark} \
  --provider {provider} \
  --models {models_config} \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-items {max_items} \
  --theta {checkpoint_path} \
  --out {results_path}
```

Set `EVAL_RESULT_POINTER=results.TRINITY` if the command writes JSON like the current
`trinity.eval` harness.

For local fallback, point `MINIROUTER_REPO_DIR` at a local checkout of the miner repo.
For remote execution, set `TRINITY_GPU_HOST`, `TRINITY_REMOTE_DIR`, and
`TRINITY_REMOTE_WORKSPACE_ROOT` to the SSH host alias, remote repo checkout, and
remote temp workspace root. The remote box should have its own `secrets.env` too.

The runner does not duplicate evaluation logic. It shells out to `python -m trinity.eval`
from the copied `minirouter` repository, so the benchmark code stays in one place.

## Local run

```bash
cp secrets.env.example secrets.env
python -m venv .venv
.venv/bin/pip install -e .
uvicorn eval_backend.main:app --reload
python -m eval_backend.worker --loop
```

The public website can read `GET /api/leaderboard` and `GET /api/submissions/{id}`.
