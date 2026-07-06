#!/usr/bin/env bash
# Run a trinity command on the remote GPU box, pinned to GPU 5.
# Usage (from local): bash scripts/run_remote.sh train --config configs/trinity.yaml
#                     bash scripts/run_remote.sh eval  --config configs/benchmarks.yaml
set -euo pipefail

HOST="${TRINITY_GPU_HOST:-trinity-gpu}"
REMOTE_DIR="${TRINITY_REMOTE_DIR:-trinity}"

if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
  echo "ERROR: source ~/.config/trinity/secrets.env first." >&2; exit 1
fi

CMD="$1"; shift
# Forward the API key over the SSH channel as an env var (never written to disk on the box).
ssh "$HOST" \
  "cd $REMOTE_DIR && source .venv/bin/activate && \
   export FIREWORKS_API_KEY='$FIREWORKS_API_KEY' TRINITY_REMOTE_DIR='$REMOTE_DIR' && \
   source scripts/remote_env.sh && \
   python -m trinity.$CMD $*"
