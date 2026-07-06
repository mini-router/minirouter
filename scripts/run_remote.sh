#!/usr/bin/env bash
# Run a trinity command on the remote GPU box, pinned to the configured GPU index.
# Usage (from local): bash scripts/run_remote.sh train --config configs/trinity.yaml
#                     bash scripts/run_remote.sh eval  --config configs/benchmarks.yaml
set -euo pipefail

HOST="${TRINITY_GPU_HOST:-trinity-gpu}"
REMOTE_DIR="${TRINITY_REMOTE_DIR:-trinity}"
GPU_INDEX="${TRINITY_GPU_INDEX:-0}"

CMD="$1"; shift
ssh "$HOST" \
  "export TRINITY_GPU_INDEX=$GPU_INDEX && cd $REMOTE_DIR && source .venv/bin/activate && \
   source scripts/remote_env.sh && \
   python -m trinity.$CMD $*"
