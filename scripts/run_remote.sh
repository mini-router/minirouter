#!/usr/bin/env bash
# Run a trinity command on the remote GPU box, pinned to the configured GPU index.
# Usage (from local): bash scripts/run_remote.sh train --config configs/trinity.yaml
#                     bash scripts/run_remote.sh eval  --config configs/benchmarks.yaml
set -euo pipefail

HOST="${TRINITY_GPU_HOST:-trinity-gpu}"
REMOTE_DIR="${TRINITY_REMOTE_DIR:-trinity}"
GPU_INDEX="${TRINITY_GPU_INDEX:-0}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYNC_DIR="${TRINITY_SYNC_DIR:-$LOCAL_DIR}"
SYNC_ENABLED="${TRINITY_SYNC_ENABLED:-1}"

if [[ $# -lt 1 ]]; then
  echo "[run_remote] usage: bash scripts/run_remote.sh <train|eval> [args...]" >&2
  exit 64
fi

CMD="$1"; shift
case "$CMD" in
  train|eval) ;;
  *)
    echo "[run_remote] unsupported command '$CMD' (expected train or eval)" >&2
    exit 64
    ;;
esac

REMOTE_ARGS="$(printf ' %q' "$@")"
REMOTE_DIR_Q="$(printf '%q' "$REMOTE_DIR")"
if ssh "$HOST" \
  "export TRINITY_GPU_INDEX=$GPU_INDEX TRINITY_REMOTE_DIR=$REMOTE_DIR_Q && cd $REMOTE_DIR_Q && \
   source .venv/bin/activate && source scripts/remote_env.sh && \
   python -m trinity.$CMD$REMOTE_ARGS"; then
  :
else
  rc=$?
  echo "[run_remote] remote $CMD failed with exit code $rc" >&2
  exit "$rc"
fi

if [[ "$SYNC_ENABLED" == "0" ]]; then
  echo "[run_remote] sync disabled"
  exit 0
fi

sync_dir() {
  local remote_subdir="$1"
  local local_subdir="$2"
  if ssh "$HOST" "test -d $(printf '%q' "$REMOTE_DIR/$remote_subdir")"; then
    echo "[run_remote] syncing $remote_subdir -> $local_subdir"
    mkdir -p "$SYNC_DIR/$local_subdir"
    rsync -az "$HOST:$REMOTE_DIR/$remote_subdir/" "$SYNC_DIR/$local_subdir/"
  fi
}

sync_file() {
  local remote_file="$1"
  local local_file="$2"
  if ssh "$HOST" "test -f $(printf '%q' "$REMOTE_DIR/$remote_file")"; then
    echo "[run_remote] syncing $remote_file -> $local_file"
    mkdir -p "$(dirname "$SYNC_DIR/$local_file")"
    rsync -az "$HOST:$REMOTE_DIR/$remote_file" "$SYNC_DIR/$local_file"
  fi
}

sync_dir experiments experiments
sync_dir test_logs test_logs
sync_file cost_ledger.jsonl cost_ledger.jsonl

echo "[run_remote] sync complete -> $SYNC_DIR"
