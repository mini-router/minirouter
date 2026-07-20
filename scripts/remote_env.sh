#!/usr/bin/env bash
# Shared environment for all remote-GPU commands.
# Default to the allocated GPU index (5) unless overridden.
set -euo pipefail

# We are allocated exactly one physical GPU on the remote box.
export CUDA_VISIBLE_DEVICES="${TRINITY_GPU_INDEX:-5}"

# Project location on the remote box (overridable).
export TRINITY_REMOTE_DIR="${TRINITY_REMOTE_DIR:-$HOME/trinity}"

# Keep HF caches local to the project so we don't pollute shared $HOME.
export HF_HOME="${HF_HOME:-$TRINITY_REMOTE_DIR/.cache/hf}"
export TRANSFORMERS_CACHE="$HF_HOME"

echo "[remote_env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  TRINITY_REMOTE_DIR=$TRINITY_REMOTE_DIR"
