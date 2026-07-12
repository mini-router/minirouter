#!/usr/bin/env bash
# Provision the Python env on the remote GPU box and verify the configured GPU is visible.
# Run from the LOCAL machine: `bash scripts/setup_remote.sh`
# It rsyncs the repo to the box and sets up a uv venv there.
set -euo pipefail

HOST="${TRINITY_GPU_HOST:-trinity-gpu}"
REMOTE_DIR="${TRINITY_REMOTE_DIR:-trinity}"
GPU_INDEX="${TRINITY_GPU_INDEX:-0}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${TRINITY_SECRETS_FILE:-}" ]]; then
  LOCAL_SECRETS_FILE="$TRINITY_SECRETS_FILE"
else
  if [[ -f "$LOCAL_DIR/secrets.env" ]]; then
    LOCAL_SECRETS_FILE="$LOCAL_DIR/secrets.env"
  elif [[ -f "$HOME/.config/trinity/secrets.env" ]]; then
    LOCAL_SECRETS_FILE="$HOME/.config/trinity/secrets.env"
  else
    LOCAL_SECRETS_FILE=""
  fi
fi

if [[ -z "$LOCAL_SECRETS_FILE" || ! -f "$LOCAL_SECRETS_FILE" ]]; then
  echo "[setup_remote] missing local secrets file; set TRINITY_SECRETS_FILE or create $LOCAL_DIR/secrets.env or ~/.config/trinity/secrets.env" >&2
  exit 1
fi

echo "[setup_remote] syncing $LOCAL_DIR -> $HOST:$REMOTE_DIR (excluding secrets/artifacts)"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '.venv*' --exclude '__pycache__' \
  --exclude 'experiments' --exclude '*.pdf' --exclude '.env' --exclude 'secrets.env' \
  --exclude '*.npy' --exclude 'docs/paper' \
  "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

echo "[setup_remote] syncing secrets file $LOCAL_SECRETS_FILE -> $HOST:$REMOTE_DIR/secrets.env"
rsync -az --chmod=F600 "$LOCAL_SECRETS_FILE" "$HOST:$REMOTE_DIR/secrets.env"

echo "[setup_remote] installing env + checking GPU ${GPU_INDEX} on $HOST"
ssh "$HOST" "TRINITY_REMOTE_DIR=$REMOTE_DIR TRINITY_GPU_INDEX=$GPU_INDEX bash -s" <<'REMOTE'
set -euo pipefail
cd "${TRINITY_REMOTE_DIR:-trinity}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv 2>/dev/null || true
source .venv/bin/activate
uv pip install -e . >/dev/null
echo "--- GPU ${TRINITY_GPU_INDEX:-0} ---"
CUDA_VISIBLE_DEVICES="${TRINITY_GPU_INDEX:-0}" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
REMOTE
echo "[setup_remote] done."
