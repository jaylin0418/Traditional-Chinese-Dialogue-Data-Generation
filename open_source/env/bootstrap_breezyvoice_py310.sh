#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-breezyvoice_py310}"
LOCK_FILE="$SCRIPT_DIR/conda-linux-64.py310.lock"
HISTORY_YML="$SCRIPT_DIR/environment.py310.from-history.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[bootstrap] conda env '$ENV_NAME' already exists; skipping create." >&2
else
  if [[ -f "$LOCK_FILE" ]]; then
    echo "[bootstrap] Creating '$ENV_NAME' from explicit lock: $LOCK_FILE" >&2
    conda create -y -n "$ENV_NAME" --file "$LOCK_FILE"
  elif [[ -f "$HISTORY_YML" ]]; then
    echo "[bootstrap] Creating '$ENV_NAME' from history yml: $HISTORY_YML" >&2
    conda env create -n "$ENV_NAME" -f "$HISTORY_YML"
  else
    echo "ERROR: Neither lock nor history yml found under $SCRIPT_DIR" >&2
    exit 2
  fi
fi

conda activate "$ENV_NAME"

# Only do pip installs when using from-history (or when user asks).
POST_PIP_INSTALLS="${POST_PIP_INSTALLS:-0}"
if [[ "$POST_PIP_INSTALLS" == "1" ]] || [[ ! -f "$LOCK_FILE" ]]; then
  echo "[bootstrap] Installing pip requirements (POST_PIP_INSTALLS=1 or no lockfile)..." >&2
  python -m pip install -U pip
  python -m pip install -r "$PROJECT_DIR/requirements.txt"
  python -m pip install -r "/home/jaylin0418/SpeechLab/tts_model/BreezyVoice/requirements.txt"
fi

echo "[bootstrap] Done. Active env: $ENV_NAME" >&2
python -V
