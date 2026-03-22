#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1
unset SSL_CERT_FILE || true

PYTHON_BIN="${SYN_PYTHON:-auto}"
if [[ "$PYTHON_BIN" == "auto" ]]; then
  if [[ -x "$HOME/miniconda3/envs/syn_data/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/syn_data/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

CFG_DEFAULT="$SCRIPT_DIR/conf/base_v2_breezy.yaml"
CFG="${CFG:-$CFG_DEFAULT}"

RUN_ROOT="${RUN_ROOT:-${1:-}}"
if [[ -z "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT is required" >&2
  echo "Usage: RUN_ROOT=/work/... ./push_to_hf.sh" >&2
  echo "   or: ./push_to_hf.sh /work/..." >&2
  exit 2
fi

if [[ "${1:-}" == "$RUN_ROOT" ]]; then
  shift || true
fi

# Defaults (override via env)
TOPICS="${TOPICS:-auto}"            # auto = scan topics with wav outputs
STRICT="${STRICT:-1}"               # 1=true, 0=false
NO_PUSH="${NO_PUSH:-0}"             # 1=true
REPO_ID="${REPO_ID:-}"              # optional override
REVISION="${REVISION:-}"            # optional branch

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[deps] Installing pinned deps (INSTALL_DEPS=1)..." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3 \
    || echo "[deps][WARN] Failed to install antlr4-python3-runtime; continuing." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps transformers==4.52.1 \
    || echo "[deps][WARN] Failed to install transformers; continuing." >&2
fi

args=(
  --config "$CFG"
  --run-root "$RUN_ROOT"
  --export
  --topics "$TOPICS"
)

if [[ "$STRICT" == "1" ]]; then
  args+=(--strict)
fi
if [[ "$NO_PUSH" == "1" ]]; then
  args+=(--no-push)
fi
if [[ -n "$REPO_ID" ]]; then
  args+=(--repo-id "$REPO_ID")
fi
if [[ -n "$REVISION" ]]; then
  args+=(--revision "$REVISION")
fi

"$PYTHON_BIN" push_dataset_to_hub.py "${args[@]}" "$@"
