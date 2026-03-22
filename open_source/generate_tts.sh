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

# Required: RUN_ROOT (either env var or first arg)
RUN_ROOT="${RUN_ROOT:-${1:-}}"
if [[ -z "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT is required" >&2
  echo "Usage: RUN_ROOT=/work/... ./generate_tts.sh" >&2
  echo "   or: ./generate_tts.sh /work/..." >&2
  exit 2
fi

# If RUN_ROOT came from $1, shift so remaining args can be passed through.
if [[ "${1:-}" == "$RUN_ROOT" ]]; then
  shift || true
fi

# Defaults (override via env)
GPUS="${GPUS:-}"                    # e.g. 0,1,2,3 (empty = auto-detect)
TOPICS="${TOPICS:-}"                # empty = infer from run_root
TOPICS_PER_GPU="${TOPICS_PER_GPU:-2}"
CONCURRENT_PER_GPU="${CONCURRENT_PER_GPU:-1}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[deps] Installing pinned deps (INSTALL_DEPS=1)..." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3 \
    || echo "[deps][WARN] Failed to install antlr4-python3-runtime; continuing." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps transformers==4.52.1 \
    || echo "[deps][WARN] Failed to install transformers; continuing." >&2
fi

echo "Running multi-topic TTS..."
echo "RUN_ROOT=$RUN_ROOT"
echo "CFG=$CFG"
echo "topics_per_gpu=$TOPICS_PER_GPU concurrent_per_gpu=$CONCURRENT_PER_GPU"
if [[ -n "$GPUS" ]]; then
  echo "gpus=$GPUS"
fi
if [[ -n "$TOPICS" ]]; then
  echo "topics=$TOPICS"
fi

args=(
  --config "$CFG"
  --run-root "$RUN_ROOT"
  --topics-per-gpu "$TOPICS_PER_GPU"
  --concurrent-per-gpu "$CONCURRENT_PER_GPU"
)

if [[ -n "$GPUS" ]]; then
  args+=(--gpus "$GPUS")
fi
if [[ -n "$TOPICS" ]]; then
  args+=(--topics "$TOPICS")
fi

"$PYTHON_BIN" run_multi_topic_tts_workers.py "${args[@]}" "$@"
