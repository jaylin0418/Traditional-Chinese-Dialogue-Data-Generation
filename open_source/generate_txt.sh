#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep the runtime environment clean/reproducible.
unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1

# SSL CA handling
# Some clusters set SSL_CERT_FILE to a broken path; others have no system CA store.
# Strategy:
#   - If SSL_CERT_FILE points to a non-existent file, unset it.
#   - If SSL_CERT_FILE is unset, default it to certifi's CA bundle (if available).
# You can opt out by setting KEEP_SSL_CERT_FILE=1.
if [[ "${KEEP_SSL_CERT_FILE:-0}" != "1" ]]; then
  if [[ -n "${SSL_CERT_FILE:-}" && ! -f "${SSL_CERT_FILE}" ]]; then
    unset SSL_CERT_FILE || true
  fi
fi

# Default LLM timeout + retry policy.
# You can override any of these via env vars.
export OPENROUTER_TIMEOUT_SECONDS="${OPENROUTER_TIMEOUT_SECONDS:-120}"
export OPENROUTER_REQUEST_MAX_RETRIES="${OPENROUTER_REQUEST_MAX_RETRIES:-20}"
export OPENROUTER_RETRY_BACKOFF_BASE="${OPENROUTER_RETRY_BACKOFF_BASE:-2}"
export OPENROUTER_RETRY_BACKOFF_CAP="${OPENROUTER_RETRY_BACKOFF_CAP:-300}"

# Choose interpreter. Override via:
#   SYN_PYTHON=/path/to/python ./generate_txt.sh ...
PYTHON_BIN="${SYN_PYTHON:-auto}"
if [[ "$PYTHON_BIN" == "auto" ]]; then
  if [[ -x "$HOME/miniconda3/envs/syn_data/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/syn_data/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ "${KEEP_SSL_CERT_FILE:-0}" != "1" ]]; then
  if [[ -z "${SSL_CERT_FILE:-}" ]]; then
    CERTIFI_CA="$($PYTHON_BIN -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [[ -n "$CERTIFI_CA" && -f "$CERTIFI_CA" ]]; then
      export SSL_CERT_FILE="$CERTIFI_CA"
    fi
  fi
fi

CFG_DEFAULT="$SCRIPT_DIR/conf/base_v2_breezy.yaml"
CFG="${CFG:-$CFG_DEFAULT}"

# Parse CLI args
WORKERS="${WORKERS:-10}"
PER_TOPIC_COUNT="${PER_TOPIC_COUNT:-220}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/work/jaylin0418}"
OVERLAP_FILLER="${OVERLAP_FILLER:-false}"
WITH_EMOTION="${WITH_EMOTION:-false}"
TOPICS="${TOPICS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)          WORKERS="$2";        shift 2 ;;
    --overlap-filler)   OVERLAP_FILLER="$2"; shift 2 ;;
    --with-emotion)     WITH_EMOTION="true"; shift 1 ;;
    --per-topic-count)  PER_TOPIC_COUNT="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2";     shift 2 ;;
    --output-root-base) OUTPUT_ROOT_BASE="$2"; shift 2 ;;
    --topics)           TOPICS="$2";         shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./generate_txt.sh [OPTIONS]

Options:
  --workers N              Number of parallel workers (default: 10)
  --overlap-filler true|false
                           Include overlap and filler stages (default: false)
  --with-emotion           Add (emotion:xxx) tags to each dialogue turn (default: off)
  --per-topic-count N      Scenarios per topic (default: 220)
  --batch-size N           Batch size (default: 4)
  --output-root-base PATH  Output root directory (default: /work/jaylin0418)
  --topics TOPIC1,TOPIC2   Comma-separated topic list (default: all topics)

Environment overrides:
  WORKERS, OVERLAP_FILLER, WITH_EMOTION, PER_TOPIC_COUNT, BATCH_SIZE,
  OUTPUT_ROOT_BASE, TOPICS, CFG, SYN_PYTHON
EOF
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Build STAGES based on --overlap-filler
if [[ "${OVERLAP_FILLER,,}" == "true" || "$OVERLAP_FILLER" == "1" ]]; then
  STAGES="scenario,system_prompt,dialogue,overlap,filler"
else
  STAGES="scenario,system_prompt,dialogue"
fi

# Optional dependency hotfix. Some environments can't write site-packages and
# also disable user site-packages, which makes `pip install --user` impossible.
# Default: skip. Enable explicitly if you need it.
INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[deps] Installing pinned deps (INSTALL_DEPS=1)..." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3 \
    || echo "[deps][WARN] Failed to install antlr4-python3-runtime; continuing." >&2
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps transformers==4.52.1 \
    || echo "[deps][WARN] Failed to install transformers; continuing." >&2
fi

echo "Running TXT-only multi-topic generation..."
echo "OPENROUTER_TIMEOUT_SECONDS=$OPENROUTER_TIMEOUT_SECONDS"
echo "OPENROUTER_REQUEST_MAX_RETRIES=$OPENROUTER_REQUEST_MAX_RETRIES"
echo "SSL_CERT_FILE=${SSL_CERT_FILE:-<unset>}"
echo "CFG=$CFG"
echo "workers=$WORKERS per_topic_count=$PER_TOPIC_COUNT batch_size=$BATCH_SIZE"
echo "overlap_filler=$OVERLAP_FILLER => stages=$STAGES"
echo "with_emotion=$WITH_EMOTION"
echo "output_root_base=$OUTPUT_ROOT_BASE"
if [[ -n "$TOPICS" ]]; then
  echo "topics=$TOPICS"
fi

auto_args=(
  --config "$CFG"
  --workers "$WORKERS"
  --per-topic-count "$PER_TOPIC_COUNT"
  --batch-size "$BATCH_SIZE"
  --output-root-base "$OUTPUT_ROOT_BASE"
  --stages "$STAGES"
)

if [[ -n "$TOPICS" ]]; then
  auto_args+=(--topics "$TOPICS")
fi

if [[ "${WITH_EMOTION,,}" == "true" || "$WITH_EMOTION" == "1" ]]; then
  auto_args+=(--with-emotion)
fi

# Any CLI args passed to this script are appended, so you can override defaults.
"$PYTHON_BIN" run_multi_topic_txt_workers.py "${auto_args[@]}" "$@"
