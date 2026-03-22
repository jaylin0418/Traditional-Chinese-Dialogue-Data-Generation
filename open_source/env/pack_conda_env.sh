#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-breezyvoice_py310}"
OUT_FILE="${OUT_FILE:-$SCRIPT_DIR/${ENV_NAME}.tar.gz}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda env '$ENV_NAME' not found" >&2
  exit 2
fi

if ! command -v conda-pack >/dev/null 2>&1; then
  echo "[pack] Installing conda-pack into base env..." >&2
  conda install -y -n base -c conda-forge conda-pack
fi

echo "[pack] Packing env '$ENV_NAME' -> $OUT_FILE" >&2
cd "$PROJECT_DIR"
conda pack -n "$ENV_NAME" -o "$OUT_FILE"

echo "[pack] Done." >&2
ls -lh "$OUT_FILE"
