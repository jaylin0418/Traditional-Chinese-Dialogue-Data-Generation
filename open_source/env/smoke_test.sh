#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

python - <<'PY'
import sys
print('python', sys.version)

import torch, torchaudio
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchaudio', torchaudio.__version__)

import omegaconf, hydra
import openai
import datasets
import librosa, soundfile
print('imports: ok')

from omegaconf import OmegaConf
cfg = OmegaConf.load('conf/base_v2_breezy.yaml')
assert 'tts' in cfg and 'huggingface' in cfg
print('config load: ok')
PY

python run_topic_txt.py --help >/dev/null
python run_multi_topic_txt_workers.py --help >/dev/null
python run_multi_topic_tts_workers.py --help >/dev/null
python push_dataset_to_hub.py --help >/dev/null

echo "smoke_test: ok"
