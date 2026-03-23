# Synthetic Taiwanese Mandarin Dialogue Pipeline

End-to-end pipeline that generates **synthetic Taiwanese Mandarin (zh-TW) conversational dialogue**, synthesises it with BreezyVoice TTS, and exports a structured HuggingFace dataset. Supports 40+ topics, dual-LLM dialogue generation, and optional per-turn emotion expression.

---

## Pipeline Overview

```
Topic input
    │
    ▼
[Stage 1] Scenario generation          (GPT-4o-mini via OpenRouter)
    │
    ▼
[Stage 2] System prompt generation     (GPT-4o-mini)
    │
    ▼
[Stage 3] Dialogue generation          (Llama-3.3-70B  ×  GPT-4o-mini, dual-LLM)
    │
    ▼
[Stage 4] Overlap insertion            (GPT-4o-mini)
    │
    ▼
[Stage 5] Filler / backchannel insertion  (GPT-4o-mini)
    │
    ▼
[Stage 6] TTS synthesis                (BreezyVoice, multi-GPU)
    │
    ▼
[Stage 7] HuggingFace dataset export
```

---

## Table of Contents

1. [Repository layout](#repository-layout)
2. [Prerequisites — files not in git](#prerequisites--files-not-in-git)
   - [BreezyVoice model checkpoints](#1-breezyvoice-model-checkpoints)
   - [Common Voice zh-TW corpus](#2-common-voice-zh-tw-corpus-no-emotion-mode)
   - [ElevenLabs emotion reference audio](#3-elevenlabs-emotion-reference-audio-emotion-mode)
3. [Reference audio formats](#reference-audio-formats)
4. [Setup](#setup)
5. [Configuration](#configuration)
6. [Running the pipeline](#running-the-pipeline)
7. [Output structure & data schema](#output-structure--data-schema)
8. [Emotion mode vs. no-emotion mode](#emotion-mode-vs-no-emotion-mode)
9. [Available emotions](#available-emotions-26)
10. [ElevenLabs speaker roster](#elevenlabs-speaker-roster)
11. [TTS pipeline notes](#tts-pipeline-notes)
12. [Supported topics](#supported-topics-40)

---

## Repository layout

```
SpeechLab/
├── .env.example                         # API key template
├── .gitignore
├── README.md
│
├── open_source/                         # Pipeline source code
│   ├── conf/
│   │   └── base_v2_breezy.yaml          # Central configuration (all parameters)
│   ├── env/                             # Conda environment files
│   │   ├── README.md                    # Environment setup instructions
│   │   ├── conda-linux-64.py310.lock    # Exact reproducible lock (Linux-64)
│   │   ├── environment.py310.from-history.yml
│   │   └── smoke_test.sh                # Post-install validation
│   ├── slurm/                           # SLURM job scripts (multi-GPU cluster)
│   ├── syn_ver2_breezy.py               # Core pipeline engine (~4 k lines)
│   ├── run_topic_txt.py                 # Single-topic text generation
│   ├── run_topic_tts.py                 # Single-topic TTS synthesis
│   ├── run_multi_topic_txt_workers.py   # Parallel text generation
│   ├── run_multi_topic_tts_workers.py   # Parallel TTS with GPU scheduling
│   ├── push_dataset_to_hub.py           # Dataset export + HuggingFace Hub push
│   ├── generate_txt.sh                  # Wrapper — text stage
│   ├── generate_tts.sh                  # Wrapper — TTS stage
│   ├── push_to_hf.sh                    # Wrapper — HF export
│   └── requirements.txt
│
├── tts_model/
│   └── BreezyVoice/                     # TTS model code (checkpoints not in git)
│       ├── single_inference.py          # Single-turn inference CLI
│       ├── batch_inference.py           # Batch inference via CSV
│       ├── api.py                       # REST API wrapper
│       ├── checkpoints/                 # ← model weights go here (not in git)
│       ├── cosyvoice/                   # CosyVoice base implementation
│       ├── utils/
│       └── third_party/
│
└── ref_audio/                           # Reference audio (not in git, see below)
    ├── cv-corpus-24.0-2025-12-05/
    │   └── zh-TW/
    │       ├── validated.tsv            # Speaker metadata
    │       └── clips/                   # MP3 audio clips
    └── eleven_lab_emotion/
        ├── batch_generate_emo_ref.py    # Script to generate all 810 WAVs
        ├── generate_emo_ref.py          # Quick test / single-voice generation
        ├── transcriptions.json          # WAV path → spoken text (generated)
        ├── male/                        # Generated emotion WAVs
        └── female/                      # Generated emotion WAVs
```

---

## Prerequisites — files not in git

The following must be obtained and placed at the paths shown **before running the pipeline**.

### 1. BreezyVoice model checkpoints

`tts_model/BreezyVoice/` contains model code, but weights are not committed. Download from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download MediaTek-Research/BreezyVoice-300M \
    --local-dir tts_model/BreezyVoice/checkpoints
```

Expected layout after download:

```
tts_model/BreezyVoice/checkpoints/
├── breezyvoice.pt
├── campplus.pt
├── flow.pt
├── hift.pt
└── speech_tokenizer_v1_25hz.onnx
```

> See `tts_model/BreezyVoice/README.md` for the authoritative file list.

---

### 2. Common Voice zh-TW corpus (no-emotion mode)

Used as per-dialogue **speaker cloning references** when `dialogue.with_emotion: false`.

1. Go to https://commonvoice.mozilla.org/zh-TW/datasets
2. Download **Common Voice Corpus 24.0**
3. Extract and place at:

```
ref_audio/
└── cv-corpus-24.0-2025-12-05/
    └── zh-TW/
        ├── validated.tsv
        └── clips/
            ├── common_voice_zh-TW_31336427.mp3
            └── ...
```

**TSV schema (`validated.tsv`)** — columns read by the pipeline:

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | string | Unique speaker hash |
| `path` | string | Filename inside `clips/` |
| `sentence` | string | Transcription text |
| `age` | string | Age group (`teens`, `twenties`, `thirties`, …) |
| `gender` | string | `male_masculine`, `female_feminine`, etc. |
| `accents` | string | Regional accent tag (may be empty) |
| `up_votes` | int | Community up-votes (quality signal) |

Update config paths if you use a different corpus version:

```yaml
# open_source/conf/base_v2_breezy.yaml
tts:
  common_voice_validated_tsv: ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/validated.tsv
  common_voice_clips_dir:     ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/clips
```

---

### 3. ElevenLabs emotion reference audio (emotion mode)

Used as per-turn **speaker cloning references** when `dialogue.with_emotion: true`. Each turn's emotion tag selects the matching reference WAV.

**810 WAV files total**: 26 emotions × 10 speakers × 3 variants + 30 neutral fallbacks (10 speakers × 3 variants).

```
ref_audio/eleven_lab_emotion/
├── transcriptions.json              # WAV path → spoken text + emotion label
├── male/
│   ├── afraid_ranbir_1.wav          # {emotion}_{speaker}_{variant}.wav
│   ├── afraid_ranbir_2.wav
│   ├── afraid_ranbir_3.wav
│   ├── ranbir_1.wav                 # neutral fallback: {speaker}_{variant}.wav
│   └── ...
└── female/
    ├── afraid_bella_1.wav
    └── ...
```

**`transcriptions.json` schema** — each key is a relative WAV path, each value is the ElevenLabs prompt:

```json
{
  "male/afraid_ranbir_1.wav": "[strong Taiwanese accent][afraid][scared][trembling] 我好害怕，感覺有什麼事情要發生了，全身都在發抖，完全不知道該怎麼辦。",
  "male/afraid_ranbir_2.wav": "[strong Taiwanese accent][afraid][panicked][breathing heavily] 那個聲音讓我毛骨悚然，我一步都不敢動，心臟跳得好快。",
  "male/ranbir_1.wav":        "[strong Taiwanese accent] 請問您需要什麼協助呢？我可以幫您查詢相關資訊或說明辦理流程。"
}
```

**How to generate** — requires `ELEVENLABS_API_KEY` in `.env`:

```bash
cd ref_audio/eleven_lab_emotion
python batch_generate_emo_ref.py
```

This calls ElevenLabs `eleven_v3` for each (emotion, speaker, variant) combination, writes all WAVs into `male/` and `female/`, and writes `transcriptions.json`.

```yaml
# open_source/conf/base_v2_breezy.yaml
tts:
  eleven_lab_emotion_dir: ref_audio/eleven_lab_emotion
```

---

## Reference audio formats

| | Common Voice clips | ElevenLabs emotion WAVs | BreezyVoice output |
|--|--|--|--|
| **Format** | MP3 | WAV (PCM) | WAV (PCM) |
| **Sample rate** | 22 050 Hz | 22 050 Hz | 24 000 Hz |
| **Channels** | Mono | Mono | Mono (per-turn) / Stereo (full dialogue) |
| **Bit depth** | 16-bit | 16-bit | 16-bit |
| **Duration** | 1–10 s | 5–15 s | — |

Full dialogue stereo mixes are panned **User = left channel, Agent = right channel**.

---

## Setup

### 1. Create `.env`

```bash
cp .env.example .env
# Fill in your keys
```

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM calls for text generation (OpenRouter) |
| `ELEVENLABS_API_KEY` | Emotion reference audio generation |

`.env` is listed in `.gitignore` and will never be committed.

### 2. Python environment

Requires **Python 3.10** (BreezyVoice ships `cp310` wheels).

**Option A — exact conda lock (Linux-64, recommended)**

```bash
cd ~/SpeechLab/open_source
source ~/miniconda3/bin/activate
conda create -y -n breezyvoice_py310 --file env/conda-linux-64.py310.lock
conda activate breezyvoice_py310
bash env/smoke_test.sh
```

**Option B — from-history + pip (more flexible)**

```bash
conda env create -n breezyvoice_py310 -f open_source/env/environment.py310.from-history.yml
conda activate breezyvoice_py310
pip install -U pip
pip install -r open_source/requirements.txt
pip install -r tts_model/BreezyVoice/requirements.txt
bash open_source/env/smoke_test.sh
```

See `open_source/env/README.md` for a third option (tarball clone, fastest on clusters).

---

## Configuration

All pipeline behaviour is controlled by `open_source/conf/base_v2_breezy.yaml`.

Key settings:

```yaml
dialogue:
  with_emotion: true          # true  → emotion tags + ElevenLabs refs
                              # false → plain text + Common Voice refs
  emotion_rate: 0.6           # fraction of turns that get an emotion tag (0.0–1.0)
  user_model:  meta-llama/llama-3.3-70b-instruct
  agent_model: openai/gpt-4o-mini
  min_turns: 12
  max_turns: 15

tts:
  backend: breezyvoice
  sample_rate: 24000
  breezyvoice_repo_dir: tts_model/BreezyVoice
  # No-emotion mode:
  common_voice_validated_tsv: ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/validated.tsv
  common_voice_clips_dir:     ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/clips
  # Emotion mode:
  eleven_lab_emotion_dir:     ref_audio/eleven_lab_emotion

multi_topic_run:
  output_root_base: /work/<username>
  workers: 2                  # parallel topic workers
  per_topic_count: 130        # dialogues per topic
  topics: [Art, Books, Cars, Coding, Cooking, Travel, ...]   # 40+ topics

huggingface:
  enabled: true
  output_dir: TEST_syn_data/huggingface_dataset
  push_to_hub: false
  hub_repo_id: Jaylin0418/synthetic_dialogue_zh
```

---

## Running the pipeline

### Step 1 — Generate dialogue text (all topics, parallel)

```bash
cd open_source
bash generate_txt.sh \
  --workers 10 \
  --per-topic-count 220 \
  --with-emotion \
  --output-root-base /work/$(whoami)
```

Creates a timestamped output folder (e.g. `syn_multi_topic_20250322_143000/`) under `--output-root-base` with one subdirectory per topic.

### Step 2 — Synthesise speech (multi-GPU TTS)

```bash
bash generate_tts.sh /work/$(whoami)/syn_multi_topic_<timestamp>
```

Control GPU distribution via environment variables:

```bash
GPUS=4 TOPICS_PER_GPU=5 bash generate_tts.sh /work/$(whoami)/syn_multi_topic_<timestamp>
```

### Step 3 — Export to HuggingFace

```bash
bash push_to_hf.sh /work/$(whoami)/syn_multi_topic_<timestamp>
# Append --push to actually upload to HuggingFace Hub (requires HF token)
```

### Single-topic run (for testing)

```bash
cd open_source

# Text generation only
python run_topic_txt.py topic=Travel

# TTS only (assumes text stage already done)
python run_topic_tts.py topic=Travel data_root=/work/$(whoami)/syn_multi_topic_<timestamp>
```

---

## Output structure & data schema

### Directory layout per topic

```
syn_multi_topic_<timestamp>/
└── <mode_name>/
    └── <Topic>_<suffix>/
        ├── txt/
        │   ├── scenario/
        │   │   └── scenarios.json                   # Stage 1 output
        │   ├── scenario_txt/
        │   │   └── <scenario_id>.txt
        │   ├── system_prompt_txt/
        │   │   └── <scenario_id>_system_prompt.txt
        │   ├── dialogue/
        │   │   └── <dialogue_id>.txt                # Raw LLM output
        │   ├── overlap/
        │   │   └── <dialogue_id>.txt                # After overlap insertion
        │   └── filler/
        │       └── <dialogue_id>.txt                # After filler/backchannel insertion
        └── wav/
            └── <dialogue_id>/
                ├── full.wav                         # Stereo mix, no effects
                ├── full_with_overlap_and_pause.wav  # Stereo mix, with effects
                └── individual/
                    ├── turn_metadata.json
                    ├── turn00.wav
                    ├── turn01.wav
                    ├── ...
                    ├── turn00_pause_overlap.wav     # Per-turn with effects
                    └── turn01_pause_overlap.wav
```

### `scenarios.json` schema

```json
{
  "scenarios": {
    "scenario1": { "description": "你正在計畫一趟台灣東部的周末旅遊…" },
    "scenario2": { "description": "…" }
  }
}
```

### Dialogue `.txt` format

Each line is one turn. After the filler stage, three special prefixes may appear:

```
User: 我想規劃一個周末旅遊，你有什麼建議嗎？
Agent: (emotion:happy)當然！台東和花蓮都很棒，你比較喜歡哪種風格？
[overlap] User: 那花蓮
[backchannel] Agent: 嗯
User: [pause]
Agent: 我覺得花蓮太魯閣超級值得去，不過要提早訂票。
```

| Prefix | Meaning |
|--------|---------|
| `User:` / `Agent:` | Normal turn (may include `(emotion:xxx)` prefix) |
| `[overlap] User:` | User interrupts agent — overlaid on agent's audio |
| `[backchannel] Agent:` | Short agent backchanneling during user turn |
| `User: [pause]` | User hesitation — rendered as silence in audio |

### `turn_metadata.json` schema

Array of objects, one per turn:

```json
[
  {
    "turn_idx": 0,
    "role": "user",
    "speaker": "user",
    "text": "我想規劃一個周末旅遊，你有什麼建議嗎？",
    "emotion": "happy",
    "audio_path": ".../individual/turn00.wav",
    "audio_start": 0.0,
    "audio_end": 2.53,
    "audio_duration": 2.53,
    "audio_path_pause_overlap": ".../individual/turn00_pause_overlap.wav",
    "audio_start_pause_overlap": 0.0,
    "audio_end_pause_overlap": 3.10,
    "audio_duration_pause_overlap": 3.10,
    "para_tags": { "gender": null, "pitch": null, "speed": null, "volume": null, "emotion": null },
    "emotion_reference": ["female/happy_bella_1.wav"],
    "speaker_reference": "/abs/path/to/common_voice_zh-TW_31336427.mp3",
    "speaker_reference_id": "common_voice_zh-TW_31336427.mp3",
    "speaker_reference_metadata": {
      "gender": "male_masculine",
      "age": "thirties",
      "accents": null,
      "sentence": "我今天和明天都要上課"
    }
  }
]
```

### HuggingFace dataset column schema

Each row is one dialogue turn.

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | `{topic}_{mode}_{scenario_id}` |
| `mode` | string | `"normal"` or `"emotion"` |
| `topic` | string | e.g. `"Travel"` |
| `scenario` | string | Scenario description used to prompt the LLM |
| `system_prompt` | string | System prompt for the agent role |
| `turn_index` | int | 0-based index within the dialogue |
| `LLM1` | string | Model used for user turns |
| `LLM2` | string | Model used for agent turns |
| `speaker` | string | `"user"` or `"agent"` |
| `text` | string | Spoken text (emotion prefix stripped) |
| `paralinguistic_info` | dict | `{gender, pitch, speed, volume, emotion}` |
| `emotion` | string | Emotion label, or `null` |
| `audio_path` | string | Relative path to per-turn WAV |
| `audio` | string | Absolute path to per-turn WAV |
| `audio_start` | float | Start offset (seconds) within `full.wav` |
| `audio_end` | float | End offset (seconds) within `full.wav` |
| `audio_duration` | float | Duration (seconds) |
| `full_dialogue_audio` | string | Absolute path to `full.wav` |
| `reference` | string | Absolute path to speaker reference audio |
| `reference_id` | string | Basename of reference file |
| `reference_metadata` | string (JSON) | Speaker metadata from TSV or ElevenLabs |
| `reference_age` | string | Speaker age group |
| `reference_gender` | string | `"male"` or `"female"` |
| `reference_accent` | string | Accent tag, or `null` |
| `emotion_reference` | list[str] | Relative paths to ElevenLabs refs used |
| `audio_path_pause_overlap` | string | (optional) Per-turn WAV with effects |
| `audio_pause_overlap` | string | (optional) Absolute path, effects WAV |
| `audio_start_pause_overlap` | float | (optional) |
| `audio_end_pause_overlap` | float | (optional) |
| `audio_duration_pause_overlap` | float | (optional) |

---

## Emotion mode vs. no-emotion mode

| | `with_emotion: false` | `with_emotion: true` |
|--|----------------------|---------------------|
| Dialogue text | Plain Mandarin turns | Turns prefixed `(emotion:xxx)` |
| TTS speaker reference | Random Common Voice clip (per dialogue) | ElevenLabs WAV matched to emotion + speaker (per turn) |
| Reference prerequisite | Download Common Voice corpus | Run `batch_generate_emo_ref.py` |
| Reference metadata fields | `sentence`, `age`, `gender`, `accents` | `voice_id`, `display_name`, `sentence`, `gender` |

Both modes produce the same `turn_metadata.json` schema and the same HuggingFace dataset columns.

---

## Available emotions (26)

| Category | Emotions |
|----------|---------|
| Positive (11) | `amusement` · `calm` · `compassion` · `contentment` · `excitement` · `gratitude` · `happy` · `hope` · `pride` · `relief` · `surprised` |
| Negative (13) | `afraid` · `angry` · `anxiety` · `cry` · `disappointment` · `disgusted` · `envy` · `frustration` · `grief` · `guilt` · `melancholic` · `sad` · `shame` |
| Special (2) | `sarcastic` · `hysteria` |

Each emotion has **3 text variants per speaker** to avoid repetitive reference audio.

---

## ElevenLabs speaker roster

| Key | Display name | Gender | Notes |
|-----|--------------|--------|-------|
| `ranbir` | Ranbir — Calm, Steady and Clear | Male | |
| `roger` | Roger — Laid-Back, Casual, Resonant | Male | |
| `charlie` | Charlie — Deep, Confident, Energetic | Male | |
| `george` | George — Warm, Captivating Storyteller | Male | |
| `callum` | Callum — Husky Trickster | Male | |
| `river` | River — Relaxed, Neutral, Informative | Female | |
| `harry` | Harry — Fierce Warrior | Male | |
| `bella` | Bella — Professional, Bright, Warm | Female | Excluded from auto-selection — strong Mainland Chinese accent |
| `sarah` | Sarah — Mature, Reassuring, Confident | Female | |
| `laura` | Laura — Enthusiast, Quirky Attitude | Female | |

To exclude additional speakers from random selection, add their key to `_EXCLUDED_SPEAKERS` in `open_source/syn_ver2_breezy.py`.

---

## TTS pipeline notes

### Traditional Chinese preservation

WeTextProcessing's `ZhNormalizer` (used by BreezyVoice's `CosyVoiceFrontEnd`) operates on Simplified Chinese, causing Traditional Chinese input to be converted before synthesis. A post-normalisation OpenCC `s2twp` conversion step in `tts_model/BreezyVoice/single_inference.py` restores Traditional Chinese (Taiwan standard) after text normalisation.

### Reference audio length limit

The BreezyVoice speech tokenizer has a maximum input length of ~1 500 frames (~15 s). Reference audio clips longer than 15 s trigger an ONNX broadcast error at inference time. All speaker prompt audio is automatically truncated to 15 s (240 000 samples @ 16 kHz) before being passed to the model.

### Emotion tag stripping

LLM-generated dialogue may contain emotion tags in two positions:

- **Leading** (canonical): `(emotion:frustration) 我真的很不開心`
- **Trailing** (variant): `我真的很不開心 (frustration)`

Both forms are stripped from the synthesis input. Leading tags are stripped by `strip_leading_emotion_tag`; trailing tags by a regex in `sanitize_tts_text` in `open_source/syn_ver2_breezy.py`.

### SLURM multi-GPU configuration

`slurm/tts_multi_gpu_all_topics.job` runs **2 SLURM tasks per GPU** (32 tasks across 4 nodes × 4 GPUs). Each task's `CUDA_VISIBLE_DEVICES` is set via `$((SLURM_LOCALID / 2))` inside a `bash -c` wrapper, since `--gpus-per-task` / `--gpu-bind` do not propagate correctly in all cluster configurations. `--kill-on-bad-exit=0` prevents early-finishing tasks from cancelling remaining work.

---

## Supported topics (40+)

Art · Books · Cars · Celebrities · Coding · Cooking · Education · Events · Fashion · Fitness · Finance · Food · Gaming · Gardening · Health · History · Hobbies · Holidays · Home · Languages · Makeup · Movies · Music · Nature · News · Pets · Philosophy · Photography · Podcasts · Politics · Relationships · Science · Shopping · Social Media · Spirituality · Sports · Technology · Traditions · Travel · Weather · Work
