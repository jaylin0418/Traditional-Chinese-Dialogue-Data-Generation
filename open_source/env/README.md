# synthetic_data_breezyvoice environment (Python 3.10 / BreezyVoice)

This project’s **full pipeline (TXT → TTS → HuggingFace export/push)** requires **Python 3.10** because BreezyVoice depends on `cp310` wheels (e.g. `ttsfrd`).

This folder contains exports from the working conda env on the source machine:

- `conda-linux-64.py310.lock`: **explicit** conda lock (best reproducibility on Linux-64)
- `pip-freeze.py310.lock.txt`: pip freeze snapshot (debug/reference)
- `environment.py310.full.yml`: full conda export (verbose)
- `environment.py310.from-history.yml`: from-history export (smaller, less strict)

## Option A (recommended): recreate from explicit lock (Linux-64)

On the target machine:

```bash
cd ~/SpeechLab/open_source

source ~/miniconda3/bin/activate
conda create -y -n breezyvoice_py310 --file env/conda-linux-64.py310.lock
conda activate breezyvoice_py310

# quick sanity
bash env/smoke_test.sh
```

Notes:
- This requires internet access to fetch the **exact** package builds referenced in the lock.
- Works best when the target is also Linux x86_64.

## Option B: recreate from history + pip requirements (more flexible)

```bash
cd ~/SpeechLab/open_source

source ~/miniconda3/bin/activate
conda env create -n breezyvoice_py310 -f env/environment.py310.from-history.yml
conda activate breezyvoice_py310

# Install project requirements (will fetch pip wheels)
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r ~/SpeechLab/tts_model/BreezyVoice/requirements.txt

bash env/smoke_test.sh
```

## Option C: pack the env as a tarball (fastest “clone”, no solver)

On the source machine:

```bash
cd /home/jaylin0418/SpeechLab/synthetic_data_breezyvoice
bash env/pack_conda_env.sh
```

Copy `env/breezyvoice_py310.tar.gz` to the target machine, then:

```bash
mkdir -p ~/conda_envs/breezyvoice_py310

tar -xzf breezyvoice_py310.tar.gz -C ~/conda_envs/breezyvoice_py310
source ~/conda_envs/breezyvoice_py310/bin/activate
conda-unpack

bash ~/SpeechLab/synthetic_data_breezyvoice/env/smoke_test.sh
```

## Running the pipeline

See the top-level `README.md` for end-to-end commands. Wrapper scripts will prefer `~/miniconda3/envs/breezyvoice_py310/bin/python` automatically when `SYN_PYTHON=auto`.
