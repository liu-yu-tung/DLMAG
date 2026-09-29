# HW1: release decade (A) and 1980s release market (B)

Frozen pretrained audio encoders with small classifiers on top, combined by averaging log-probabilities.

| Task | Recipe | Validation S (top-1 / top-3) |
|---|---|---|
| A (decade, 6 classes) | MERT-v2 layer probes with ordinal soft labels + zero-shot Qwen2-Audio-7B-Instruct, temperature-calibrated | 0.977 (0.523 / 0.909) |
| A fallback (`--no-alm`) | MERT-v2 layer probes with ordinal soft labels | 1.011 (0.553 / 0.917) |
| B (market, 6 classes) | MERT-v2 layer probes + Whisper language ID, equal weight | 1.069 (0.627 / 0.882) |

S = top-1 + 0.5 × top-3. The models are fit on the train split only. The recipes were chosen by 5-fold cross-validation on train (A 0.947, B 1.023), not by validation.

## Setup

Python 3.12, Linux, NVIDIA GPU with CUDA (tested on an RTX 4060 Ti 16 GB, driver 615, torch 2.14.0+cu130).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with uv: `uv venv --python 3.12 && uv pip install -r requirements.txt`.

`requirements.txt` holds only what inference needs and installs this folder as the `hw1` package.

The first run downloads three models from the Hugging Face Hub into `~/.cache/huggingface`:

- **`m-a-p/MERT-v2-30s`:** about 2.5 GB, both tasks. Loaded with `trust_remote_code`, at a pinned revision.
- **`openai/whisper-large-v3-turbo`:** about 1.6 GB, task B.
- **`Qwen/Qwen2-Audio-7B-Instruct`:** about 16 GB, task A only, skipped with `--no-alm`.

## Data layout

`--data` points to a folder with one subfolder per task, as distributed:

```
data/
  dataset_A/manifest.csv   columns: sample_id, split, label, audio_path, ...
  dataset_A/audio/*.wav    30 s, 24 kHz mono
  dataset_B/manifest.csv
  dataset_B/audio/*.wav
```

Only rows with `split == test` are predicted (change with `--split`).

## Inference

```
python scripts/predict.py --data /path/to/data --out <studentID>.json
```

Without enough GPU memory for Qwen2-Audio (it needs about 10 GB), run the fallback for task A:

```
python scripts/predict.py --data /path/to/data --out <studentID>.json --no-alm
```

The output is `{"dataset_A": {sample_id: [top1, top2, top3]}, "dataset_B": {...}}`, checked against the manifests before writing.

Measured on the machine above, from audio, models already downloaded (132 + 102 test clips):

| Mode | Wall time | Models loaded |
|---|---|---|
| default | 5 min 52 s (Qwen2-Audio about 2.1 s per A clip) | MERT-v2, Qwen2-Audio (8-bit), Whisper |
| `--no-alm` | 50 s | MERT-v2, Whisper |

Both modes were checked in a fresh environment built only from `requirements.txt`: the predictions match those made from the cached training features on all 234 test clips.

Other options: `--datasets A` or `B` for one task, `--batch` (MERT batch size, default 4), `--ckpt-dir` (default `checkpoints/`).

## Files

- **`checkpoints/A.joblib`, `checkpoints/B.joblib`:** fitted classifiers, calibration temperatures, and the recipe (about 1 MB each).
- **`scripts/predict.py`:** audio → features → checkpoints → JSON.
- **`src/hw1/`:** data loading, feature extractors (`features/mert.py`, `features/langid.py`), the final models (`final.py`), metrics.
- **`scripts/alm_qwen.py`:** Qwen2-Audio loading and label scoring, shared with `predict.py`.

## Rebuilding the checkpoints

These steps need the full training environment (`uv sync` with `pyproject.toml`), not only `requirements.txt`.

1. Extract features for all splits:
   - `scripts/extract_mert.py --dataset A` (and `B`);
   - `scripts/extract_langid.py --dataset B`;
   - `scripts/alm_qwen.py --dataset A --prompts plain --splits train test`, then `--splits validation`.
2. `scripts/train_final.py` fits on train, writes `checkpoints/`, and prints validation scores and the test predictions from the cached features.

The experiments behind the recipe choice (5-fold screens, ablations, the other encoders) are in `scripts/` and logged in `docs/worklog/`.
