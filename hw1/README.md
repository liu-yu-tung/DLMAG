# HW1: release decade (A) and 1980s release market (B)

Pretrained audio models with small classifiers on top, plus MERT-v2 with its top 12 blocks fine-tuned, combined by averaging temperature-calibrated log-probabilities.

| Task | Recipe | 5-fold S on train | Validation S (top-1 / top-3) |
|---|---|---|---|
| A (decade, 6 classes) | MERT-v2 layer probes with ordinal soft labels + zero-shot Qwen2-Audio-7B-Instruct + fine-tuned MERT-v2 top 12 | 0.969 | 1.000 (0.545 / 0.909) |
| A fallback (`--no-alm`) | MERT-v2 layer probes with ordinal soft labels + fine-tuned MERT-v2 top 12 | 0.943 | 0.996 (0.538 / 0.917) |
| B (market, 6 classes) | MERT-v2 layer probes + Whisper language ID + fine-tuned MERT-v2 top 12 | 1.051 | 1.064 (0.608 / 0.912) |

S = top-1 + 0.5 × top-3. Everything is fit on the train split only; validation was used for checking, not for choosing. The recipes were chosen by 5-fold cross-validation on train, plus artist-proxy grouped folds.

- **Fine-tuned MERT-v2 top 12:** blocks 13-24 of MERT-v2 and a linear head, trained on the cached output of the frozen block 12. Three seeds; their log-probs are averaged.
- **Calibration:** each part gets one temperature, fitted on its 5-fold out-of-fold log-probs on train. The parts then have equal weight.

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

The default mode peaks at about 13 GB of GPU memory (Qwen2-Audio in 8-bit). With less GPU memory, run the fallback for task A, which peaks at about 6 GB:

```
python scripts/predict.py --data /path/to/data --out <studentID>.json --no-alm
```

The output is `{"dataset_A": {sample_id: [top1, top2, top3]}, "dataset_B": {...}}`, checked against the manifests before writing.

Measured on the machine above, from audio, models already downloaded (132 + 102 test clips):

| Mode | Wall time | Models loaded |
|---|---|---|
| default | 7 min 3 s (Qwen2-Audio about 2.1 s per A clip) | MERT-v2, 3 fine-tuned top stacks per task, Qwen2-Audio (8-bit), Whisper |
| `--no-alm` | 1 min 46 s | MERT-v2, 3 fine-tuned top stacks per task, Whisper |

Both modes were checked in a fresh environment built only from `requirements.txt` (2026-10-01). The top-3 lists equal those made from the cached training features on all 234 test clips, order included.

Other options: `--datasets A` or `B` for one task, `--batch` (MERT batch size, default 4), `--ckpt-dir` (default `checkpoints/`).

## Files

- **`checkpoints/A.joblib`, `checkpoints/B.joblib`:** fitted classifiers, calibration temperatures, and the recipe (about 1 MB each).
- **`checkpoints/ft_{A,B}_L12_s{0,1,2}.pt`:** the fine-tuned MERT-v2 blocks 13-24 and head, one file per task and seed (6 files, 1.1 GB each, fp32).
- **`scripts/predict.py`:** audio → features → checkpoints → JSON.
- **`scripts/finetune_mert.py`:** fine-tuning; its `Top` module is also used by `predict.py`.
- **`src/hw1/`:** data loading, feature extractors (`features/mert.py`, `features/langid.py`), the final models (`final.py`), metrics.
- **`scripts/alm_qwen.py`:** Qwen2-Audio loading and label scoring, shared with `predict.py`.
- **`scripts/make_submission.sh`:** builds the TA folder `dist/hw1_submission/`: inference code, the final training pipeline, checkpoints, `submission/README.md`, and the pins of `requirements.txt` without `-e .`.

## Rebuilding the checkpoints

These steps need the full training environment (`uv sync` with `pyproject.toml`), not only `requirements.txt`.

1. Extract features for all splits:
   - `scripts/extract_mert.py --dataset A` (and `B`);
   - `scripts/extract_langid.py --dataset B`;
   - `scripts/alm_qwen.py --dataset A --prompts plain --splits train test`, then `--splits validation`.
2. Fine-tune, per task (`A` shown):
   - `scripts/finetune_mert.py --dataset A --cache 12` caches the block-12 output of every clip (2.0 GB for A, 1.5 GB for B);
   - `scripts/finetune_mert.py --dataset A --from-layer 12 --seed 0` (and seeds 1, 2) trains the 5 folds and the full fit. It writes `checkpoints/ft_A_L12_s0.pt` and the log-probs in `features/ft_A_L12_s0.npz`.
3. `scripts/train_final.py` fits on train, writes `checkpoints/{A,B}.joblib`, and prints validation scores and the test predictions from the cached features.

The experiments behind the recipe choice (5-fold screens, ablations, the other encoders) are in `scripts/` and logged in `docs/worklog/`.
