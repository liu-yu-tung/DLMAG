# HW1: release decade (A) and 1980s release market (B)

Pretrained audio models with small classifiers on top, plus MERT-v2 with its top 12 blocks fine-tuned, combined by averaging temperature-calibrated log-probabilities.

| Task | Recipe | Validation S (top-1 / top-3) |
|---|---|---|
| A (decade, 6 classes) | MERT-v2 layer probes with ordinal soft labels + zero-shot Qwen2-Audio-7B-Instruct + fine-tuned MERT-v2 top 12 | 1.000 (0.545 / 0.909) |
| A fallback (`--no-alm`) | MERT-v2 layer probes with ordinal soft labels + fine-tuned MERT-v2 top 12 | 0.996 (0.538 / 0.917) |
| B (market, 6 classes) | MERT-v2 layer probes + Whisper language ID + fine-tuned MERT-v2 top 12 | 1.064 (0.608 / 0.912) |

S = top-1 + 0.5 × top-3. Everything is fit on the train split only; validation was used for checking, not for choosing.

## Setup

Python 3.12, Linux, NVIDIA GPU with CUDA (tested on an RTX 4060 Ti 16 GB, driver 615, torch 2.14.0+cu130).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nothing else is installed: `scripts/predict.py` imports the `hw1` package from `src/`.

The first run downloads three models from the Hugging Face Hub into `~/.cache/huggingface`:

- **`m-a-p/MERT-v2-30s`:** about 2.5 GB, both tasks. Loaded with `trust_remote_code`, at a pinned revision.
- **`openai/whisper-large-v3-turbo`:** about 1.6 GB, task B.
- **`Qwen/Qwen2-Audio-7B-Instruct`:** about 16 GB, task A only, skipped with `--no-alm`.

## Checkpoints

If `checkpoints/` is empty (the GitHub copy: the files are larger than GitHub's 100 MiB limit), download the 8 files (6.5 GB) from the release `hw1` and check them:

```
mkdir -p checkpoints
for f in A.joblib B.joblib ft_A_L12_s0.pt ft_A_L12_s1.pt ft_A_L12_s2.pt ft_B_L12_s0.pt ft_B_L12_s1.pt ft_B_L12_s2.pt; do
  curl -L --fail -o checkpoints/$f https://github.com/liu-yu-tung/DLMAG/releases/download/hw1/$f
done
sha256sum -c checkpoints.sha256
```

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

| Mode | Wall time |
|---|---|
| default | 7 min 3 s (Qwen2-Audio about 2.1 s per A clip) |
| `--no-alm` | 1 min 46 s |

Other options: `--datasets A` or `B` for one task, `--batch` (MERT batch size, default 4), `--ckpt-dir` (default `checkpoints/`), `--device`.

## Files

- **`scripts/predict.py`:** audio → features → checkpoints → JSON.
- **`checkpoints/A.joblib`, `checkpoints/B.joblib`:** fitted classifiers, calibration temperatures, and the recipe (about 1 MB each).
- **`checkpoints/ft_{A,B}_L12_s{0,1,2}.pt`:** the fine-tuned MERT-v2 blocks 13-24 and head, one file per task and seed (6 files, 1.1 GB each, fp32).
- **`src/hw1/`:** data loading, MERT-v2 and Whisper feature extractors, the final models (`final.py`), metrics.
- **`scripts/alm_qwen.py`:** Qwen2-Audio loading and label scoring, used by `predict.py`.
- **`scripts/finetune_mert.py`:** fine-tuning; its `Top` module is used by `predict.py`.
- **`scripts/extract_mert.py`, `scripts/extract_langid.py`, `scripts/train_final.py`:** the rest of the training pipeline below.
- **`checkpoints.sha256`:** SHA-256 of the 8 checkpoint files.
- **`results/final_validation.json`:** the validation scores in the table above.

## Rebuilding the checkpoints

Not needed for inference. Put the data in `data/` inside this folder, then run from this folder with `PYTHONPATH=src`. Features go to `features/`, checkpoints to `checkpoints/`.

1. Extract features for all splits:
   - `scripts/extract_mert.py --dataset A` (and `B`);
   - `scripts/extract_langid.py --dataset B`;
   - `scripts/alm_qwen.py --dataset A --prompts plain --splits train test`, then `--splits validation`.
2. Fine-tune, per task (`A` shown):
   - `scripts/finetune_mert.py --dataset A --cache 12` caches the block-12 output of every clip (2.0 GB for A, 1.5 GB for B);
   - `scripts/finetune_mert.py --dataset A --from-layer 12 --seed 0` (and seeds 1, 2) trains the 5 folds and the full fit, about 11 min per fit on A and 9 min on B. It writes `checkpoints/ft_A_L12_s0.pt` and the log-probs in `features/ft_A_L12_s0.npz`.
3. `scripts/train_final.py` fits on train, writes `checkpoints/{A,B}.joblib`, and prints validation scores and the test predictions from the cached features.
