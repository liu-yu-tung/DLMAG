# HW3 evaluation notes

Evaluation uses av-benchmark with the LAION-CLAP checkpoint the TA requires, `music_audioset_epoch_15_esc_90.14.pt`.

## Setup (works on this machine)

- **av-benchmark:** `external/av-benchmark` (gitignored), upstream commit `f351b9a6fc6abde746d5f8e1d4c47c883319cb41` plus the local patch `docs/av_benchmark.patch` (`git diff` of the clone).
- **Dependency commits (pulled by its `pyproject.toml`):** laion-clap `590b52df03a5e480634b8714135d3534313be860` (hkchengrex/CLAP), passt `1ab778984f67cca09dd8bbad11f728a0122cce24`, imagebind `9989650c87d393d7e8c144194182cbf124cd03a0`, msclap `3ad265d353aea4d6d4baa4bec2545184f0a841e1`.
- **Install commands:**

```sh
cd external/av-benchmark
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .
```

  `torchvision` must come from the cu128 index first, or `-e .` pulls a mismatched PyPI build. Do not install `torchcodec` (see patch notes).
- **Weights:**
  - `external/av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt` (2,352,471,003 bytes, from `lukewys/laion_clap`).
  - PANN `~/.cache/audioldm_eval/ckpt/Cnn14_16k_mAP=0.438.pth` (zenodo; very slow single connection, 12 parallel `curl -r` ranges worked).
  - VGGish and PaSST are fetched by `torch.hub` / `hear21passt` into `~/.cache/torch/hub`.

## How the CLAP checkpoint is chosen

- `av_bench/extraction_models.py` and `extract_text.py` hardcode `weights/music_speech_audioset_epoch_15_esc_89.98.pt` and build `laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')`.
- There was no CLI flag. Patch: `--clap_ckpt <path>` in `av_bench/args.py`, `evaluate.py`, `extract_text.py`; model type unchanged.
- `HTSAT-base`, `enable_fusion=False` is also right for the 90.14 checkpoint: loading it leaves no model key missing, and the text pooler weights come from the checkpoint (the `roberta-base` "MISSING pooler" warning at construction is overwritten by `load_ckpt`).

## Other patches (all in `docs/av_benchmark.patch`)

- `--skip_ms_clap`; with `--skip_video_related` the ImageBind, Synchformer and MS-CLAP models are no longer built (upstream builds all of them and needs `synchformer_state_dict.pth` and a 4.5 GB ImageBind download even when skipped). `evaluate()` now also receives the skip flags.
- `AudioDataset` reads with `soundfile` instead of `torchaudio.load`: torchaudio 2.11 needs `torchcodec`, which cannot load against this machine's FFmpeg 9.
- `extract()` also globs `*.mp3` (used for the all-MusicCaps reference; MP3 decoding via soundfile).
- `Cnn14` downloads only the checkpoint for its sample rate (upstream also fetches an unused 1.3 GB 32 kHz one).
- `torch.hub.load(..., trust_repo=True)` for VGGish (non-interactive runs fail on the trust prompt).

## What av-benchmark does (read from the source and the smoke test)

- **Pairing:** GT and predicted features are dictionaries keyed by the file stem; `unroll_paired_dict` pairs by stem (names of length 18, 20, 21 or 30 are cut by `clean_sample_name`; `0`..`99` are untouched). KL (PANNs, PaSST) and CLAP are per pair; FD, ISC are set-level.
- **CLAP score:** mean cosine between the LAION text embedding cached in `<gt_cache>/clap_laion_text.pth` (made by `extract_text.py` from a CSV with columns `name,caption`) and the prediction's `clap_laion_audio.pth`. The text is therefore the caption, not a GT audio feature.
- **Audio:** `.wav`, `.flac` (and `.mp3` after the patch), mono mean, DC removed, resampled to 16 kHz (PANN, VGGish), 32 kHz (PaSST) and 48 kHz (CLAP), then cut to `--audio_length` seconds. It never pads (PaSST and VGGish pad internally); prediction batches need equal lengths, ground truth uses batch size 1. We use 10 s.
- **Metrics printed:** `FD-VGG`, `FD-PANN`, `FD-PASST`, `KL-PANNS-softmax`, `KL-PASST-softmax`, `ISC-PANNS-mean/std`, `ISC-PASST-mean/std`, `LAION-CLAP-Score` (also `MS-CLAP-Score`, `IB-Score`, `DeSync` when not skipped). KL is KL(ground truth || prediction) on the softmax of the classifier logits, summed over classes, mean over pairs.
- **Caches:** `<cache>/pann_features.pth`, `vggish_features.pth`, `passt_features_embed.pth`, `passt_logits.pth`, `clap_laion_audio.pth` (prediction side), `clap_laion_text.pth` (ground-truth side), `output_metrics.json`. The extraction is skipped when `passt_features_embed.pth` exists, so `--recompute_*_cache` is needed after the audio changes.

### Smoke test (5 retrieved-style MusicCaps clips vs 5 others, `outputs/smoke/evaluate.log`)

`FD-VGG 10.886`, `FD-PANN 140.437`, `FD-PASST 1828.809`, `KL-PANNS-softmax 3.052`, `KL-PASST-softmax 2.054`, `LAION-CLAP-Score 0.0100` (captions 0-4 against unrelated clips), `ISC-*` = nan (5 clips cannot fill 10 splits). Runtime about 50 s on the GPU. Only the metric plumbing is tested; with n = 5 the FD numbers mean nothing.

## Retrieval (`scripts/retrieval/retrieval.py`)

- LAION-CLAP 90.14, `HTSAT-base`, no fusion. Audio: 48 kHz mono, DC removed (same as av-benchmark), cut or zero-padded to 10 s. Cosine of L2-normalised audio and text embeddings; top-1 per caption.
- Run with the av-benchmark venv: `external/av-benchmark/.venv/bin/python scripts/retrieval/retrieval.py` (GPU about 2 GB, 2 min 41 s in total; cached embeddings in `outputs/cache/musiccaps_clap90.npy` + `_ids.txt`, `test_text_clap90.npy` + `_ids.txt`).
- Outputs: `results/retrieval.json`, `results/retrieval_top5.csv`, `results/retrieval_ceiling.csv`.
- Top-1 cosine min 0.3477, mean 0.4521, max 0.5645. Only 59 distinct clips for the 100 captions (hubness: some clips are the top-1 for up to several captions); the retrieved set therefore has duplicates, and FD against it uses the duplicates as they are.
- **HF cross-check** (`scripts/retrieval/retrieval_crosscheck_hf.py`, hw1 venv, `laion/larger_clap_music`, transformers 5.17): top-1 agreement 0.00, correlation of all 536,700 scores 0.08, HF top-1 scores only 0.044 to 0.046, and the HF audio embeddings of different clips have mean pairwise cosine 0.95 (0.27 for the .pt model). So in that environment the HF path is near-degenerate and is not a usable proxy for the .pt model; whether the weights are the same could not be decided from this (see `results/retrieval_crosscheck_hf.json`).

## Scripts

- `scripts/evaluation/evaluate.py --pred DIR [--gt DIR] [--name NAME]` (av-benchmark venv): `DIR` holds `0.wav`..`99.wav`, each at least 10 s. Builds `outputs/ref/test_retrieved/` (`<id>.wav`, 16-bit, native sample rate) from `results/retrieval.json`, the text cache with `extract_text.py`, runs av-benchmark's `evaluate.py` (CLI), then adds:
  - per-prompt values in `results/<name>/per_prompt.csv` (`clap_cosine`, `kl_passt_softmax`, `kl_panns_softmax`; their means are asserted equal to av-benchmark's numbers);
  - 95% percentile bootstrap intervals (1000 resamples over prompt ids, seed 0; `eval_common.bootstrap_indices`) for the three means and for every FD (resample the prediction features with replacement, reference fixed);
  - FD against a second reference, all 5,367 MusicCaps clips (`*_musiccaps`; cached in `outputs/ref/musiccaps_cache`, about 10 min the first time, GPU under 3 GB);
  - `code_version` (short HEAD of HW3, `+dirty` if `git status --porcelain` is not empty).
  - FD in the CI loop uses an exact eigenvalue form of the same formula (`FrechetReference`), because `sqrtm` at 2048 dimensions takes seconds per resample; the full-sample value is checked against av-benchmark's (`value_fast_formula`). Headline `value` of `FD-*` is av-benchmark's own number.
  - **Caveat:** FD is biased upward at small n and a bootstrap resample has duplicates, so the percentile interval can sit above the point value. `bootstrap_mean` is stored next to it to show the shift. Read the interval as spread, and compare runs on paired per-prompt metrics where possible.
  - ISC values have no interval.
- `scripts/evaluation/fd_mert.py --pred DIR [--name NAME]` (hw1 venv `/home/liuyu/Developments/DLMAG/hw1/.venv/bin/python`): FD-MERT with `hw1.features.mert.load_mert` and `pooled_layers`. **Embedding:** model `m-a-p/MERT-v2-30s` (pinned revision from hw1), first 10 s of mono audio at 24 kHz (resampled with torchaudio), the per-layer masked time-mean of the hidden states (first half of `pooled_layers`' output), averaged over all 25 hidden states (embedding output plus 24 layers): a 1024-vector per clip. Writes `fd_mert_retrieved` (reference: the retrieved clips' embeddings taken from the MusicCaps cache, so duplicates stay) and `fd_mert_musiccaps` into `results/<name>/metrics.json`, same bootstrap. Cache: `outputs/cache/musiccaps_mert.npy` + `_ids.txt`. Run `evaluate.py` first (it creates `metrics.json`).
- `scripts/leaderboard.py`: `results/leaderboard.md` and `.csv`, one row per run, value with interval per metric; header line with the real-clip CLAP ceiling (mean of `retrieval_ceiling.csv`, with its bootstrap interval); if a run named `baseline` exists, paired per-prompt differences for CLAP and KL-PaSST with a bootstrap interval.
- Shared helpers are in `text_to_music/eval_common.py`; the scripts put `src/` on `sys.path` themselves.

## Usage

```sh
external/av-benchmark/.venv/bin/python scripts/evaluation/evaluate.py --pred outputs/gen/baseline --name baseline
/home/liuyu/Developments/DLMAG/hw1/.venv/bin/python scripts/evaluation/fd_mert.py --pred outputs/gen/baseline --name baseline
external/av-benchmark/.venv/bin/python scripts/leaderboard.py
```

`outputs/gen/baseline` is only an example path.
