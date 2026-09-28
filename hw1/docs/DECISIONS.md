# Decision log

Append-only. Newest at the bottom. Format: date, decision, reason, source.

## 2026-09-28

- **One repo for the course (`DLMAG`), one uv project per homework.** The user asked for it; each homework keeps its own deps.
- **Data stays out of git** (`data/`, audio, features, checkpoints). The course rules forbid uploading the dataset, and the files are too large anyway.
- **Course slides and PDFs stay out of git.** They are course material, not ours to redistribute.
- **Python 3.12.** It's the version that passed the CUDA check on this PC (`~/scratch/gpu-test`).
- **torch from PyPI, no extra index.** The PyPI wheel is `2.14.0+cu130`, and `torch.cuda.is_available()` is True on the RTX 4060 Ti.
- **Selection on validation only; no cross-validation on train.** The manifest has no artist field, so train folds would share artists and give optimistic scores. The course rule is train for fitting, validation for selection.
- **Final models are trained on train only.** The course rules forbid training or tuning on validation or test.
- **Priority order: classification first (MERT-v2 probe), then the prediction JSON, then the ALM, then the extras.** This follows points per hour: the accuracy points depend on the JSON, and the ALM is required only for the report. See `PLAN.md`.
- **Main encoder: `m-a-p/MERT-v2-30s`.** It was trained on 30 s at 24 kHz, which matches the data exactly (no resampling or chunking). The teacher hinted at MERT.
- **CLAP (`laion/larger_clap_music`) as the second encoder and zero-shot baseline.** The teacher hinted at it, and it adds text-aligned style features.
- **The ALM (Qwen2-Audio-7B-Instruct) loads in 8-bit.** In bf16 it needs about 16.8 GB, more than the 16 GB card.
- **ALM answers come from ranking the label strings by likelihood, not from parsing free text.** That always yields a valid ranked top-3.
- **Parallel workers (subagents) report through files in `docs/worklog/` and `results/`.** The user asked for file pointers instead of pasted text.
- **At most 2 parallel workers, and only for heavy, independent items** (MERT extraction, CLAP extraction). Every worker starts without context and adds token overhead (user request). Small pieces (review fixes, glue code, docs, commits) stay in the main session.
- **Review fixes to `data.py` / `metrics.py`:**
  - Unknown labels in train or validation raise an error instead of silently becoming -1.
  - The sample-rate check raises an error instead of using `assert`, which Python skips under `-O`.
  - `LABELS` is imported at the top of `metrics.py`.
- **Model downloads use `HF_HUB_DISABLE_XET=1` (plain HTTPS).** The xet backend stalled for over 4 minutes with no progress on MERT-v2 and CLAP. The connection here is about 1 MB/s in total (a pacman mirror is just as slow), so the downloads take hours. The order is MERT-v2, then CLAP, Whisper, MERT-v1, Qwen2-Audio. Workers pause instead of polling while they wait.
- **When the weights land, the main session runs the MERT and CLAP scripts itself instead of resuming the workers.** The scripts exist, so what's left is running them and reviewing the results. Resuming a worker reloads its full context, which costs more (see `worklog/usage.md`).
- **No outside lookups about the dataset or its recordings (user rule, anti-cheating).** Don't search for the Discogs-VI paper, song IDs, artists or labels, and don't use audio fingerprinting or search to identify tracks. Work from hypotheses, test them on train/validation only, and write them up in `docs/HYPOTHESES.md`. Public pretrained models and course material are allowed, with citation.

## 2026-09-29

- **MERT-v2 extraction runs in fp32.** The real-weight smoke test on transformers 5.17.0 passes: 24 layers of (750, 1024), all finite, 0.12 s per clip, 2.8 GB peak. bf16 and fp16 are 2x faster but deviate from fp32 by up to 32% (relative error) on some layer's pooled vector, with minimum cosine 0.948. Layer-wise probing needs faithful per-layer features, and fp32 is fast enough (about 5 min for all 2,292 clips).
- **CLAP on transformers 5.17: audio embeddings used, zero-shot text dropped (known issue).** Three API changes were fixed in `features/clap.py`:
  - the processor keyword is now `audio=`, not `audios=`;
  - `get_*_features` returns an output object, so the code takes `.pooler_output`;
  - runs use `HF_HUB_OFFLINE=1`, because an erroring run hung instead of exiting (possibly a background Hub request; not confirmed).

  The text tower still gives near-identical embeddings for unrelated prompts (cosine 0.999, e.g. "a dog barking" vs "heavy metal"), even at the CLS hidden state and with explicit RoBERTa position IDs, although all weights load without missing keys. Probable cause: a 5.17 regression in the CLAP text model, not investigated further (time-boxed). Zero-shot needs the text tower, so it is excluded unless re-checked in an isolated older-transformers environment.
- **All reported numbers are also exported as CSV** (`scripts/export_tables.py` → `results/tables/*.csv`) for the report. Rerun after each new result.
- **Correction (web check): the constant CLAP text embeddings come from the checkpoint, not transformers 5.17.** MTEB issue #5069 (https://github.com/embeddings-benchmark/mteb/issues/5069, Aug 2026) reports that `laion/larger_clap_music` gives near-constant text embeddings (mean pairwise cosine 0.9993) on both transformers 4.57.6 and 5.14.1. `laion/larger_clap_general` and `laion/clap-htsat-unfused` behave normally there. This matches our measurement (0.999). The three API fixes for transformers 5 (`audio=`, `pooler_output`) are still needed; others report the same return-type change (torchmetrics #3407, google-research/mseb PR #559). Whether the checkpoint's audio tower is also degraded is unknown; the issue only tested text retrieval.
