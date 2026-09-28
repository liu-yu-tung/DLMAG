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
