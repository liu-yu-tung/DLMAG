# HW1 plan: music era and release-market classification

## Status and remaining plan (updated 2026-09-29 19:32)

Deadline 2026-10-05 23:59. Grade bar: A needs work beyond the basics in implementation, performance or analysis; A+ a creative approach with strong accuracy.

### Submission now

| Task | Recipe | Validation S (top-1 / top-3) | 5-fold out-of-fold S |
|---|---|---|---|
| A | MERT-v2 probe with ordinal soft labels (eps 0.1) + zero-shot Qwen2-Audio (label-mean removed), temperature-calibrated equal weight | 0.977 (0.523 / 0.909) | 0.947 (+0.042 over MERT, CI 0.014 to 0.071) |
| A fallback (`predict.py --no-alm`) | ordinal MERT alone | 1.011 (0.553 / 0.917) | 0.916 |
| B | MERT-v2 probe + Whisper language ID (mixture), equal weight | 1.069 (0.627 / 0.882) | 1.023 (+0.059, CI 0.038 to 0.081) |

- Checkpoints: `checkpoints/{A,B}.joblib` (train split only). `predict.py` from audio (MERT, Whisper, Qwen) matches `outputs/pred_from_cache.json` on all 234 test clips, including label order (checked 9/29 19:32).

### Selection method (fixed)

- Same 5 stratified train folds for every candidate; adopt only a gain above 0.03 with a bootstrap CI above zero; nested checks for anything learned on top; validation is a check only.

### Done today (all logged in `docs/worklog/`, pushed)

- Stems, mix balance, language ID, Demucs U-Net encoder features, Short-Chunk CNN (5 folds + 3 seeds per task, ordinal and no-gain variants), MuQ (ported to transformers 5), Qwen2-Audio (validation, train and test scores, explanations), fusion rules (calibrated, per-class weights, confusion-matrix Bayes, language gating), non-MERT signal analysis, Qwen vs Whisper comparison.
- Only two things beat plain MERT: language on B, and ordinal MERT + Qwen on A.

### Remaining work

| Day | Item | Notes |
|---|---|---|
| 9/30 | **Deliverables:** README (setup, data layout, `predict.py` usage, both modes, runtime and GPU needs), inference-only `requirements.txt`, reproduction in a fresh venv (both modes), cloud upload (code + checkpoints, no data or model caches) | the TA must be able to run it |
| 9/30 | **Report skeleton** (16:9, about 10-11 slides; outline below) and figures from `results/` | 50% of the grade |
| 9/30 night (optional) | MERT top-layer fine-tuning on B, 2 seeds, fixed epochs; adopt only if it beats the probe by more than 0.05 | user decides; otherwise skip |
| 10/1 | **Freeze the recipe**; final JSON; re-run the reproduction check | |
| 10/2-10/4 | Write the report | |
| 10/5 | Buffer, submit | |

### Report outline

1. Task, data, metric, chance level.
2. Method: frozen encoders + probes; the 5-fold selection protocol and why validation alone misled us twice.
3. MERT layer sweep (A flat, B upper layers); the rotary-buffer bug and fix.
4. Final recipes, scores, confusion matrices.
5. Source separation: which input carries each label; mix-balance trends; stem spectra.
6. Week-4 U-Net idea: Demucs encoder features (0.85 on A alone, redundant with MERT).
7. Language on B: Whisper language by market; Qwen vs Whisper.
8. Signal outside MERT: standalone, leave-one-out, rank on MERT's misses; per-class fusion weights.
9. Ordinal structure of A: error distance, soft labels, why higher eps fails.
10. ALM: Qwen2-Audio zero-shot, prompts, label bias, explanations; why it complements MERT on A.
11. Limits: mono, 12 kHz band limit, small validation, label ambiguity, MuQ port caveat.

### Open decisions for the user

- Refit final models on train + validation: allowed by the HW?
- B fine-tuning on the 9/30 night, or skip for the report?

## Original plan (9/28), kept for the record


## Context

HW1 of DLMAG (CommE5070, NTU 2026). The deadline is 2026-10-05 23:59, 7 days from today (9/28).
- **Points:** report 50% + accuracy 50%. Accuracy points = 50 × (S_A + S_B) / 2, with S = Top-1 + 0.5 × Top-3.
- **Grades:** A- if all basic requirements are met; A if the work goes beyond them in implementation, performance or analysis; A+ for a creative approach with strong accuracy.
- **Required:** a model for Task 1, a model for Task 2, an ALM run on both datasets, the report, the prediction JSON, and a reproducible cloud folder.
- **Teacher's hints:** MERT, log-mel Short-Chunk CNN, CLAP.

**Priority: classification first, the ALM after, ordered by points per hour.**

State (9/28): `~/Developments/DLMAG/hw1` is a uv project (Python 3.12) with no deps yet.
- **Data:** unzipped to `hw1/data/dataset_A` (1.8 GB) and `hw1/data/dataset_B` (1.4 GB), all ignored by git.
- **Check:** 1,290 and 1,002 WAVs, matching the manifest rows; none missing. Every file is mono 16-bit, 24 kHz, 720,000 frames (exactly 30 s).
- **Paths:** `audio_path` in the manifest is relative to each dataset folder.
- **Leftovers:** the original zips and a stale `.part` are still in `hw1/data/HW1/`.

## Categories (from `manifest.csv`)

Label strings exactly as they appear in the manifest, which must also be the output strings in the JSON.

**Dataset A: Task 1, release decade (US releases)**

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| `1960s` | 171 | 22 | hidden |
| `1970s` | 171 | 22 | hidden |
| `1980s` | 171 | 22 | hidden |
| `1990s` | 171 | 22 | hidden |
| `2000s` | 171 | 22 | hidden |
| `2010s` | 171 | 22 | hidden |
| **Total** | **1,026** | **132** | **132** |

The classes are ordered in time, so a neighboring-decade error is "less wrong" than a distant one.

**Dataset B: Task 2, release market (1980s releases)**

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| `US` | 133 | 17 | hidden |
| `UK` | 133 | 17 | hidden |
| `Brazil` | 133 | 17 | hidden |
| `Spain` | 133 | 17 | hidden |
| `Germany` | 133 | 17 | hidden |
| `Italy` | 133 | 17 | hidden |
| **Total** | **798** | **102** | **102** |

The classes have no order. Groupings worth checking in the confusion matrix:
- **English-language:** US, UK
- **Romance-language:** Brazil (Portuguese), Spain, Italy
- **Germanic, non-English:** Germany

## Facts from the data (read from the zips without extracting)

- **Layout:** `dataset_X/audio/*.wav` (flat), `manifest.csv`, `README.md`. The split is a column, not a folder as the slide showed.
- **Manifest columns:** `sample_id, split, label, audio_path, duration_seconds, sample_rate, sha256`. Test labels are empty.
- **Audio:** every clip is exactly 30 s at 24 kHz, decoded from Opus. The README says compression losses are not recovered.
- **Balance:** perfect.
  - A: 171 per class in train, 22 per class in validation.
  - B: 133 per class in train, 17 per class in validation.
- **Chance level:** Top-1 16.7%, Top-3 50%, so S = 0.417, about 21 of the 50 accuracy points.
- **Validation is small:** 132 and 102 clips, so one clip is about 0.8-1% accuracy. Model selection is noisy: prefer simple probes and few choices.
- **Train CV:** the manifest has no artist field, so cross-validation on train would leak artists across folds and look too optimistic. Use validation only for selection.
- **Rules:**
  - The final model is trained on train only; the rules forbid training on validation.
  - The ALM "selected split" should be validation, because metrics need labels.

## Shared pipeline (built once, used by both tasks)

1. **`data.py`:** reads the manifest, loads the WAVs (24 kHz, no resampling for MERT), and maps labels.
2. **`metrics.py`:** Top-1, Top-3, S, confusion matrix (counts and row-normalized), and a plot.
3. **`extract_*.py`:** cache features once per model into `features/{A,B}_{model}.npz`.
4. **`probe.py`:** z-score on train, then logistic regression or a small MLP, selected by validation S.
5. **`predict.py --data <dataset dir>`:** loads the checkpoint and writes `<id>.json` with the top-3 in descending order.

## Task 1: release decade (dataset A, US, 1960s-2010s)

- **Likely cues:**
  - Production and recording technology: drum sound, reverb, compression and loudness, synths vs tape, vocal processing.
  - Genre drift over the decades.
  - The Opus decode at 24 kHz caps bandwidth at 12 kHz for every clip, so high-frequency "era" cues are gone; worth one line in the report.
- **Models, in order:**
  1. **MERT-v2-30s**, frozen: all 24 layers, mean+std pooled, then a per-layer logistic-regression sweep, then the best layer or a learned layer weighting.
  2. **CLAP** (`laion/larger_clap_music`):
     - Zero-shot with prompts like "a song from the 1970s", as a no-training baseline.
     - Its embedding as a probe, then fused with MERT.
  3. **Short-Chunk CNN** on log-mel, trained from scratch: random ~3.7 s crops, averaged over chunks at eval. It's the learned-from-scratch comparison.
- **Ideas specific to decades:**
  - The classes are ordered, so errors should fall on neighboring decades. The HW asks exactly this.
  - Try ordinal-aware training: label smoothing toward neighboring decades, or an extra year-regression head. It can raise Top-3, because the top 3 tend to form a contiguous window.
  - Report the share of errors within ±1 decade.
- **Expected confusions:** 1960s↔1970s, and 2000s↔2010s. 1980s may stand out (synths, gated reverb).

## Task 2: release market (dataset B, 1980s, US/UK/Brazil/Spain/Germany/Italy)

- **Likely cues:**
  - **Sung language:** Portuguese, Spanish, Italian, German, or English for both US and UK.
  - **Regional style:** MPB, Italo-disco, Schlager, and so on.
  - Every clip is from the same decade, so production era is not a cue here.
- **Models, in order:**
  1. **MERT-v2-30s**, frozen: same pipeline as Task 1, with its own layer choice. Language cues may sit in different layers.
  2. **CLAP:** zero-shot ("a Brazilian song from the 1980s") plus the probe and fusion.
  3. **Short-Chunk CNN:** same as Task 1.
  4. **Beyond the hints (the A/A+ step):**
     - Separate vocals with Demucs, then run Whisper-large-v3-turbo language ID on the vocal stem to get a 99-dim language-probability vector.
     - Fuse it with MERT.
     - This also covers the optional "mixture vs vocal vs accompaniment" experiment.
- **Expected confusions:**
  - US↔UK (both English), which is the main error source.
  - Spain↔Italy, Brazil↔Spain (related languages).
  - Instrumental tracks carry no language cue.
  - Top-3 should absorb much of the US/UK confusion.
- **Caveat:** the HW says release market is not the artist's language. Treat language as an acoustic cue, not the label definition, and say so in the report.

## ALM part (required, after classification)

- **Model:** Qwen2-Audio-7B-Instruct in 8-bit (bf16 needs about 16.8 GB, more than the 16 GB card), on the validation splits of A and B.
- **Answers:** score the log-likelihood of each of the 6 label strings instead of parsing free text. That always gives a valid ranked top-3, which settles the invalid-output requirement.
- **Two prompts:** plain zero-shot vs cue-guided (production hints for A; language + style for B).
- **Report:** Top-1, Top-3, confusion matrices, and a comparison with the trained models.

## Priority by points and time (rough estimates)

| Pri | Item | Points it serves | Est. time |
|---|---|---|---|
| P0 | Unzip, env, `data.py` / `metrics.py`, MERT-v2 smoke test | everything | 3 h |
| P0 | MERT-v2 features + probe, A and B, val confusion matrices | accuracy (most of it) + A- requirements | 4 h |
| P0 | `predict.py` + JSON + fresh-venv check | accuracy (no JSON = 0) | 2 h |
| P1 | ALM on A and B validation, 2 prompts | required report item | 5 h (setup risk) |
| P1 | Report slides (methods, CMs, error analysis, citations) | 50% | 6 h |
| P2 | CLAP zero-shot + probe + fusion | report depth + maybe accuracy | 3 h |
| P2 | Short-Chunk CNN baseline | report depth ("what the model learned") | 4 h |
| P3 | B: Demucs + Whisper language feature | accuracy on B + A/A+ creativity | 4 h |
| P3 | A: ordinal smoothing / year head | accuracy on A (Top-3) + analysis | 2 h |
| P4 | t-SNE/UMAP, input-length sweep, partial fine-tuning | extra analysis | as time allows |

The P0 + P1 rows total about 20 h and meet every requirement. The P2 and P3 rows are where the A grade comes from.

## Day plan

- **9/28-29:** P0 setup, MERT-v2 for A and B, first validation numbers, a first `predict.py` and JSON (a safety submission).
- **9/30:** CLAP (P2), Short-Chunk CNN started (P2).
- **10/1:** CNN done; Task 2 Demucs + Whisper (P3); Task 1 ordinal (P3).
- **10/2:** ALM (P1).
- **10/3:** final model per task by validation S; final JSON; README, `requirements.txt`, fresh-venv reproduction; cloud upload.
- **10/4-5:** report slides, buffer, submit.

## Packages and models

- **uv deps:** `torch torchaudio numpy pandas soundfile scikit-learn joblib tqdm matplotlib librosa transformers accelerate huggingface-hub safetensors`.
- **Added later:** `demucs` (P3), `bitsandbytes` (ALM).
- **Models:**
  - `m-a-p/MERT-v2-30s` (fallback `m-a-p/MERT-v1-330M`)
  - `laion/larger_clap_music`
  - `openai/whisper-large-v3-turbo`
  - Demucs `htdemucs`
  - `Qwen/Qwen2-Audio-7B-Instruct`
- **Left out:** MuQ and Audio Flamingo 3 (only if time remains); Jukebox and SALMONN (too big); PANNs, AST, BEATs (general sound-event models).

## Risks

- **MERT-v2 vs `transformers`:** the card pins `transformers==4.53.2`. Test on day 1; pin if needed.
- **ALM vs that pin:** if they conflict, give the ALM its own uv project (`hw1/alm/`), outside the submitted requirements.
- **Demucs:** unknown whether it works with current torchaudio. Test one clip before relying on it.
- **Git:** add `slides/` and `*.pdf` to `.gitignore` before the next commit.

## Verification

- 1,290 + 1,002 WAVs extracted; the counts per split and label match the table above.
- The validation Top-1, Top-3, S and a row-normalized confusion matrix are saved for every model per task.
- `predict.py` in a fresh venv from `requirements.txt` reproduces the JSON, which has 132 A IDs + 102 B IDs, each with 3 distinct valid labels. Compare its structure with `prediction_format_example_NOT_ANSWERS.json`.
