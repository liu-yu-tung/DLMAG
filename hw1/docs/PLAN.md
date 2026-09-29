# HW1 plan: music era and release-market classification

## Status and remaining plan (updated 2026-09-29 16:20)

Deadline 2026-10-05 23:59. Grade bar (from the HW): A needs work beyond the basics in implementation, performance or analysis; A+ needs a creative approach with strong accuracy.

### Submission now (verified, pushed)

- **A:** MERT-v2 all-layer average probe. Validation S 0.943 (top-1 0.500, top-3 0.886); 5-fold out-of-fold 0.906.
- **B:** MERT-v2 + Whisper language ID (mixture), equal-weight log-prob fusion. Validation 1.069 (top-1 0.627, top-3 0.882); out-of-fold 1.023, +0.059 over MERT alone (CI 0.038 to 0.081).
- `scripts/predict.py` from audio matches the cached-feature predictions on all 234 test clips.

### Selection method (fixed)

- Every candidate is scored on the same 5 stratified train folds (out-of-fold S, paired bootstrap CI against the current recipe); validation is a check only.
- Adopt a change only if it gains more than 0.03 out-of-fold with a CI above zero; learned layers on top (stackers, remaps) need a nested check.

### What is settled

- **A is near a ceiling:** MERT, the Demucs encoder features (0.852), the CNN (0.813) and hand-crafted features all carry real signal, but no fusion, ordinal loss (+0.01 to +0.02) or remap (+0.026) clears the bar.
- **B:** language is the only signal that complements MERT; CNN, Demucs, CLAP and mix balance lower it.
- **Stems (report):** accompaniment carries A, vocals carry B; separation is analysis, not accuracy.

### Running now

| Job | Expected done |
|---|---|
| Qwen2-Audio plain prompt on train + test, B then A (`scripts/alm_train_test.sh`) | B about 16:30, A about 19:00 |
| CNN no-gain folds on A (`scripts/day_0929b.sh`) | about 16:40 |
| MuQ-large download (1.33 GB), then two-load check, extraction and `scripts/cv_muq.py` | results about 17:30-18:00 |
| Qwen explanations, 16 validation clips (`scripts/alm_explain.py`) | about 19:05 |

### Remaining work, ranked

| Pri | Item | Serves | When |
|---|---|---|---|
| P1 | Qwen screen: train-fitted label-bias correction, out-of-fold fusion with the B recipe; ALM confusion matrices vs MERT | required ALM item, maybe B accuracy | 9/29 evening |
| P1 | MuQ screen: alone, with MERT, with the current recipes | the one untested source of new information for A | 9/29 evening |
| P1 | Report (16:9 PDF, about 10 pages) | 50% of the grade | skeleton 9/30, write 10/2-10/4 |
| P1 | Freeze recipe; README, inference-only `requirements.txt`, fresh-venv reproduction, cloud upload | required deliverables | 10/1 |
| P2 | MERT top-layer fine-tuning on B, overnight, 2 seeds, fixed epochs | B accuracy | 9/30 night, only if MuQ and Qwen leave B unchanged |
| P2 | Refit final models on train + validation | small free gain | 10/1, only if the HW allows it (open question) |
| P3 | Report-only analyses: A error clustering, per-class confusion, t-SNE of MERT/MuQ | analysis depth | during report writing |

- **Dropped:** more hand-crafted or stem features, more fusion variants, ordinal CNN, A fine-tuning (A's signal is spread over all layers), MuQ-MuLan zero-shot (Qwen covers zero-shot), MARBLE runs.

### Report outline (about 10 slides)

1. Task, data, metric, chance level.
2. Method overview: frozen encoders + probes, the out-of-fold selection protocol.
3. MERT layer sweep (A flat, B upper layers) and the rotary-buffer bug found and fixed (reproducibility).
4. Final recipes and validation / out-of-fold scores; confusion matrices.
5. Source separation: which input carries each label; mix-balance trends by decade; stem spectra.
6. Week-4 U-Net idea: Demucs encoder features as a classifier input (0.85 on A).
7. Language: Whisper language ID on B (language by market table); why it helps and nothing else does.
8. Signal outside MERT: standalone, leave-one-out, complementarity on MERT's misses.
9. Short-Chunk CNN from scratch; ordinal soft labels (A errors are 62% one decade off).
10. ALM: Qwen2-Audio zero-shot, prompt comparison, label bias, its explanations; comparison with trained models.
11. Limits: mono, 12 kHz band limit, small validation, label ambiguity.

### Open decisions for the user

- Refit on train + validation: allowed by the HW?
- B fine-tuning on 9/30 night: run it, or spend the time on the report?

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
