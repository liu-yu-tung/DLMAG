# Worklog: Demucs stems and mix-balance features

## Setup (2026-09-29)

- **Separator:** torchaudio `HDEMUCS_HIGH_MUSDB_PLUS` (Hybrid Demucs v3, trained with extra data; week 4 p.87-92), no extra package. Weights: 335 MB in `~/.cache/torch/hub/torchaudio/models/`.
- **Input handling** (`src/hw1/features/separation.py`): 24 kHz mono is resampled to 44.1 kHz and duplicated to stereo; 10 s segments, 0.1 s linear cross-fade (overlap-and-add, week 4 p.92); stems averaged to mono and resampled back to 24 kHz. Our clips are mono, so the stereo cues Demucs could use are absent.
- **Storage:** `data/stems/dataset_{A,B}/{sample_id}/{drums,bass,other,vocals}.flac`, 16-bit, about 3.4 MB per clip (about 7.8 GB total, not committed).

## Smoke test (`scripts/smoke_demucs.py`, one train clip per class, `results/demucs_smoke.json`)

- 0.36 s per clip after warm-up, peak GPU 833 MiB.
- Stems sum back to the mixture at 25-36 dB SNR.
- Energy shares vary in plausible ways: two B clips (UK, Germany) have 0% vocals (likely instrumentals); one A clip (2000s) has no drums or bass.

## Features

- **MERT-v2 per stem:** `extract_mert.py --stem {vocals,drums,bass,other}` → `features/{key}_mertv2_{stem}.npz`, same layout as the mixture features.
- **Mix balance (29 features, `src/hw1/features/mixbalance.py`):** per stem: energy share, level relative to the mix, active-frame fraction (within 20 dB of the mixture frame), frame-level dB spread, crest factor, spectral centroid; plus vocals/drums/bass-to-rest ratios, vocal-to-accompaniment level while singing, and the separation residual.
- **Overnight run:** `scripts/night_0929.sh` (stems → MERT on stems → mix balance), logs in `runs/night_0929/`.

## Results (validation S; chance 0.417; noise about +-0.05; `results/stems.json`, `results/tables/stems.csv`)

- **Inputs, one probe each (MERT all-layer average):**

| Input | A | B |
|---|---:|---:|
| mixture | 0.943 | 1.034 |
| vocals | 0.746 | 1.015 |
| accompaniment (drums+bass+other) | 0.947 | 0.819 |
| drums / bass / other | 0.849 / 0.776 / 0.788 | 0.676 / 0.691 / 0.745 |
| mix-balance features (29) | 0.640 | 0.637 |
| language ID, vocals / mixture | 0.561 / 0.651 | 0.951 / 0.980 |

- **Reading:**
  - **A (decade)** is carried by the accompaniment (0.947, equal to the mixture); the vocal stem alone is much weaker (0.746). Drums are the best single stem (0.849).
  - **B (market)** is the opposite: vocals alone reach 1.015, the accompaniment only 0.819. This fits H3 (language and vocal style).
  - Mixture plus stems never beats the mixture by more than noise (A 0.928-0.943, B 0.990-1.029).
- **Mix balance:** weak alone (0.64 on both), but on B it lifts the recipe from 1.034 to 1.073 (top-1 0.608 to 0.647), a gain inside the noise band. On A it changes nothing (0.947).
  - Strongest A trends (train ANOVA F): `drums_share` 15.9 (1960s 0.08, 1970s 0.13, 1980s 0.21, 1990s 0.21, 2000s 0.16, 2010s 0.18), `other_crest_db` 11.3 (about 17.8 dB until the 1990s, 16.2-16.4 in the 2000s-2010s), `bass_frame_db_std` 10.4 (4.9 in the 1960s, falling to 3.6-3.9).
  - B features are weak (F at most 5.5; vocal crest, vocal activity).
- **Language ID (Whisper turbo, one decoder step):**
  - **B:** 100 numbers reach 0.98 from the mixture, close to MERT (1.034). Train cross-table of the vocal-stem top-1 language: Brazil 117/133 pt; Germany 22/133 de but 106 en; Italy 57/133 it; Spain 68/133 es; US and UK are almost all en (123/133, 127/133).
  - The vocal stem does not help language ID (0.951 vs 0.980 from the mixture): Demucs artifacts cost more than the removed accompaniment.
  - Fusing with MERT: 1.034 to 1.069 (mixture), or 1.034 (vocals): no gain beyond noise.
  - **A:** language is nearly constant (en for 1,217 of 1,290 clips; the top-1 varies only in the small Spanish share, 8 clips in the 1960s to 4 in the 2010s). Fusion gives 0.992 (+0.045), which is inside the noise and has no plausible mechanism; treated as chance.
- **Selection rule outcome:** nothing beats the current recipes by more than 0.03 with a clear reason. The B mix-balance gain (+0.039) is inside the noise band and costs a Demucs pass at inference; not adopted for now. Revisit only if the CNN or fine-tuning changes the picture.
