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
