# Worklog: Short-Chunk CNN baseline

## Changes

- `src/hw1/models/shortchunk.py`: Short-Chunk CNN after Won et al. 2020: input BatchNorm, 7 blocks of 3x3 conv + BN + ReLU + 2x2 max-pool (128,128,256,256,256,256,512 channels), global max-pool, dense 512 + BN + ReLU + dropout 0.5, dense 6. Plus crop helpers.
- `src/hw1/features/logmel.py`, `scripts/extract_logmel.py`: ln(1e-6 + mel power), 24 kHz, n_fft 2048, hop 512, 128 mel bins, cached as float16 `features/{A,B}_logmel.npy` (N, 128, 1407) with a meta npz in manifest order. B takes 25 s with 3 jobs.
- `scripts/train_cnn.py`: random 3.7 s crops (160 frames), 4 crops per train clip per epoch, batch 32, AdamW lr 1e-3, weight decay 1e-4, cosine schedule, 60 epochs, resumable. Augmentation: random gain (+-6 dB), one time mask (up to 24 frames) and one frequency mask (up to 16 bins) per crop. Clip score = mean of crop log-softmax over 8 evenly spaced crops.
- `scripts/night_cnn.sh`: the overnight launcher (log-mel for A if missing, then 3 seeds each for A and B).

## Decisions

- **Fixed 60 epochs, chosen before any run.** Validation is logged per epoch (`runs/cnn_{key}_s{seed}/log.csv`) for the report curve and is never used to pick an epoch; the final-epoch weights are saved. The number is a normal budget for a small from-scratch CNN on about 1,000 clips, not tuned.
- **Train split only.** The test split gets predictions (log-probs) but no labels are read.
- **Mixing later:** `features/{key}_cnn_s{seed}_logp.npz` holds validation and test clip log-probs, so it can join the equal-weight log-probability mixtures.

## Tests

- `tests/test_shortchunk.py` (5 tests): output shape, crop sizes, crop start ranges, log-mel shape. Whole suite: 38 pass.
- Dry run on B, 64 clips, 2 epochs, then resume to 3: loop, resume and outputs work (files removed afterwards). Speed about 200 crops/s with evaluation, peak GPU under 1 GB.

## Open issues

- **Time estimate:** about 15-20 min per seed for B (3,192 crops per epoch), about 20-25 min for A; 3 seeds each is about 2 h. Not yet measured on full data.
- **From-scratch on about 1,000 clips is expected to be well below the MERT probes;** its value is as the "learned from scratch" comparison in the report and a possible mixture member.
- **Run:** `nohup scripts/night_cnn.sh` at 23:00 (needs a free GPU; it shares nothing else). Status in `runs/night_cnn/status.txt`.
