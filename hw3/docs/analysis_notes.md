# Analysis notes

What `scripts/analyze_runs.py` and `tests/test_apg.py` produce, and which report claim each output supports.

## Commands

```bash
./.venv/bin/python tests/test_apg.py
./.venv/bin/python scripts/analyze_runs.py --set dev --baseline cfg7 --threshold-run cfg7
```

Outputs go to `results/analysis/<set>/`. The same inputs give the same files (fixed bootstrap seed, ids sorted).

## Inputs

Per run, from `results/<set>__<run>/`: `gen_info.json` (settings), `quick.jsonl` (one line per clip; last line wins for a repeated id; a cut-off line is skipped), `trajectory.jsonl` (per-step guidance statistics).

## `runs.csv`, `runs.md`

One row per run, sorted by mode, then scale.

- **mode, scale, eta, momentum, threshold:** guidance settings; the APG fields are empty for CFG runs.
- **n:** clips in `quick.jsonl`.
- **clap_quick_mean, CI:** mean quick CLAP and a 95% bootstrap interval over clips (2000 resamples). Quick CLAP is the in-loop score (HF `laion/larger_clap_music`), not the TA metric; use it to rank runs, not as the reported number.
- **n_shared, clap_quick_difference, CI:** run minus baseline on shared ids, bootstrap over per-clip differences. An interval that excludes 0 is the evidence that a run beats or loses to the baseline. Compare runs with the same `seed_index`, so paired clips share the starting noise.
- **median_peak:** largest absolute sample before clamping; above 1 means the decoder output went past full scale.
- **clip_fraction_percent:** share of samples at or beyond full scale.
- **median_rms:** loudness before clamping.
- **mean_seconds_per_clip:** generation time.

## `apg_threshold.json`

The APG norm threshold `r` (`apg_norm_threshold`) caps the L2 norm of `D_cond - D_uncond` at each step.

- **Rule:** choose `r` comparable on average to the norm of the difference (Sadat et al., ICLR 2025, appendix on hyperparameters; checked in the review of 11-15).
- **Measured:** median of `difference_norm` over all clips and steps of a CFG run (`--threshold-run`), over the whole 64 x 1024 latent, because that is the norm `apg()` rescales. `difference_norm_valid` (the frames of the requested 10 s) is listed for comparison.
- **suggested_thresholds:** `median` and `twice_median`.
- **per_step_median:** first, middle and last step. If they differ a lot, one fixed `r` acts on only part of the schedule; say so in the report.

## `saturation_vs_scale.png`

Clipped samples (percent) and median RMS against guidance scale; CFG runs as a line, each APG run as a marker.

- **Claim it tests:** the APG paper's premise that high guidance oversaturates. For audio that shows as RMS and clipping rising with the CFG scale.
- **Line rises:** the premise holds; an APG marker below the line at the same scale shows APG reduces it.
- **Line flat:** the premise does not carry over to this model; report that as a result.
- **Limit:** waveform measures only; pair with listening notes.

## `trajectory.png`

Per-step median over clips of `difference_norm` (left) and `parallel_fraction` (right), one line per run, interquartile band for the baseline.

- **parallel_fraction:** absolute cosine between the difference and `D_cond`; its square is the share of the update's energy along `D_cond`, which APG with `eta = 0` removes. Near 0: APG and CFG give almost the same update.
- **Limit:** logged before guidance is applied, on each run's own latents.

## `tests/test_apg.py`

CPU checks: v to denoised round trip, APG with `eta = 1` equals CFG, the `eta = 0` update is orthogonal to `D_cond`, the norm cap, the momentum limit, argument checks, `guidance_stats`.

- **Momentum:** with a steady difference the buffer converges to difference / (1 - momentum), so momentum -0.5 gives an effective scale `(w - 1) / 1.5`. Compare such an APG run against CFG at that effective scale, not only at the nominal one.
