
# HW3 deck numbers (checked sheet)

Written by `scripts/analysis/deck_numbers.py` from files in `results/`. Intervals are 95% bootstrap over the 100 prompts. Values from `findings_table.csv`, `knowledge.md` and the analysis CSVs use their own bootstrap (1000 resamples, seed 0); values marked 'computed' use 2000 resamples, seed 0, on the per-item files. Test appears only as aggregate numbers.


## 1. Main metric table

Baseline `cfg7` (CFG 7, seed 0); `rescale12` (CFG 12, rescale 0.7, seed 0); `rescale12 best-of-3 (c8998 ranker)` is the submission (best of 3 seeds per prompt, ranked by the CLAP 89.98 checkpoint, scored by the 90.14 checkpoint).

- **dev baseline**: CLAP 0.299 [0.280, 0.318]; KL-PaSST 1.506 [1.271, 1.741]; FD-VGG 4.02 (vs all MusicCaps, point value); FD-MERT 43.5; MQAScore 0.546 [0.502, 0.588]; reference (n=100 prompts) — `results/dev__cfg7/metrics.json, results/analysis/findings_table.csv`
- **dev rescale12**: CLAP 0.326 [0.309, 0.344]; KL-PaSST 1.421 [1.192, 1.656]; FD-VGG 4.13 (vs all MusicCaps, point value); FD-MERT 41.0; MQAScore 0.552 [0.512, 0.594]; vs cfg7 paired: dCLAP +0.028 [+0.017, +0.039], dKL -0.084 [-0.170, +0.003], dMQAScore (computed, n=98) +0.007 [-0.010, +0.024], dFD-VGG +0.11 [-0.27, +0.36] (n=100 prompts) — `results/dev__rescale12/metrics.json, results/analysis/findings_table.csv, results/analysis/paired_fd_dev.json`
- **dev rescale12 best-of-3 c8998 (submission)**: CLAP 0.341 [0.322, 0.361]; KL-PaSST 1.297 [1.100, 1.510]; FD-VGG 4.02 (vs all MusicCaps, point value); FD-MERT 41.3; MQAScore 0.566 [0.525, 0.608]; vs cfg7 paired: dCLAP +0.042 [+0.027, +0.057], dKL -0.209 [-0.321, -0.099], dMQAScore (computed, n=98) +0.020 [-0.001, +0.042], dFD-VGG -0.00 [-0.43, +0.36] (n=100 prompts) — `results/dev__rescale12_best3_c8998/metrics.json, results/analysis/findings_table.csv, results/analysis/paired_fd_dev.json`
- **test baseline**: CLAP 0.314 [0.294, 0.335]; KL-PaSST 0.950 [0.786, 1.136]; FD-VGG 8.16 (vs all MusicCaps, point value); FD-MERT 55.3; MQAScore 0.651 [0.616, 0.682]; reference (n=100 prompts) — `results/test__cfg7/metrics.json, results/analysis/findings_table.csv`
- **test rescale12**: CLAP 0.325 [0.304, 0.346]; KL-PaSST 0.929 [0.775, 1.101]; FD-VGG 8.01 (vs all MusicCaps, point value); FD-MERT 52.0; MQAScore 0.658 [0.626, 0.687]; vs cfg7 paired: dCLAP +0.010 [-0.000, +0.021], dKL -0.021 [-0.101, +0.053], dMQAScore (computed, n=100) +0.007 [-0.009, +0.023], dFD-VGG not computed for test (paired FD file is dev only) (n=100 prompts) — `results/test__rescale12/metrics.json, results/analysis/findings_table.csv`
- **test rescale12 best-of-3 c8998 (submission)**: CLAP 0.354 [0.333, 0.375]; KL-PaSST 0.883 [0.721, 1.064]; FD-VGG 8.08 (vs all MusicCaps, point value); FD-MERT 52.7; MQAScore 0.672 [0.643, 0.700]; vs cfg7 paired: dCLAP +0.040 [+0.027, +0.051], dKL -0.067 [-0.170, +0.031], dMQAScore (computed, n=100) +0.021 [+0.003, +0.039], dFD-VGG not computed for test (paired FD file is dev only) (n=100 prompts) — `results/test__rescale12_best3_c8998/metrics.json, results/analysis/findings_table.csv`

Notes: MQAScore is over the 98 dev captions that have at least one tag (100 on test). FD-MERT has no paired interval (per-run MERT embeddings are not cached; `paired_fd_dev.json`, key `fd_mert`). The FD interval in `metrics.json` is a per-run bootstrap, not a difference. dev FD floor for 100 real clips: FD-VGG 1.9, FD-MERT 21 (`docs/findings.md`).


## 2. Caption-word study


### 2a. Probes and detectors

- Probe groups tested on held-out real dev clips: 28; valid (MERT AUC >= 0.75, rule in `valid_groups`): **22**; valid MERT AUC range 0.76 to 0.95, median 0.85 — `results/analysis/probe_validity.csv, scripts/selection/probe_rank.py (valid_groups)`
- CLAP 90.14 probe AUC >= 0.75 on 22 of 28 groups; median 0.84 — `results/analysis/probe_validity.csv`
- Highest MERT AUCs: genre: hip hop, r&b, funk 0.95; genre: electronic, dance 0.94; vocals: male voice 0.93; instrument: brass, winds 0.93; genre: latin, reggae, world 0.91; vocals: female voice 0.91 — `results/analysis/probe_validity.csv`
- Invalid groups (MERT AUC < 0.75): genre: ambient, soundtrack 0.35; context: use or setting 0.63; dynamics: soft, calm 0.66; recording: low quality 0.67; timbre: shimmering, warm, lush 0.72; tempo: fast 0.72 — `results/analysis/probe_validity.csv`
- Validity table, MERT / CLAP AUC per valid group: dynamics: loud, intense 0.78/0.85; genre: electronic, dance 0.94/0.93; genre: hip hop, r&b, funk 0.95/0.85; genre: latin, reggae, world 0.91/0.88; instrument: bass 0.77/0.81; instrument: brass, winds 0.93/0.90; instrument: drums, percussion 0.87/0.82; instrument: guitar 0.89/0.89; instrument: piano, keys 0.84/0.87; instrument: strings 0.85/0.88; instrument: synth, electronic sounds 0.84/0.83; mood: happy, bright 0.77/0.69; mood: sad, emotional 0.86/0.84; other sounds: non-music sounds 0.82/0.90; recording: live, crowd 0.79/0.74; timbre: acoustic 0.80/0.84; timbre: electric, distorted 0.76/0.82; timbre: groovy, punchy 0.78/0.85; vocals: choir, backing vocals 0.85/0.83; vocals: female voice 0.91/0.85; vocals: male voice 0.93/0.95; vocals: no vocals 0.86/0.80 — `results/analysis/probe_validity.csv`
- Probe cross-validated AUC on the training pool: 41 groups, MERT median 0.81, min 0.54; CLAP median 0.82 — `results/analysis/attribute_probes_cv.csv`
- Detector check `tempo`, reader `single` (features: 1): CV AUC 0.676 (n=150+150), fresh sample AUC 0.76 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `tempo`, reader `interpretable` (features: 4): CV AUC 0.823 (n=150+150), fresh sample AUC 0.84 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `tempo`, reader `chroma entropy` (features: 1): CV AUC 0.859 (n=150+150), fresh sample AUC 0.85 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `tempo`, reader `all features` (features: 110): CV AUC 0.907 (n=150+150), fresh sample AUC 0.89 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `loudness`, reader `single` (features: 1): CV AUC 0.695 (n=150+150), fresh sample AUC 0.70 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `loudness`, reader `interpretable` (features: 5): CV AUC 0.774 (n=150+150), fresh sample AUC 0.78 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `loudness`, reader `chroma entropy` (features: 1): CV AUC 0.773 (n=150+150), fresh sample AUC 0.78 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `loudness`, reader `all features` (features: 110): CV AUC 0.805 (n=150+150), fresh sample AUC 0.83 (n=100+100) — `results/analysis/dsp_detectors.csv`
- Detector check `quality`, reader `single` (features: 1): CV AUC 0.714 (n=83+150) — `results/analysis/dsp_detectors.csv`
- Detector check `quality`, reader `interpretable` (features: 5): CV AUC 0.715 (n=83+150) — `results/analysis/dsp_detectors.csv`
- Detector check `quality`, reader `chroma entropy` (features: 1): CV AUC 0.524 (n=83+150) — `results/analysis/dsp_detectors.csv`
- Detector check `quality`, reader `all features` (features: 110): CV AUC 0.763 (n=83+150) — `results/analysis/dsp_detectors.csv`

### 2b. Word-swap edits: which axes are followed

`moved` = pairs where the detector moved in the predicted direction out of n pairs. Registered = the 12 pre-registered pairs, Holm over the registered axes; pooled = registered plus the extension pairs (no Holm). Probe = linear probe on MERT or CLAP 90.14 embeddings of the generated clips, the group of the target word.

- **guitar_to_piano**: MERT: registered 9/12 (p=0.073, Holm p=0.438), pooled 25/28 (p=1.4e-05), mean shift 5.06 [3.34, 6.93]; CLAP90: registered 10/12 (p=0.019), pooled 25/28 (p=1.4e-05), mean shift 5.02 [3.33, 6.88] — `results/analysis/edit_effects.csv`
- **voice_male_to_female**: MERT: registered 10/12 (p=0.019, Holm p=0.135), pooled 21/26 (p=0.001), mean shift 2.83 [1.73, 3.93]; CLAP90: registered 12/12 (p=2.4e-04), pooled 20/26 (p=0.005), mean shift 4.51 [2.81, 6.16] — `results/analysis/edit_effects.csv`
- **acoustic_to_electric**: MERT: registered 11/12 (p=0.003, Holm p=0.025), pooled 14/17 (p=0.006), mean shift 2.35 [1.04, 3.84]; CLAP90: registered 7/12 (p=0.387), pooled 12/17 (p=0.072), mean shift 2.44 [0.82, 4.17] — `results/analysis/edit_effects.csv`
- **insert_jazz**: MERT: registered 8/12 (p=0.194), pooled 39/50 (p=4.5e-05), mean shift 2.42 [1.21, 3.53]; CLAP90: registered 10/12 (p=0.019), pooled 36/50 (p=0.001), mean shift 2.14 [1.20, 3.14] — `results/analysis/edit_effects.csv`
- **insert_jazz_vs_country**: MERT: registered 7/12 (p=0.387, Holm p=0.775), pooled 39/50 (p=4.5e-05), mean shift 2.76 [1.72, 3.90]; CLAP90: registered 11/12 (p=0.003), pooled 42/50 (p=5.8e-07), mean shift 2.62 [1.93, 3.34] — `results/analysis/edit_effects.csv`
- **insert_country**: MERT: registered 7/12 (p=0.387), pooled 27/50 (p=0.336), mean shift 0.34 [-0.59, 1.37]; CLAP90: registered 7/12 (p=0.387), pooled 27/50 (p=0.336), mean shift 0.48 [-0.45, 1.37] — `results/analysis/edit_effects.csv`
- **soft_to_loud**: MERT: registered 7/12 (p=0.387), pooled 10/16 (p=0.227), mean shift 0.91 [-0.06, 1.87]; CLAP90: registered 11/12 (p=0.003), pooled 15/16 (p=2.6e-04), mean shift 1.94 [1.20, 2.79] — `results/analysis/edit_effects.csv`
- **happy_to_sad**: MERT: registered 8/12 (p=0.194, Holm p=0.775), pooled 10/15 (p=0.151), mean shift 1.28 [-0.02, 2.55]; CLAP90: registered 9/12 (p=0.073), pooled 11/15 (p=0.059), mean shift 1.30 [0.42, 2.19] — `results/analysis/edit_effects.csv`
- **fast_to_slow**: MERT: registered 10/12 (p=0.019), pooled 10/14 (p=0.090), mean shift 1.10 [-0.00, 2.24]; CLAP90: registered 8/12 (p=0.194), pooled 10/14 (p=0.090), mean shift 1.41 [0.11, 2.58] — `results/analysis/edit_effects.csv`
- **low_quality_to_studio**: MERT: registered 4/12 (p=0.927), pooled 21/35 (p=0.155), mean shift 1.61 [0.35, 2.84]; CLAP90: registered 9/12 (p=0.073), pooled 17/35 (p=0.632), mean shift 0.03 [-0.85, 0.92] — `results/analysis/edit_effects.csv`

Hand-crafted detectors on the same pairs (pooled moved/n, p binomial):

- fast_to_slow, `dsp: rhythm_tempo_bpm`: registered 4/12 (p=0.927), pooled 5/14 (p=0.910) — `results/analysis/edit_effects.csv`
- fast_to_slow, `reader: all features`: registered 11/12 (p=0.003), pooled 12/14 (p=0.006) — `results/analysis/edit_effects.csv`
- fast_to_slow, `reader: interpretable`: registered 8/12 (p=0.194), pooled 9/14 (p=0.212) — `results/analysis/edit_effects.csv`
- fast_to_slow, `reader: chroma entropy`: registered 6/12 (p=0.613), pooled 7/14 (p=0.605) — `results/analysis/edit_effects.csv`
- soft_to_loud, `dsp: energy_rms_db_mean`: registered 9/12 (p=0.073), pooled 12/16 (p=0.038) — `results/analysis/edit_effects.csv`
- soft_to_loud, `reader: interpretable`: registered 10/12 (p=0.019), pooled 14/16 (p=0.002) — `results/analysis/edit_effects.csv`
- voice_male_to_female, `dsp stem: pitch_f0_median_energy_gated`: registered 8/9 (p=0.020), pooled 17/20 (p=0.001) — `results/analysis/edit_effects.csv`
- low_quality_to_studio, `reader: interpretable`: registered 8/12 (p=0.194), pooled 18/35 (p=0.500) — `results/analysis/edit_effects.csv`
- low_quality_to_studio, `reader: single`: registered 9/12 (p=0.073), pooled 20/35 (p=0.250) — `results/analysis/edit_effects.csv`
- low_quality_to_studio, `reader: all features`: registered 7/12 (p=0.387), pooled 19/35 (p=0.368) — `results/analysis/edit_effects.csv`
- low_quality_to_studio, `dsp: energy_band_6000_12000hz_ratio`: registered 8/12 (p=0.194), pooled 20/35 (p=0.250) — `results/analysis/edit_effects.csv`
- Registered predictions with a result: 8; met 6, missed 2 (missed: fast_to_slow / dsp: rhythm_tempo_bpm moved 4/12, predicted 6-10, insert_jazz_vs_country / probe moved 7/12, predicted 8-12) — `results/analysis/edit_effects.json`
- Registered pairs not run (result null): 4 (insert_1960s_vs_2010s / hw1_decade, insert_brazil_vs_germany / hw1_market, insert_latin_vs_techno / probe, insert_latin_vs_techno / hw1_latin_market) — `results/analysis/edit_effects.json`

### 2c. Meaning-free controls and word vs seed (re-roll share)

`profile_change_vs_seed` = change of the 22-group probe profile under the edit divided by the change under a new seed; `groups_clear_of_zero` = groups whose 95% interval excludes 0 (of 22). Controls: `control_song_to_track` (word swap without meaning) and `control_neutral_insert`.

- MERT: meaning-free controls change the profile 0.72 to 0.74 of a seed change (signed shifts clear of zero: 2, 2 of 22); real edits 0.77 to 1.03 (clear groups up to 8) — `results/analysis/edit_controls.csv`
- MERT per edit (change vs seed / clear groups, n): control_song_to_track 0.72/2, n=24; control_neutral_insert 0.74/2, n=50; voice_male_to_female 1.03/4, n=26; guitar_to_piano 1.03/8, n=28; acoustic_to_electric 1.00/3, n=17; soft_to_loud 0.84/4, n=16; happy_to_sad 0.83/6, n=15; fast_to_slow 0.77/1, n=14; low_quality_to_studio 0.85/4, n=35; insert_jazz 0.92/8, n=50; insert_country 0.81/4, n=50 — `results/analysis/edit_controls.csv`
- CLAP90: meaning-free controls change the profile 0.68 to 0.77 of a seed change (signed shifts clear of zero: 2, 3 of 22); real edits 0.78 to 1.27 (clear groups up to 9) — `results/analysis/edit_controls.csv`
- CLAP90 per edit (change vs seed / clear groups, n): control_song_to_track 0.68/2, n=24; control_neutral_insert 0.77/3, n=50; voice_male_to_female 1.01/7, n=26; guitar_to_piano 1.27/9, n=28; acoustic_to_electric 1.13/5, n=17; soft_to_loud 1.08/7, n=16; happy_to_sad 0.87/7, n=15; fast_to_slow 0.78/5, n=14; low_quality_to_studio 0.97/6, n=35; insert_jazz 0.97/8, n=50; insert_country 0.86/2, n=50 — `results/analysis/edit_controls.csv`
- guitar_to_piano: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 2.23, CLAP 3.52 (n=28) — `results/analysis/edit_effects.csv`
- voice_male_to_female: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 1.05, CLAP 2.39 (n=26) — `results/analysis/edit_effects.csv`
- acoustic_to_electric: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 1.04, CLAP 1.72 (n=17) — `results/analysis/edit_effects.csv`
- insert_jazz: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 1.02, CLAP 0.99 (n=50) — `results/analysis/edit_effects.csv`
- insert_country: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 0.71, CLAP 0.77 (n=50) — `results/analysis/edit_effects.csv`
- fast_to_slow: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 0.70, CLAP 1.17 (n=14) — `results/analysis/edit_effects.csv`
- low_quality_to_studio: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT 0.99, CLAP 1.02 (n=35) — `results/analysis/edit_effects.csv`

### 2d. Spillover bundles

- Spillover table: 198 axis x group cells; 42 have a 95% interval that excludes 0 (about 10 expected by chance); 9 axes, 22 groups — `results/analysis/edit_spillover.csv`
- loud adds groove (timbre: groovy, punchy): +0.78 seed SD, raw change +1.54 [+0.64, +2.37], up in 13 of 16 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- loud adds drums (instrument: drums, percussion): +0.64 seed SD, raw change +1.21 [+0.13, +2.39], up in 11 of 16 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- loud adds male voice (vocals: male voice): +0.53 seed SD, raw change +1.13 [+0.22, +2.00], up in 11 of 16 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- loud adds happy (mood: happy, bright): +0.41 seed SD, raw change +0.85 [+0.21, +1.43], up in 12 of 16 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- sad removes drums (instrument: drums, percussion): -0.61 seed SD, raw change -1.00 [-1.86, -0.31], up in 4 of 15 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- sad removes groovy (timbre: groovy, punchy): -0.51 seed SD, raw change -0.98 [-1.59, -0.43], up in 2 of 15 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- acoustic to electric removes sad (mood: sad, emotional): -0.93 seed SD, raw change -1.45 [-1.95, -0.82], up in 1 of 17 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- acoustic to electric adds synth (instrument: synth, electronic sounds): +0.65 seed SD, raw change +1.24 [+0.52, +1.91], up in 13 of 17 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- guitar to piano removes acoustic (timbre: acoustic): -0.63 seed SD, raw change -1.50 [-2.33, -0.62], up in 6 of 28 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- guitar to piano removes bass (instrument: bass): -0.54 seed SD, raw change -0.94 [-1.48, -0.42], up in 9 of 28 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- guitar to piano removes electric (timbre: electric, distorted): -0.47 seed SD, raw change -1.06 [-2.07, -0.12], up in 11 of 28 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- guitar to piano removes latin (genre: latin, reggae, world): -0.46 seed SD, raw change -1.59 [-2.74, -0.55], up in 9 of 28 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- country adds acoustic (timbre: acoustic): +0.42 seed SD, raw change +0.89 [+0.37, +1.46], up in 31 of 50 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- country adds hip hop (genre: hip hop, r&b, funk): +0.35 seed SD, raw change +0.88 [+0.35, +1.40], up in 32 of 50 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- country adds latin (genre: latin, reggae, world): +0.28 seed SD, raw change +0.96 [+0.28, +1.69], up in 31 of 50 pairs; interval excludes 0 — `results/analysis/edit_spillover.csv`
- soft_to_loud: its own target probe on MERT, pooled 10/16, mean +0.91 [-0.06, +1.87] (interval includes 0) — `results/analysis/edit_effects.csv`
- happy_to_sad: its own target probe on MERT, pooled 10/15, mean +1.28 [-0.02, +2.55] (interval includes 0) — `results/analysis/edit_effects.csv`
- country inserted: country not rendered as country, MERT probe moved 27/50 (p=0.336) — `results/analysis/edit_effects.csv`

### 2e. Bundle origin: caption vs real-audio co-occurrence

- Spillover vs caption co-occurrence Spearman 0.32 (p=6.4e-06); vs real-audio co-occurrence 0.32 (p=7.6e-06); n=189 axis x group pairs — `results/analysis/bundle_origin.json`
- The two co-occurrence measures correlate 0.89 with each other — `results/analysis/bundle_origin.json`
- Sign agreement on the 36 pairs with a clear spillover: 0.81 (caption), 0.81 (audio); linear fit R2 0.15 — `results/analysis/bundle_origin.json`
- Exception check, country insert -> genre: hip hop, r&b, funk: spillover +0.35 SD, caption co-occurrence -0.07, audio co-occurrence -0.17 (target clips 317, source clips 4709) — `results/analysis/bundle_origin.csv`
- Exception check, country insert -> genre: latin, reggae, world: spillover +0.28 SD, caption co-occurrence +0.08, audio co-occurrence +0.41 (target clips 317, source clips 4709) — `results/analysis/bundle_origin.csv`
- Exception check, country insert -> timbre: acoustic: spillover +0.42 SD, caption co-occurrence +0.22, audio co-occurrence +0.71 (target clips 317, source clips 4709) — `results/analysis/bundle_origin.csv`

Caveat in the file: the 'audio' side uses the same probes (scripts/analysis/bundle_origin.py help text).


### 2f. Probe leakage (drums probe vs level)

- Drums probe score on real-clip level alone: 0.37 SD per SD of RMS; given percussion features 0.34 (n=782 real clips) — `results/analysis/probe_leakage.json`
- loud: drums probe change +0.64 SD; predicted from level alone +0.26 (41% of it); level change +2.33 dB; percussive share change +0.34 SD (up in 10 of 16 pairs); percussive onset rate +0.76 SD (n=16 pairs) — `results/analysis/probe_leakage.json`
- sad: drums probe change -0.61 SD; predicted from level alone +0.01 (-2% of it); level change +0.11 dB; percussive share change -0.44 SD (up in 4 of 15 pairs); percussive onset rate -0.35 SD (n=15 pairs) — `results/analysis/probe_leakage.json`
- voice swap: drums probe change -0.41 SD; predicted from level alone +0.11 (-26% of it); level change +0.76 dB; percussive share change -0.15 SD (up in 12 of 26 pairs); percussive onset rate -0.10 SD (n=26 pairs) — `results/analysis/probe_leakage.json`

### 2g. Recording-quality words

- Quality reader (`interpretable`, 5 features) CV AUC 0.715 on real audio (83 low quality vs 150 others); single-feature reader 0.714; all 110 features 0.763 — `results/analysis/dsp_detectors.csv`
- Swap low quality to studio, quality reader: registered 8/12, pooled 18/35 (pooled p=0.500) — `results/analysis/edit_effects.csv`
- Swap low quality to studio, MERT probe: registered 4/12, pooled 21/35 (pooled p=0.155) — `results/analysis/edit_effects.csv`
- Swap low quality to studio, CLAP probe: registered 9/12, pooled 17/35 (pooled p=0.632) — `results/analysis/edit_effects.csv`
- Removing the recording and setting sentences (29 dev captions), scored against the original caption: CLAP -0.012 [-0.052, +0.027]; MQAScore +0.015 [-0.024, +0.051]; KL-PaSST +0.04 [-0.20, +0.33] (n=29) — `results/analysis/ablate_dev_scores.json`
- Mean |dCLAP| on the same 29 ids: cleaning 0.083 vs a new seed 0.075 (computed, mean over seeds 1 and 2 vs seed 0 of rescale12; same recipe as `scripts/analysis/ablate_followed.py`) — `results/analysis/ablate_dev_scores.json, results/dev__rescale12*/per_prompt.csv, results/analysis/clean_dev.json`
- Power limit: the CLAP interval half-width is 0.039 (about +-0.04) — `results/analysis/ablate_dev_scores.json`
- Full-100 run with cleaned captions vs rescale12: prediction [-0.010, +0.010] MET, knowledge.md got -0.003 [-0.016, +0.008]; prediction [+0.000, +0.030] MET, knowledge.md got +0.004 [-0.007, +0.015]; prediction [+0.000, +0.080] MET, knowledge.md got +0.013 [-0.057, +0.089] — `results/analysis/knowledge.md`
- Cleaning drops 29 captions' sentences (groups: context: use or setting, recording: high quality, recording: low quality); median words 48 to 45 — `results/analysis/clean_dev.json`
- Caption words for `low quality`: MusicCaps 33.8% of captions, dev 38, test 3 captions (of 100 each); `hear` flag in file: yes — `results/analysis/caption_vocab_groups.csv`
- Caption words for `high quality`: MusicCaps 3.7% of captions, dev 3, test 14 captions (of 100 each); `hear` flag in file: partly — `results/analysis/caption_vocab_groups.csv`

### 2h. Chroma check

Features are computed on the HPSS harmonic part only (percussive part removed), so the plan's 'drums removed' means 'percussive part removed'.

- `tempo`, `harmony_chroma_entropy_mean`: AUC 0.862 (n=300); control-only CV AUC 0.811, with the feature 0.868, gain +0.057 — `results/analysis/chroma_check.csv`
- `loudness`, `harmonic_frame_entropy`: AUC 0.788 (n=300); control-only CV AUC 0.681, with the feature 0.768, gain +0.088 — `results/analysis/chroma_check.csv`
- `jazz vs country`, `harmonic_profile_entropy`: AUC 0.592 (n=200); control-only CV AUC 0.487, with the feature 0.557, gain +0.071 — `results/analysis/chroma_check.csv`
- `jazz vs country`, `harmony_chroma_entropy_mean`: AUC 0.454 (n=200); control-only CV AUC 0.487, with the feature 0.532, gain +0.045 — `results/analysis/chroma_check.csv`
- `electric vs acoustic`, `harmonic_profile_entropy`: AUC 0.576 (n=200); control-only CV AUC 0.491, with the feature 0.504, gain +0.013 — `results/analysis/chroma_check.csv`
- `piano vs guitar`, `harmony_chroma_entropy_mean`: AUC 0.495 (n=200); control-only CV AUC 0.512, with the feature 0.492, gain -0.020 — `results/analysis/chroma_check.csv`
- `happy vs sad`, `harmony_chroma_entropy_mean`: AUC 0.756 (n=200); control-only CV AUC 0.767, with the feature 0.791, gain +0.024 — `results/analysis/chroma_check.csv`
- Word-swap insert_jazz, chroma moved (pooled): harmonic_profile_entropy 37/50 (p=0.0005); harmonic_frame_entropy 22/50 (p=0.8389) — `results/analysis/edit_effects.csv`
- Word-swap insert_country, chroma moved (pooled): harmonic_profile_entropy 22/50 (p=0.8389); harmonic_frame_entropy 33/50 (p=0.0164) — `results/analysis/edit_effects.csv`
- Word-swap guitar_to_piano, chroma moved (pooled): harmonic_profile_entropy 10/28 (p=0.9564); harmonic_frame_entropy 19/28 (p=0.0436) — `results/analysis/edit_effects.csv`
- Word-swap fast_to_slow, chroma moved (pooled): harmonic_profile_entropy 7/14 (p=0.6047); harmonic_frame_entropy 9/14 (p=0.2120) — `results/analysis/edit_effects.csv`

### 2i. Specificity (target moves, unrelated groups move like seed noise)

`target_rise`, `source_drop`: mean change of the target and source group probe scores (probe score units of the file); `unrelated_mean_abs` and `unrelated_seed_abs`: mean absolute change of the other probe groups under the edit and under a new seed; ratio = `unrelated_edit_vs_seed`.

- guitar_to_piano (MERT, pooled n=28): target_rise 2.08, source_drop 2.98, unrelated mean |change| 1.87 vs seed 1.92 (ratio 0.98) — `results/analysis/edit_effects.csv`
- voice_male_to_female (MERT, pooled n=26): target_rise 1.45, source_drop 1.38, unrelated mean |change| 1.79 vs seed 1.82 (ratio 0.98) — `results/analysis/edit_effects.csv`
- acoustic_to_electric (MERT, pooled n=17): target_rise 0.90, source_drop 1.46, unrelated mean |change| 1.75 vs seed 1.77 (ratio 0.99) — `results/analysis/edit_effects.csv`
- insert_jazz (MERT, pooled n=50): target_rise 3.58, source_drop -1.16, unrelated mean |change| 1.74 vs seed 1.87 (ratio 0.93) — `results/analysis/edit_effects.csv`
- insert_country (MERT, pooled n=50): target_rise 1.12, source_drop -0.78, unrelated mean |change| 1.52 vs seed 1.87 (ratio 0.81) — `results/analysis/edit_effects.csv`
- soft_to_loud (MERT, pooled n=16): target_rise 0.47, source_drop 0.44, unrelated mean |change| 1.72 vs seed 2.01 (ratio 0.86) — `results/analysis/edit_effects.csv`
- happy_to_sad (MERT, pooled n=15): target_rise -0.17, source_drop 1.46, unrelated mean |change| 1.56 vs seed 1.89 (ratio 0.82) — `results/analysis/edit_effects.csv`
- fast_to_slow (MERT, pooled n=14): target_rise 0.73, source_drop 0.37, unrelated mean |change| 1.46 vs seed 1.86 (ratio 0.79) — `results/analysis/edit_effects.csv`
- low_quality_to_studio (MERT, pooled n=35): target_rise 1.03, source_drop 0.58, unrelated mean |change| 1.73 vs seed 1.92 (ratio 0.90) — `results/analysis/edit_effects.csv`

## 3. Positive control (ablate) and why CLAP, MQAScore and KL differ

- Dropping a sentence with a followed word (guitar, piano, voice, acoustic, electric), 29 captions, generated clip scored against the original caption: CLAP -0.086 [-0.132, -0.041] (mean |change| 0.126); MQAScore +0.004 [-0.042, +0.049]; KL-PaSST +0.32 [-0.08, +0.68] (mean |change| 0.81) — `results/analysis/ablate_dev_scores.json`
- Removed-group probe -1.09 [-1.49, -0.72] seed SD; 81% of 43 removed groups go down — `results/analysis/ablate_dev_scores.json`
- Seed change mean |dCLAP| on the ablated ids: 0.0638 — `results/analysis/ablate_dev_scores.json`
- Full 100-caption run `dev__rescale12_ablate` vs `dev__rescale12`: prediction [-0.015, -0.002] MISSED, knowledge.md got -0.025 [-0.041, -0.010]; prediction [-0.030, -0.003] MISSED, knowledge.md got +0.001 [-0.013, +0.015] — `results/analysis/knowledge.md`
- MQAScore per-tag, `kept instrument` tags: change +0.08 [+0.02, +0.16] (n=77 tags, mean before 0.33) — `results/analysis/ablate_mqa_tags.csv`
- MQAScore per-tag, `kept vocal` tags: change +0.11 [-0.12, +0.36] (n=8 tags, mean before 0.42) — `results/analysis/ablate_mqa_tags.csv`
- MQAScore per-tag, `removed instrument` tags: change -0.15 [-0.35, +0.04] (n=22 tags, mean before 0.40) — `results/analysis/ablate_mqa_tags.csv`
- MQAScore per-tag, `removed vocal` tags: change -0.30 [-0.45, -0.16] (n=18 tags, mean before 0.32) — `results/analysis/ablate_mqa_tags.csv`
- After removing the instrument sentence, caption `mqascore` +0.02 [-0.05, +0.09] (n=17 captions; `tag_share` 0.51) — `results/analysis/ablate_mqa_attributes.csv`
- After removing the instrument sentence, caption `instrument_mean` -0.02 [-0.14, +0.12] (n=17 captions; `tag_share` 0.51) — `results/analysis/ablate_mqa_attributes.csv`
- After removing the instrument sentence, caption `vocal_mean` +0.12 [-0.11, +0.39] (n=7 captions; `tag_share` 0.51) — `results/analysis/ablate_mqa_attributes.csv`
- After removing the vocal sentence, caption `mqascore` -0.01 [-0.05, +0.03] (n=15 captions; `tag_share` 0.129) — `results/analysis/ablate_mqa_attributes.csv`
- After removing the vocal sentence, caption `instrument_mean` +0.12 [+0.01, +0.25] (n=15 captions; `tag_share` 0.129) — `results/analysis/ablate_mqa_attributes.csv`
- After removing the vocal sentence, caption `vocal_mean` -0.27 [-0.41, -0.14] (n=14 captions; `tag_share` 0.129) — `results/analysis/ablate_mqa_attributes.csv`
- Tag count per caption: mean 7.8 on the 29 ablated captions (6.5 over all 100), so one removed tag is about 1 of 8 — `results/dev__rescale12/mqa_per_item.csv, results/analysis/ablate_dev.json`
- MQAScore driver `instrument`: 44% of tags, mean tag score 0.42, share of tags above 0.5 42%, score variance share across all clips 47%, across seeds of one caption 58% (300 clips: dev rescale12 seeds 0-2) — `results/analysis/mqa_drivers.csv`
- MQAScore driver `mood_theme`: 29% of tags, mean tag score 0.64, share of tags above 0.5 71%, score variance share across all clips 23%, across seeds of one caption 18% (300 clips: dev rescale12 seeds 0-2) — `results/analysis/mqa_drivers.csv`
- MQAScore driver `context`: 10% of tags, mean tag score 0.82, share of tags above 0.5 90%, score variance share across all clips 32%, across seeds of one caption 10% (300 clips: dev rescale12 seeds 0-2) — `results/analysis/mqa_drivers.csv`
- MQAScore driver `vocal`: 9% of tags, mean tag score 0.24, share of tags above 0.5 25%, score variance share across all clips -4%, across seeds of one caption 4% (300 clips: dev rescale12 seeds 0-2) — `results/analysis/mqa_drivers.csv`
- MQAScore driver `genre`: 8% of tags, mean tag score 0.52, share of tags above 0.5 56%, score variance share across all clips 3%, across seeds of one caption 10% (300 clips: dev rescale12 seeds 0-2) — `results/analysis/mqa_drivers.csv`
- MQAScore asks only genre, instrument, mood_theme, vocal and context tags; tempo and production-quality tags are not asked (`attribute` column lists exactly these five) — `results/analysis/mqa_drivers.csv`

## 4. Overnight runs, each vs its baseline

All paired differences are computed from per-item files (2000 resamples, seed 0, over prompt ids; MQAScore over the prompts with at least one tag). `pending` = the run has no per-item MQAScore file.

- **dev__rescale12_followed vs dev__rescale12** (followed words push (caption weight 12, extra term)): CLAP +0.002 [-0.010, +0.014] (n=100); KL-PaSST -0.093 [-0.160, -0.023] (n=100); MQAScore +0.001 [-0.015, +0.017] (n=98) — `results/dev__rescale12_followed/per_prompt.csv, results/dev__rescale12_followed/mqa_per_item.csv, results/dev__rescale12/per_prompt.csv, results/analysis/knowledge.md`
- verdict line (clap): prediction [+0.000, +0.015] MET, knowledge.md got +0.002 [-0.010, +0.014] — `results/analysis/knowledge.md`
- verdict line (kl): prediction [-0.050, +0.050] MISSED, knowledge.md got -0.093 [-0.164, -0.025] — `results/analysis/knowledge.md`
- verdict line (mqa): prediction [+0.000, +0.020] MET, knowledge.md got +0.001 [-0.016, +0.017] — `results/analysis/knowledge.md`
- **dev__rescale12_followed_sum12 vs dev__rescale12** (followed words push, weights summed to 12): CLAP -0.006 [-0.018, +0.007] (n=100); KL-PaSST +0.001 [-0.079, +0.084] (n=100); MQAScore +0.013 [-0.005, +0.031] (n=98) — `results/dev__rescale12_followed_sum12/per_prompt.csv, results/dev__rescale12_followed_sum12/mqa_per_item.csv, results/dev__rescale12/per_prompt.csv, results/analysis/knowledge.md`
- verdict line (clap): prediction [-0.005, +0.012] MISSED, knowledge.md got -0.006 [-0.018, +0.007] — `results/analysis/knowledge.md`
- verdict line (mqa): prediction [-0.005, +0.020] MET, knowledge.md got +0.013 [-0.005, +0.029] — `results/analysis/knowledge.md`
- **dev__rescale16 vs dev__rescale12** (caption weight 16, no change of words): CLAP +0.005 [-0.006, +0.016] (n=100); KL-PaSST -0.089 [-0.172, -0.005] (n=100); MQAScore -0.016 [-0.037, +0.005] (n=98) — `results/dev__rescale16/per_prompt.csv, results/dev__rescale16/mqa_per_item.csv, results/dev__rescale12/per_prompt.csv, results/analysis/knowledge.md`
- verdict line (clap): prediction [-0.010, +0.010] MET, knowledge.md got +0.005 [-0.005, +0.015] — `results/analysis/knowledge.md`
- **dev__rescale12_followed vs dev__rescale16** (followed push vs equal total weight): CLAP -0.003 [-0.016, +0.010] (n=100); KL-PaSST -0.004 [-0.098, +0.087] (n=100); MQAScore +0.016 [-0.005, +0.039] (n=98) — `results/dev__rescale12_followed/per_prompt.csv, results/dev__rescale12_followed/mqa_per_item.csv, results/dev__rescale16/per_prompt.csv, results/analysis/knowledge.md`
- verdict line (clap): prediction [+0.000, +0.012] MISSED, knowledge.md got -0.003 [-0.017, +0.010] — `results/analysis/knowledge.md`
- **dev__rescale12_loop2 vs dev__rescale12_best3_probe** (round 2 emphasis on the same seed vs round 1 pick): CLAP -0.001 [-0.008, +0.006] (n=100); KL-PaSST -0.024 [-0.071, +0.016] (n=100); MQAScore pending (no per-item file) — `results/dev__rescale12_loop2/per_prompt.csv, results/dev__rescale12_loop2/mqa_per_item.csv, results/dev__rescale12_best3_probe/per_prompt.csv, results/analysis/knowledge.md`
- verdict line (clap): prediction [-0.005, +0.010] MET, knowledge.md got -0.001 [-0.009, +0.006] — `results/analysis/knowledge.md`
- verdict line (kl): prediction [-0.050, +0.030] MET, knowledge.md got -0.024 [-0.071, +0.018] — `results/analysis/knowledge.md`
- dev__rescale12_followed vs dev__cfg7: dFD-VGG +0.45 [-0.01, +0.76] (paired FD is only available vs cfg7) — `results/analysis/paired_fd_dev.json`
- dev__rescale12_followed_sum12 vs dev__cfg7: dFD-VGG +0.27 [-0.08, +0.60] (paired FD is only available vs cfg7) — `results/analysis/paired_fd_dev.json`
- dev__rescale16 vs dev__cfg7: dFD-VGG +0.17 [-0.27, +0.46] (paired FD is only available vs cfg7) — `results/analysis/paired_fd_dev.json`
- dev__rescale12_loop2 vs dev__cfg7: dFD-VGG +0.28 [-0.21, +0.65] (paired FD is only available vs cfg7) — `results/analysis/paired_fd_dev.json`
- Round 2 changed 89 of 100 captions (22 probe groups checked); emphasis not added for token limit in 21; round-2 clips kept (score above round 1) for 27 of 89 changed prompts = 0.303 (knowledge.md `kept_fraction` 0.303); 27 of all 100 prompts = 0.270 — `results/analysis/loop_dev.json, results/dev__rescale12_loop2/loop_selection.csv, results/analysis/knowledge.md`
- Most-missing groups in round 1 (count of captions): instrument: drums, percussion 34; instrument: bass 24; instrument: piano, keys 21; mood: sad, emotional 19; mood: happy, bright 19 — `results/analysis/loop_dev.json`
- Followed-word push: 95 of 100 dev captions have a followed-word phrase; median 5 phrases per caption — `results/analysis/followed_dev.json`

**Bundle cut** (negation formula, caption weight kept at 12; `bundle` arm = extra term away from (soft_to_loud) or toward (happy_to_sad) the bundle text, `neutral` arm = meaning-free extra text at the same weight). Effects in seed-SD units against the plain edited clip; `edit_effect_sd` = the edit's own effect in the same units.

- MERT soft_to_loud (n=16 pairs, extra sign -1): bundle / target minus source: +0.02 [-0.24, +0.28] (edit effect +0.28); bundle / instrument: drums, percussion: -1.11 [-1.61, -0.63] (edit effect +0.64); bundle / timbre: groovy, punchy: -1.11 [-1.60, -0.61] (edit effect +0.78); neutral / target minus source: -0.19 [-0.38, -0.02] (edit effect +0.28); neutral / instrument: drums, percussion: -0.05 [-0.34, +0.27] (edit effect +0.64); neutral / timbre: groovy, punchy: -0.09 [-0.42, +0.24] (edit effect +0.78) — `results/analysis/bundle_guidance.csv`
- MERT happy_to_sad (n=15 pairs, extra sign +1): bundle / target minus source: -0.38 [-0.82, +0.09] (edit effect +0.48); bundle / instrument: drums, percussion: +1.71 [+1.05, +2.42] (edit effect -0.61); bundle / timbre: groovy, punchy: +0.91 [+0.60, +1.25] (edit effect -0.51); neutral / target minus source: -0.14 [-0.49, +0.28] (edit effect +0.48); neutral / instrument: drums, percussion: +0.22 [-0.19, +0.63] (edit effect -0.61); neutral / timbre: groovy, punchy: +0.12 [-0.10, +0.34] (edit effect -0.51) — `results/analysis/bundle_guidance.csv`
- CLAP90 soft_to_loud (n=16 pairs, extra sign -1): bundle / target minus source: -0.08 [-0.51, +0.39] (edit effect +1.27); bundle / instrument: drums, percussion: -0.98 [-1.50, -0.43] (edit effect +0.57); bundle / timbre: groovy, punchy: -0.82 [-1.36, -0.32] (edit effect +0.62); neutral / target minus source: -0.03 [-0.36, +0.32] (edit effect +1.27); neutral / instrument: drums, percussion: -0.09 [-0.32, +0.14] (edit effect +0.57); neutral / timbre: groovy, punchy: -0.01 [-0.19, +0.18] (edit effect +0.62) — `results/analysis/bundle_guidance.csv`
- CLAP90 happy_to_sad (n=15 pairs, extra sign +1): bundle / target minus source: -0.66 [-0.95, -0.39] (edit effect +0.52); bundle / instrument: drums, percussion: +0.78 [+0.38, +1.21] (edit effect -0.38); bundle / timbre: groovy, punchy: +0.63 [+0.20, +1.08] (edit effect -0.19); neutral / target minus source: +0.01 [-0.27, +0.27] (edit effect +0.52); neutral / instrument: drums, percussion: +0.14 [-0.11, +0.41] (edit effect -0.38); neutral / timbre: groovy, punchy: +0.04 [-0.26, +0.34] (edit effect -0.19) — `results/analysis/bundle_guidance.csv`
- Reading, MERT soft_to_loud: bundle arm moves the bundle probes clearly in 2 of 2 groups, neutral arm in 0; target-minus-source (the word's own attribute) in the bundle arm includes 0, in the neutral arm excludes 0 — `results/analysis/bundle_guidance.csv`
- Reading, MERT happy_to_sad: bundle arm moves the bundle probes clearly in 2 of 2 groups, neutral arm in 0; target-minus-source (the word's own attribute) in the bundle arm includes 0, in the neutral arm includes 0 — `results/analysis/bundle_guidance.csv`
- Reading, CLAP90 soft_to_loud: bundle arm moves the bundle probes clearly in 2 of 2 groups, neutral arm in 0; target-minus-source (the word's own attribute) in the bundle arm includes 0, in the neutral arm includes 0 — `results/analysis/bundle_guidance.csv`
- Reading, CLAP90 happy_to_sad: bundle arm moves the bundle probes clearly in 2 of 2 groups, neutral arm in 0; target-minus-source (the word's own attribute) in the bundle arm excludes 0, in the neutral arm includes 0 — `results/analysis/bundle_guidance.csv`

## 5. Rankers and best-of-N

Each ranker's best-of-3 vs the seed-0 run of the same setting (rescale12): paired change in CLAP (90.14 scorer), KL-PaSST and MQAScore; `same pick` = share of prompts where two rankers pick the same seed (chance 1/3).

- dev `best3_c8998`: dCLAP +0.015 [+0.003, +0.028] (n=100); dKL -0.124 [-0.218, -0.038] (n=100); dMQAScore +0.014 [-0.003, +0.030] (n=98); same pick as c8998 1.00, as probe 0.31, as MQAScore 0.40 — `results/analysis/ranker_compare.csv`
- dev `best3_probe`: dCLAP -0.002 [-0.015, +0.011] (n=100); dKL -0.040 [-0.125, +0.038] (n=100); dMQAScore +0.006 [-0.010, +0.022] (n=98); same pick as c8998 0.31, as probe 1.00, as MQAScore 0.32 — `results/analysis/ranker_compare.csv`
- dev `best3_mqa`: dCLAP +0.003 [-0.010, +0.016] (n=100); dKL -0.041 [-0.124, +0.043] (n=100); dMQAScore +0.057 [+0.043, +0.073] (n=98); same pick as c8998 0.40, as probe 0.32, as MQAScore 1.00 — `results/analysis/ranker_compare.csv`
- test `best3_c8998`: dCLAP +0.029 [+0.017, +0.042] (n=100); dKL -0.046 [-0.113, +0.031] (n=100); dMQAScore +0.014 [-0.001, +0.029] (n=100); same pick as c8998 1.00, as probe 0.35, as MQAScore 0.43 — `results/analysis/ranker_compare.csv`
- test `best3_probe`: dCLAP +0.010 [-0.001, +0.022] (n=100); dKL -0.053 [-0.113, +0.002] (n=100); dMQAScore +0.010 [-0.005, +0.026] (n=100); same pick as c8998 0.35, as probe 1.00, as MQAScore 0.41 — `results/analysis/ranker_compare.csv`
- test `best3_mqa`: dCLAP +0.011 [-0.001, +0.023] (n=100); dKL +0.015 [-0.049, +0.091] (n=100); dMQAScore +0.053 [+0.040, +0.067] (n=100); same pick as c8998 0.43, as probe 0.41, as MQAScore 1.00 — `results/analysis/ranker_compare.csv`
- Between-ranker agreement on the same pick: 0.31 to 0.43 over all pairs and both sets (chance 0.33) — `results/analysis/ranker_compare.csv`

Within-caption rank correlations (scores centred per caption over 3 seeds; Spearman; dev n=98 captions = 294 clips, test n=100 = 300 clips; |rho| below about 0.12 is noise):

- dev rank agreement: clap8998 vs eval_clap +0.47; clap8998 vs mqa +0.19; mqa vs eval_clap +0.07; probe vs clap8998 -0.05; probe vs mqa -0.03; probe vs eval_clap +0.05 (`eval_clap` = the 90.14 scoring CLAP, `clap8998` = the 89.98 ranker CLAP, `probe` = probe ranker) — `results/analysis/ranker_explain.csv`
- dev `mqa` ranker, strongest feature links: harmony_chroma_entropy_mean +0.21; rms_db +0.15; percussive_share +0.13 — `results/analysis/ranker_explain.csv`
- dev `eval_clap` ranker, strongest feature links: centroid_hz +0.18; rms_db +0.14; flatness -0.09 — `results/analysis/ranker_explain.csv`
- dev `probe` ranker, strongest feature links: crest_factor_db -0.14; harmony_chroma_entropy_mean +0.10; rms_db +0.08 — `results/analysis/ranker_explain.csv`
- dev `clap8998` ranker, strongest feature links: centroid_hz +0.11; harmony_chroma_entropy_mean +0.09; percussive_share +0.09 — `results/analysis/ranker_explain.csv`
- test rank agreement: clap8998 vs eval_clap +0.50; clap8998 vs mqa +0.24; mqa vs eval_clap +0.16; probe vs clap8998 +0.18; probe vs mqa +0.17; probe vs eval_clap +0.20 (`eval_clap` = the 90.14 scoring CLAP, `clap8998` = the 89.98 ranker CLAP, `probe` = probe ranker) — `results/analysis/ranker_explain.csv`
- test `mqa` ranker, strongest feature links: rms_db +0.07; flatness -0.05; centroid_hz +0.04 — `results/analysis/ranker_explain.csv`
- test `eval_clap` ranker, strongest feature links: flatness -0.23; centroid_hz -0.09; harmony_chroma_entropy_mean +0.09 — `results/analysis/ranker_explain.csv`
- test `probe` ranker, strongest feature links: flatness -0.09; percussive_share +0.08; rms_db -0.05 — `results/analysis/ranker_explain.csv`
- test `clap8998` ranker, strongest feature links: flatness -0.11; share_above_12khz +0.04; centroid_hz +0.04 — `results/analysis/ranker_explain.csv`
- Largest |rho| of any judge with any acoustic feature (both sets): 0.23 — `results/analysis/ranker_explain.csv`

Best-of-3 vs the mean of the 3 seeds per prompt (a random pick), paired; `dclap9014` = scoring checkpoint, `dclap8998` = other checkpoint:

- dev cfg7, ranker same-checkpoint (90.14): dCLAP 90.14 +0.056 [+0.050, +0.062]; dCLAP 89.98 +0.025 [+0.016, +0.033]; dKL-PaSST -0.08 [-0.14, -0.01]; picks per seed index 30/37/33 — `results/analysis/best_of_n_vs_seed_mean.csv`
- dev cfg7, ranker cross-checkpoint (89.98): dCLAP 90.14 +0.036 [+0.027, +0.044]; dCLAP 89.98 +0.048 [+0.042, +0.054]; dKL-PaSST -0.04 [-0.11, +0.03]; picks per seed index 34/33/33 — `results/analysis/best_of_n_vs_seed_mean.csv`
- dev rescale12, ranker same-checkpoint (90.14): dCLAP 90.14 +0.047 [+0.041, +0.053]; dCLAP 89.98 +0.025 [+0.017, +0.034]; dKL-PaSST -0.10 [-0.16, -0.05]; picks per seed index 39/40/21 — `results/analysis/best_of_n_vs_seed_mean.csv`
- dev rescale12, ranker cross-checkpoint (89.98): dCLAP 90.14 +0.020 [+0.010, +0.029]; dCLAP 89.98 +0.047 [+0.042, +0.053]; dKL-PaSST -0.09 [-0.15, -0.03]; picks per seed index 39/37/24 — `results/analysis/best_of_n_vs_seed_mean.csv`
- test rescale12, ranker same-checkpoint (90.14): dCLAP 90.14 +0.046 [+0.040, +0.051]; dCLAP 89.98 +0.018 [+0.011, +0.025]; dKL-PaSST +0.01 [-0.04, +0.06]; picks per seed index 36/32/32 — `results/analysis/best_of_n_vs_seed_mean.csv`
- test rescale12, ranker cross-checkpoint (89.98): dCLAP 90.14 +0.028 [+0.021, +0.036]; dCLAP 89.98 +0.038 [+0.034, +0.043]; dKL-PaSST -0.02 [-0.08, +0.03]; picks per seed index 26/41/33 — `results/analysis/best_of_n_vs_seed_mean.csv`
- test cfg7, ranker same-checkpoint (90.14): dCLAP 90.14 +0.049 [+0.043, +0.055]; dCLAP 89.98 +0.028 [+0.021, +0.035]; dKL-PaSST -0.04 [-0.11, +0.03]; picks per seed index 32/40/28 — `results/analysis/best_of_n_vs_seed_mean.csv`
- test cfg7, ranker cross-checkpoint (89.98): dCLAP 90.14 +0.033 [+0.025, +0.041]; dCLAP 89.98 +0.044 [+0.039, +0.049]; dKL-PaSST -0.05 [-0.10, +0.03]; picks per seed index 29/39/32 — `results/analysis/best_of_n_vs_seed_mean.csv`

CLAP gain split (`down` = own-caption cosine change, `dother` = change of mean cosine to the other 99 captions, `dmargin` = own minus other; definitions in `scripts/analysis/review_checks/clap_margin.py`):

- dev rescale12 vs cfg7: down +0.028 [+0.017, +0.039]; dother +0.010 [+0.004, +0.016]; dmargin +0.017 [+0.006, +0.028] — `results/analysis/clap_margin.csv`
- dev rescale12_seed1 vs cfg7_seed1: down +0.032 [+0.020, +0.043]; dother +0.012 [+0.006, +0.018]; dmargin +0.020 [+0.010, +0.029] — `results/analysis/clap_margin.csv`
- dev rescale12_seed2 vs cfg7_seed2: down +0.024 [+0.007, +0.039]; dother +0.014 [+0.007, +0.021]; dmargin +0.009 [-0.005, +0.023] — `results/analysis/clap_margin.csv`
- dev rescale12 (3 seed pairs) vs cfg7 (same seeds): down +0.028 [+0.020, +0.035]; dother +0.012 [+0.009, +0.016]; dmargin +0.015 [+0.009, +0.021] — `results/analysis/clap_margin.csv`
- dev cfg12 vs cfg7: down +0.015 [+0.001, +0.029]; dother +0.003 [-0.004, +0.010]; dmargin +0.012 [-0.000, +0.023] — `results/analysis/clap_margin.csv`
- dev cfg3 vs cfg7: down -0.029 [-0.045, -0.013]; dother -0.003 [-0.010, +0.004]; dmargin -0.026 [-0.041, -0.013] — `results/analysis/clap_margin.csv`
- dev apg12 vs cfg7: down +0.011 [+0.002, +0.020]; dother +0.006 [+0.002, +0.010]; dmargin +0.005 [-0.003, +0.012] — `results/analysis/clap_margin.csv`
- dev tags vs cfg7: down -0.051 [-0.074, -0.028]; dother -0.017 [-0.027, -0.007]; dmargin -0.035 [-0.054, -0.015] — `results/analysis/clap_margin.csv`
- dev interval_late vs cfg7: down -0.027 [-0.034, -0.019]; dother -0.005 [-0.010, -0.001]; dmargin -0.022 [-0.028, -0.015] — `results/analysis/clap_margin.csv`
- dev rescale12_best3_c8998 vs rescale12: down +0.015 [+0.003, +0.027]; dother +0.012 [+0.006, +0.018]; dmargin +0.003 [-0.009, +0.015] — `results/analysis/clap_margin.csv`
- dev rescale12_best3 vs rescale12: down +0.042 [+0.032, +0.053]; dother +0.012 [+0.006, +0.018]; dmargin +0.030 [+0.020, +0.041] — `results/analysis/clap_margin.csv`
- dev cfg7_best3_c8998 vs cfg7: down +0.031 [+0.019, +0.042]; dother +0.012 [+0.004, +0.019]; dmargin +0.019 [+0.009, +0.029] — `results/analysis/clap_margin.csv`
- dev cfg7_best3 vs cfg7: down +0.051 [+0.041, +0.062]; dother +0.015 [+0.007, +0.022]; dmargin +0.037 [+0.025, +0.049] — `results/analysis/clap_margin.csv`
- dev rescale12_best3_c8998 vs cfg7: down +0.042 [+0.028, +0.057]; dother +0.022 [+0.015, +0.029]; dmargin +0.020 [+0.006, +0.034] — `results/analysis/clap_margin.csv`
- dev real vs cfg7: down +0.004 [-0.023, +0.034]; dother -0.021 [-0.034, -0.008]; dmargin +0.024 [-0.001, +0.054] — `results/analysis/clap_margin.csv`
- test rescale12 vs cfg7: down +0.010 [-0.000, +0.021]; dother +0.000 [-0.005, +0.006]; dmargin +0.010 [+0.001, +0.019] — `results/analysis/clap_margin.csv`
- test rescale12_seed1 vs cfg7_seed1: down +0.007 [-0.004, +0.018]; dother +0.003 [-0.003, +0.009]; dmargin +0.004 [-0.004, +0.013] — `results/analysis/clap_margin.csv`
- test rescale12_seed2 vs cfg7_seed2: down +0.014 [+0.005, +0.023]; dother +0.009 [+0.003, +0.016]; dmargin +0.005 [-0.003, +0.013] — `results/analysis/clap_margin.csv`
- test rescale12 (3 seed pairs) vs cfg7 (same seeds): down +0.011 [+0.005, +0.016]; dother +0.004 [+0.001, +0.008]; dmargin +0.006 [+0.001, +0.011] — `results/analysis/clap_margin.csv`
- test rescale12_best3_c8998 vs rescale12: down +0.029 [+0.017, +0.042]; dother +0.016 [+0.009, +0.025]; dmargin +0.013 [+0.003, +0.023] — `results/analysis/clap_margin.csv`
- test rescale12_best3_c8998 vs cfg7: down +0.040 [+0.027, +0.051]; dother +0.016 [+0.008, +0.025]; dmargin +0.023 [+0.013, +0.033] — `results/analysis/clap_margin.csv`
- Submission picks on test (seed 0 / 1 / 2): 26 / 41 / 33 of 100; ranker score `clap8998_cosine` — `results/test__rescale12_best3_c8998/selection.json`

## 6. Prediction tally

- Predictions with a result: 47; **met 35, missed 12**; pending 12 — `results/analysis/knowledge.md`
- MISSED `dev__cfg7_best3` vs `dev__cfg7`, mqa predicted [+0.010, +0.040] (written 11-15 11:00): got -0.007 [-0.036, +0.019] — `results/analysis/knowledge.md`
- MISSED `dev__rescale7` vs `dev__cfg7`, clap predicted [+0.000, +0.015] (written 11-15 11:58): got -0.003 [-0.012, +0.004] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_phi1` vs `dev__rescale12`, clap predicted [-0.015, +0.005] (written 11-15 11:58): got +0.006 [-0.004, +0.015] — `results/analysis/knowledge.md`
- MISSED `dev__cfg7` vs `dev__real`, vocal_db predicted [-50.000, -15.000] (written 11-15 16:10): got -12.742 [-18.419, -7.100] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_degraded` vs `dev__rescale12`, kl predicted [-0.200, +0.000] (written 11-15 18:10): got +0.075 [+0.032, +0.118] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_best3_probe` vs `dev__rescale12`, clap predicted [+0.000, +0.015] (written 11-15 20:04): got -0.002 [-0.015, +0.011] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_loop2` vs `dev__rescale12_best3_probe`, kept_fraction predicted [+0.350, +0.650] (written 11-15 23:31): got +0.303 [+0.303, +0.303] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_ablate` vs `dev__rescale12`, clap predicted [-0.015, -0.002] (written 11-16 00:40): got -0.025 [-0.041, -0.010] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_ablate` vs `dev__rescale12`, mqa predicted [-0.030, -0.003] (written 11-16 00:40): got +0.001 [-0.013, +0.015] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_followed` vs `dev__rescale12`, kl predicted [-0.050, +0.050] (written 11-16 01:02): got -0.093 [-0.164, -0.025] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_followed` vs `dev__rescale16`, clap predicted [+0.000, +0.012] (written 11-16 01:04): got -0.003 [-0.017, +0.010] — `results/analysis/knowledge.md`
- MISSED `dev__rescale12_followed_sum12` vs `dev__rescale12`, clap predicted [-0.005, +0.012] (written 11-16 01:05): got -0.006 [-0.018, +0.007] — `results/analysis/knowledge.md`

## Mismatches

Plan value vs file value, plus plan items whose file does not exist yet. None of these changes a conclusion.

- Cleaning (recording and setting sentences removed, 29 captions) CLAP interval: plan `-0.012 [-0.054, +0.026]`; file `-0.012 [-0.052, +0.027]`; `results/analysis/ablate_dev_scores.json` (key `clean recording words`).
- Cleaning MQAScore interval: plan `+0.015 [-0.022, +0.052]`; file `+0.015 [-0.024, +0.051]`; `results/analysis/ablate_dev_scores.json`.
- Ranker agreement 'MQAScore and CLAP agree only weakly (0.19 / 0.24)': those are MQAScore vs the 89.98 ranker CLAP (`clap8998`). Against the 90.14 scoring CLAP (`eval_clap`) the values are 0.07 dev and 0.16 test; `results/analysis/ranker_explain.csv`. Say which CLAP on the slide.
- Loop round 2 `kept_fraction`: knowledge.md 0.303 is 27 of the 89 changed captions; 27 of all 100 prompts is 0.270; `results/dev__rescale12_loop2/loop_selection.csv`, `results/analysis/loop_dev.json`.
- Slide 8 cites `results/dev__rescale12_best5_*`: no such result folders exist (best-of-5 runs are still pending in knowledge.md); `dev__rescale12_followed_sentence` is also missing. No numbers can be given for them.
- Test paired FD-VGG: `paired_fd_dev.json` covers dev only, so the test rows have no paired dFD-VGG.
- dev__rescale12_best3_probe has no `mqa_per_item.csv` and no `metrics.json`, so loop2 vs best3_probe has no MQAScore and `findings_table.csv` shows its CLAP as 0.000 (pending); the CLAP and KL differences above come from `per_prompt.csv`.
- Caveat, 'followed' axes: with Holm over the 12 registered pairs only acoustic_to_electric on MERT is significant (Holm p=0.025). Guitar to piano, voice and jazz are followed on the pooled counts (extension pairs added, no Holm), so label them 'pooled'. Probe-based mood and dynamics results are weak on MERT (happy_to_sad pooled interval includes 0; soft_to_loud interval includes 0) but soft_to_loud is clear on CLAP (15/16).
- Caveat, plan wording 'drums removed' for the chroma check: the file removes the percussive part by HPSS (harmonic part only), not by a drum stem; `scripts/analysis/chroma_check.py`.
