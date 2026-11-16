# HW3 extras review (11-16 13:42)

Extras = everything beyond the required CFG baseline and metrics: the caption-word study, extra methods, extra judges, and the overnight word-level methods. Intervals are 95% paired bootstraps over 100 prompts, no multiplicity correction unless stated.

## 1. What we did beyond the baseline, ranked by insight

1. **Caption-word study (the lead).**
   - **Method:** swap one word in a caption with seed and everything else fixed, then read the change with MERT and CLAP probes and DSP detectors.
   - **Finding:** the model follows instrument, voice, acoustic/electric and jazz words. It does not follow recording-quality or "country" words, and tempo words at best weakly.
   - **Controls:** a null edit reproduces the clip bit-identically, and meaningless edits move a clip 0.68-0.77 of a seed change.
2. **Spillover bundles and the bundle cut.**
   - **Spillover:** "loud" brings drums and groove; "sad" strips them.
   - **Bundle cut:** cutting the loud bundle in guidance removes drums -1.11 SD while the loud probe stays (+0.02). For "sad", the cut restores drums, but the CLAP sad score falls -0.66 SD, so for sad the rhythm loss is part of the meaning.
   - **Origin:** the bundles track caption and real-audio co-occurrence equally (Spearman 0.32 each).
3. **Why the metrics disagree.**
   - **Removing a followed word:** CLAP falls -0.086, caption-mean MQAScore +0.004, KL +0.32 with an interval that includes 0. Per tag, MQAScore does see the removed voice (-0.30).
   - **Rankers:** the three rankers pick the same seed at chance rate (31-43% vs 33%).
   - **Degraded audio:** KL gets worse (+0.075), against the prediction.
4. **CLAP margin split.** Of the rescale gain, +0.015 of +0.028 is specific to the caption. The submission's extra best-of-3 CLAP is mostly generic: margin +0.003 [-0.009, +0.015].
5. **Methods beyond CFG:**
   - **Rescale:** the one honest gain (dev +0.028 [+0.020, +0.035] and KL -0.119 over 3 seed pairs; test +0.010).
   - **Others:** APG with the paper's momentum and norm cap, the guidance interval, tag prompts, best-of-3 with a second CLAP checkpoint, an MQAScore ranker and a probe ranker.
6. **MQAScore on one 16 GB GPU:** bf16 with CPU offload; matched 0.700 vs mismatched 0.359; reruns are bit-identical.
7. **Prediction ledger:** 35 met, 12 missed, 12 pending. Every miss changed a claim.

## 2. Why the last runs (overnight 11-16) gave nothing

All are on dev and paired against their baseline. Re-derived from per-clip files today.

| run | vs | dCLAP | dKL | dMQAScore |
|---|---|---|---|---|
| followed-word guidance on top of 12 | rescale 12 | +0.002 [-0.010, +0.014] | -0.093 [-0.166, -0.021] | +0.001 [-0.015, +0.017] |
| scale-16 control | rescale 12 | +0.005 [-0.006, +0.016] | -0.089 [-0.175, -0.002] | -0.016 [-0.039, +0.005] |
| followed words | scale 16 | -0.003 [-0.016, +0.010] | -0.004 [-0.098, +0.091] | +0.016 [-0.004, +0.042] |
| followed words, same total weight 12 | rescale 12 | -0.006 [-0.018, +0.007] | +0.001 [-0.083, +0.083] | +0.013 [-0.005, +0.031] |
| probe loop round 2 | round 1 | -0.001 [-0.009, +0.006] | -0.024 [-0.069, +0.018] | not scored |
| caption cleaning | rescale 12 | -0.003 [-0.016, +0.008] | +0.013 [-0.060, +0.093] | +0.004 [-0.007, +0.016] |

Why:

- **Followed words** are the words the model already follows, so pushing them harder is just more guidance weight. The scale-16 control gets the same KL gain (-0.089 vs -0.093) without any word choice.
- **The words that matter on test** (texture, mood, recording) are the ones the model does not follow. A method that re-weights caption words cannot reach them.
- **Round 2 of the probe loop** kept only 27 of 89 changed captions (0.303), and adding emphasis sentences re-rolls the clip like a seed.
- **Cleaning** removes words the model ignores, so nothing changes but the random roll.
- **All three judges** need a stronger effect than 100 prompts can show: the CLAP interval half-width is about 0.012.

Under-reported by the deck (both include 0, so they are hints only):

- **Scale 16 alone trends down on MQAScore** (-0.016), while followed words at the same weight do not (+0.016 vs scale 16). "Nothing beyond weight" holds for CLAP and KL. On MQAScore it is not shown either way.
- **Same total weight:** the trade is CLAP -0.006 vs MQAScore +0.013.

## 3. Numbers re-checked

- **Re-derivation:** 21 paired comparisons (dev and test, CLAP, KL, MQAScore) recomputed from `results/<run>/per_prompt.csv` and `mqa_per_item.csv` with a fresh bootstrap (10k resamples). All match `knowledge.md` and `findings_table.md` within resampling noise (±0.001 on the bounds).
- **Pooled test rescale:** +0.010 [+0.004, +0.016] when averaged per prompt over 3 seeds; the deck has +0.011 [+0.005, +0.016] from `clap_margin.csv`. Same result, different pooling. Use one and name it.
- **New and not in the deck:** test MQAScore, submission vs cfg 7, is **+0.021 [+0.003, +0.039]**. It is the only test gain on a judge other than CLAP whose interval excludes 0. The ranker was CLAP 89.98, so this is not circular for MQAScore. No multiplicity correction. Rescale seed 0 alone gives +0.007 [-0.009, +0.024].
- **Ablation KL:** over all 100 captions it is +0.094 [-0.028, +0.211]. The `metrics` slide's +0.32 is the 29-caption subset; both are fine, but label the subset.

## 4. What was compromised (ranked by how much it weakens a claim)

1. **Multiplicity in the lead study.** With Holm over the 12 registered swap pairs, only acoustic-to-electric (MERT, p 0.025) survives. Guitar-to-piano, voice and jazz count as followed only on pooled counts after extension pairs were added. Label them "pooled, not corrected".
2. **Detector vs model.** "Followed" means a probe or detector moved. A weak detector and an ignored word look alike (the quality reader is near chance), and the CLAP probe shares an encoder with the CLAP metric.
3. **Rules set after dev was seen.** Both "improve" definitions (11-15 11:50 and 16:15) came after dev results. Scale 18 and phi 1 look at least as good as the pick; rescale 12 won because only it had 3 seeds and a best-of-3 pool.
4. **Test is not independent of CLAP.** Test references were retrieved with the scoring checkpoint (90.14). They are hub-heavy: 59 distinct clips for 100 captions, and hub references make KL easier (0.82 vs 1.04). Best-of-3 ranks per test item with a near-copy of the metric (picks agree 53% vs 33% chance).
5. **Bandwidth gap: measured, and it does not bias the scores (11-16 14:04).** Generated audio has about +18 to +24 dB more energy above 16 kHz than the real clips (`ltas_dev.csv`). Low-passing real and generated audio and rescoring changes nothing that matters (`results/analysis/extras_mel_hw1.md`, `lowpass_rescore.csv`):
   - **cfg7 vs real:** KL-PaSST is 1.506 unfiltered and 1.504 at 16 kHz; FD-VGG is 3.90 both times.
   - **Method gains:** rescale12 dCLAP is +0.028 [+0.017, +0.039] before and +0.027 [+0.016, +0.038] at 16 kHz; dKL is -0.084 both times.
   - **Why:** the scoring encoders resample first (`external/av-benchmark/av_bench/extract.py`). VGGish runs at 16 kHz (sees up to 8 kHz) and PaSST at 32 kHz (up to 16 kHz), so the high band never reaches them. CLAP runs at 48 kHz, but its scores did not move either.
   - **Consequence:** the gap to real on FD and KL is about music content below 8-16 kHz. At an 8 kHz cut, PaSST loses its upper band (KL 1.597) and the cfg12 CLAP gain drops to +0.008 [-0.004, +0.018].
6. **Dev does not look like test.** Dev averages 48.1 words, 41% with quality words and 50 vocal tags; test averages 28.4 words, 1% and 1. Tuning on dev tuned for a different caption style, which likely explains much of dev +0.028 shrinking to test +0.010.
7. **Methods not run as the papers set them:**
   - **APG:** the projection, cap and eta-1 ablations were never run, so the APG negative is not decomposed.
   - **Rescale:** the std is taken over all 1,024 latent frames, but only 216 carry audio, which dilutes it. The valid-frame fix changed KL -0.055 [-0.115, +0.007].
   - **Guidance interval:** tried at 3 settings, not the paper's grid.
   - **Other pending or left-out runs:** best-of-5, sentence-form followed words, composable prompts, CFG++ and negative prompts were never run.
8. **Small samples.**
   - **Prompts:** 100 per split. FD on 100 clips has a real-vs-real floor of 1.9 (FD-VGG), so FD differences of ±0.3 are noise; paired FD-VGG exists for dev only.
   - **Listening:** one listener (the author), 16 pairs per session, side bias 20 of 29.
9. **MQAScore unverified.** The repo ships no reference scores, so it is checked only for determinism and matched vs mismatched. Whether the judge was trained on MusicCaps is not checked.
10. **Reproducibility.**
    - **Picks:** 41 of 100 test picks have a top-2 CLAP gap under 0.022, and CPU vs GPU TF32 rescoring differs by up to 0.022.
    - **Code versions:** most runs record a `+dirty` code version, so the commit alone does not pin the code. The null-edit and determinism checks cover the sampler path.

## 5. Experiments stopped (11-16 14:27, user: gains are small, lock down for the report)

- **Samples page:** done, at `pages.liuyu.dev/hw3-samples/`.
- **Extra analyses:** done (low-pass rescore, mel texture, HW1 judges).
- **Step-count sweep:** only rescale 12 at 25 steps ran. Against 100 steps (dev, paired): CLAP -0.013 [-0.023, -0.002], KL +0.077 [-0.011, +0.164], MQAScore -0.003 [-0.022, +0.016] (98 items). 25 steps costs CLAP; the other metrics do not separate.
- **Not run:** rescale 12 and cfg 7 at 50 steps (stopped after 3 clips, discarded), the 11 s condition cut to 10 s, the fine-tune.

## 6. Fine-tune estimate (for after 17:00)

The task is a LoRA fine-tune of the DiT on MusicCaps clips. It must exclude the 100 dev clips and all retrieved test references (5,367 local clips, so about 5,200 remain). The times below are derived from measured generation speed, not from a training benchmark:

- **Generation (measured):** 16.4 s per clip at 100 steps with CFG, which is about 0.16 s per batched forward.
- **Training step (guess):** about 4x a forward with gradient checkpointing, so about 0.3 s per sample. 2 epochs is about 10k samples, about 55 min.
- **Latents (guess):** VAE encoding of about 5,200 clips takes about 10-25 min.
- **Code (guess):** install `peft`, write the loop on the v-objective, and smoke-test: 45-60 min.
- **Evaluation (measured):** dev regeneration 28 min, metrics 4 min, MQAScore 12 min.
- **Total:** about 2.5-3 h to a dev result, plus 30 min for a test run if dev passes. A 1-epoch pilot on 2,000 clips cuts training to about 12 min, for about 1.5-2 h in total.

Risks:

- **Bandwidth:** checked and not a confound for FD-VGG or KL-PaSST (section 4 item 5).
- **Leakage:** keep the exclusion list in the results so the leakage check is visible.
- **First step after 17:00:** a 10-minute timing benchmark (20 training steps), to replace the guesses above.

## 7. How the extra results feed the fine-tune (agreed 11-16 13:56)

- **Low-pass result (in, 11-16 14:04):** low-passing did not improve FD or KL, so the rule's "does not" branch applies. There is no low-pass control. The fine-tune is judged on raw scores, and MusicCaps band-limiting is not a confound for FD-VGG or KL-PaSST. It could still matter for CLAP (48 kHz) and MQAScore; report the fine-tune's high-band energy (`ltas`) next to its scores.
- **Other stats, applied later the same way:**
  - **Mel texture `gv_ratio` (result in):** generated audio is rougher than real, not flatter: cfg7 2.07 [1.75, 2.46], cfg12 2.33. Guidance roughens it (cfg12 - cfg7 +0.171 log2 [+0.088, +0.260]), and rescale takes back about half (-0.086 [-0.174, -0.007]). This is the audio version of over-saturation. Fine-tune check: does the ratio move toward 1?
  - **Digital silence at clip ends:** 4.8% of frames for cfg7 vs 0.8% for real.
    - **Cause:** not checked.
    - **Fine-tune check:** does the share change?
  - **HW1 decade and market judges (result in):**
    - **Market:** generated audio reads as UK 59% vs real 31%. JSD vs real is 0.119 bits [0.069, 0.202] (dev cfg7) and 0.137 (test), against about 0.02 between two real resamples.
    - **Decade:** more 1960s and fewer 2010s; JSD 0.070 [0.038, 0.144].
    - **Per clip:** agreement with the real clip is near chance (0.27 vs 0.24 shuffled).
    - **Caveat:** HW1 models were trained on 30 s clips of other data.
    - **Fine-tune check:** does the JSD shrink toward the real-vs-real level?
  - **Step sweep:** if 50 steps match 100, generate the fine-tune evals at 50 steps to halve the time, otherwise keep 100.
  - **Oracle best-of-N ceiling:** check whether the fine-tune moves the mean, or only the spread across seeds.

## 8. Digital silence and the spectral centroid (11-16 14:10)

- **Silent tail (`scripts/silence_tail.py`, `results/analysis/silence_tail.csv`; silence = below -90 dBFS in 1,024-sample blocks):**
  - **Size:** generated clips end with about 0.3-0.5 s of digital silence; the dev mean is cfg3 0.49, cfg7 0.43, cfg12 0.40 and rescale12 0.38 s. 74-92% of generated clips have a tail over 0.1 s, against 1% of real clips (mean 0.007 s).
  - **Position:** the end of the clip only; the start is clean (mean under 0.02 s).
  - **Trend:** more guidance gives a shorter tail.
  - **Selection:** the submission's best-of-3 picks have shorter tails (test 60% over 0.1 s vs 77% for cfg7).
- **Effect on scores:** weak. Within a run, tail length vs CLAP is rho -0.18 (p 0.07) for cfg7 and -0.11 (p 0.26) for rescale12; vs KL it is about -0.06 (n.s.).
- **Likely cause (not tested):** the model is conditioned on `seconds_end` = 10. Our code and the TA code both cut the decode at exactly 10 s (`text_to_music/sampler.py` line 228). Stable Audio's timing conditioning was trained on clips padded with silence after their end, so the model may close the music just before the stated end.
- **Cheap test:** condition on 11 s, cut to 10 s, on dev cfg7 and rescale12 (about 30 min GPU each with scoring). If the tail disappears and the scores do not drop, it is a free fix for the fine-tune and the report.
- **Spectral centroid vs the high band:**
  - **The >16 kHz band** adds only 7-18 Hz to the centroid. Removing it moves cfg7 from 957 to 940 Hz, while the gap to real is 176 Hz.
  - **The 8-16 kHz band** carries most of the gap: it contributes 187 Hz for cfg7 vs 62 Hz for real. With everything above 8 kHz removed, the gap shrinks to 68 Hz (792 vs 724).
  - **A few clips drive it:** the median clip's 8-16 kHz share is lower for generated audio (0.06%) than for real (0.18%). The 10 brightest clips hold 83% of the cfg7 total.
  - **Consequence:** the centroid and mean-energy-share figures describe a few bright clips, not the typical clip. The mean log spectrum (LTAS dB) is the better summary. This also explains why cfg12 has a lower centroid than cfg7 although its mean 8-16 kHz level is higher: fewer extreme bright clips.
