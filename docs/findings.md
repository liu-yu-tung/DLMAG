# HW3 findings vs assumptions (11-15 11:00)

Numbers come from `scripts/analysis/findings_table.py`, written to `results/analysis/findings_table.csv`.

## Conventions

- **dCLAP / dKL:** paired per-prompt differences against the CFG-7 run of the same set. The baseline uses seed 0 and so does every method run, so each method run shares its initial noise with the baseline.
- **Intervals:** 95% bootstrap over the 100 prompts (1000 resamples, seed 0).
- **FD:** measured against all 5,367 MusicCaps clips ("mc"). The floor for 100 real clips is FD-VGG 1.9 and FD-MERT 21.

## Dev results so far

| run | CLAP | dCLAP [95%] | KL-PaSST | dKL [95%] | FD-VGG mc | FD-MERT mc | clip % | s/clip | MQAScore |
|---|---|---|---|---|---|---|---|---|---|
| cfg3 | 0.270 | -0.029 [-0.045, -0.013] | 1.70 | +0.19 [+0.06, +0.35] | 4.16 | 41.0 | 0.024 | 16.2 | |
| cfg7 (seed 0) | 0.299 | 0 | 1.51 | 0 | 4.02 | 43.5 | 0.087 | 16.1 | 0.546 [0.502, 0.588] |
| cfg7 seed 1 | 0.293 | -0.006 [-0.025, +0.011] | 1.46 | -0.04 [-0.16, +0.07] | 4.10 | 43.4 | 0.196 | 16.2 | |
| cfg7 seed 2 | 0.290 | -0.009 [-0.029, +0.012] | 1.54 | +0.04 [-0.08, +0.16] | 4.48 | 45.7 | 0.289 | 16.2 | |
| cfg12 | 0.314 | +0.015 [+0.001, +0.028] | 1.46 | -0.05 [-0.13, +0.03] | 4.38 | 44.6 | 0.281 | 16.3 | |
| apg7 | 0.290 | -0.009 [-0.020, +0.002] | 1.54 | +0.04 [-0.03, +0.11] | 4.16 | 42.0 | 0.027 | 16.2 | |
| apg12 | 0.310 | +0.011 [+0.002, +0.020] | 1.51 | +0.01 [-0.06, +0.08] | 4.38 | 43.8 | 0.088 | 16.2 | |
| apg12_r2 (almost no cap) | 0.305 | +0.006 [-0.006, +0.017] | 1.51 | +0.01 [-0.06, +0.07] | 4.40 | 44.6 | 0.089 | 16.2 | |
| tags (Qwen) | 0.248 | -0.051 [-0.074, -0.027] | 1.94 | +0.43 [+0.28, +0.63] | 4.92 | 51.9 | 0.195 | 16.2 | |
| tags_oracle (human) | 0.258 | -0.041 [-0.064, -0.020] | 1.73 | +0.22 [+0.07, +0.39] | 5.16 | 50.7 | 0.104 | 16.1 | |
| tags_caption | 0.298 | -0.001 [-0.017, +0.013] | 1.54 | +0.03 [-0.08, +0.17] | 4.25 | 43.4 | 0.156 | 16.2 | |
| interval_late (off for sigma <= 2.6) | 0.272 | -0.027 [-0.034, -0.019] | 1.66 | +0.16 [+0.08, +0.24] | 4.18 | 44.5 | 0.073 | 14.2 | |
| interval_late12 (off for sigma <= 12) | 0.224 | -0.075 [-0.094, -0.056] | 1.87 | +0.36 [+0.16, +0.57] | 4.16 | 43.9 | 0.124 | 12.7 | |
| best-of-3 by CLAP (cfg7) | 0.350 | +0.051 [+0.041, +0.062] | 1.43 | -0.08 [-0.18, +0.01] | 4.20 | 42.6 | | | queued |
| real dev clips | 0.302 | +0.004 [-0.023, +0.034] | 0 | | 1.88 | 20.8 | | | 0.646 [0.610, 0.681] |

- **Test:**
  - cfg7 seeds 0, 1, 2 give CLAP 0.314 / 0.321 / 0.311, KL 0.95 / 0.91 / 0.93, FD-VGG 8.16 / 7.89 / 8.07, FD-MERT 55.3 / 55.5 / 55.6.
  - MQAScore of seed 0 is 0.651 [0.616, 0.682].
- **MQAScore per attribute** (genre / instrument / mood / vocal / context):

  | run | genre | instrument | mood | vocal | context |
  |---|---|---|---|---|---|
  | dev cfg7 | 0.49 | 0.49 | 0.67 | 0.22 | 0.81 |
  | dev real | 0.73 | 0.51 | 0.84 | 0.49 | 0.86 |
  | test cfg7 | 0.52 | 0.58 | 0.73 | 0.03 | 0.94 |

## What met earlier assumptions

1. **CFG scale trade-off.**
   - **Assumption:** a higher scale buys text match at the cost of diversity and saturation.
   - **Result:**
     - CLAP rises steadily, and the gain is significant at both ends (cfg3 -0.029, cfg12 +0.015).
     - KL improves with scale.
     - FD-MERT gets worse as scale rises (41.0, 43.5, 44.6).
     - FD-VGG is lowest at 7.
     - Clipping is about 3x higher at scale 12 than at 7 (0.28% vs 0.09%).
2. **APG lowers the effective scale (momentum -0.5).**
   - APG7 lands between cfg3 and cfg7 on CLAP; APG12 lands between cfg7 and cfg12.
   - APG12 keeps about 70% of cfg12's CLAP gain (+0.011 vs +0.015) with cfg7's clipping (0.088%). This is the APG claim of less oversaturation at high scale, but see caveat 5 below.
3. **MQAScore is valid.**
   - Matched captions score above mismatched ones (0.700 vs 0.359, 11 of 12 items).
   - Repeat runs are bit-identical.
   - Real dev clips score above generated ones (0.646 vs 0.546).
4. **Dev and test levels differ** (caption-style gap). FD-VGG is 4.0 on dev and 8.1 on test; MQAScore is 0.55 on dev and 0.65 on test. Only within-set paired differences can be compared across sets.
5. **Determinism and code safety.** Reruns are bit-identical. The merged methods code reproduces the baseline clips exactly.
6. **Guidance-interval cost.** It saves compute as planned: 12% at sigma <= 2.6 and 21% at sigma <= 12, against the predicted about 15% fewer forward passes.

## What did not

1. **"Tag prompts help, because SAO was trained on comma-joined tags": wrong.**
   - Tags-only prompts are clearly worse on every metric: dCLAP -0.051, dKL +0.43, FD-VGG 4.92, FD-MERT 51.9.
   - Human oracle tags are also worse (-0.041), so poor Qwen tags are not the cause.
   - Tags + caption ties the caption alone.
   - KL and FD compare the generated audio with the real clip and never read the caption text, so tags really do produce audio further from the reference. This is not an artifact of CLAP scoring against the caption.
2. **"Late guidance is free to drop": wrong.**
   - The trajectory showed the guidance difference below 2% of ||D_c|| for sigma < 2.6, yet turning guidance off there costs dCLAP -0.027 [-0.034, -0.019] and dKL +0.16.
   - Turning it off for sigma <= 12 costs -0.075.
   - The norm ratio was the wrong yardstick. The guided update is (w - 1) = 6 times that difference, and late steps move x0 only a little, so guidance makes up a large share of what the late steps change.
3. **"CLAP of the real clip is a ceiling": wrong on dev.**
   - Real dev clips score CLAP 0.302 against their own captions; generated cfg7 scores 0.299 (difference +0.004 [-0.023, +0.034]).
   - CLAP cannot separate real from generated on dev, but MQAScore can (0.646 vs 0.546).
   - The 0.452 retrieved-clip "ceiling" on test comes from max-selection, not from the audio being better.
4. **Projection matters little (prediction from the trajectory): not yet tested.** APG12 with almost no cap (r2) sits close to APG12, which hints that the cap does little too. The ablation runs decide.
5. **Clipping as an APG benefit: weaker than it looks.** Clip fraction varies a lot between seeds at the same setting (cfg7 seeds give 0.087, 0.196 and 0.289%), so means are dominated by a few clips. Claims about clipping need a per-clip paired comparison on the same seed.

## Updated assumptions

1. **A1, guidance:** guidance matters at every noise level for SAO. Late guidance refines timbre and detail, which CLAP and PaSST read. A small relative guidance norm does not mean a small effect.
2. **A2, prompt format:** the prompt format is not a bottleneck for SAO's T5 conditioning. Full sentences carry information that tag lists lose (relations, recording quality, structure). Tags add nothing on top of the caption.
3. **A3, CFG and APG:** CFG scale is a one-knob trade-off: CLAP and KL against FD-MERT and clipping. APG moves along roughly the same curve. On the CLAP vs FD-MERT plane, APG7 and APG12 sit 0.5-0.7 FD-MERT below the CFG line, but the seed spread of FD-MERT at one setting is 2.3. So APG shows no proven gain beyond its effective scale.
4. **A4, noise floor:** seed noise is about ±0.01 CLAP. A paired method difference under 0.01 is not a finding. Rank on paired differences that share the seed, and repeat the final setting's seeds.
5. **A5, metrics:** MQAScore is the more discriminative text-match metric here. CLAP saturates near 0.30 on dev, where real and generated are equal.
6. **A6, vocals:** vocals are SAO's largest failure. The MQAScore vocal attribute is 0.22 on dev (real 0.49) and 0.03 on test. SAO almost never produces the vocals a caption asks for.
7. **A7, best-of-3:** best-of-3 by CLAP gains +0.051 CLAP (partly circular). It does not hurt the independent metrics: KL is better (n.s.), FD-VGG equals the seed mean (4.20) and FD-MERT is better (42.6 vs a seed mean of 44.2).

## New predictions (written 11-15 11:00, before these results exist)

| run | prediction | reason |
|---|---|---|
| interval_early (off for sigma > 110) | dCLAP between -0.03 and -0.08; FD-MERT lower (better) than cfg7 | ETTA found early CFG matters for music; Kynkaanniemi et al. found high-noise guidance costs diversity |
| rescale12 (phi 0.7) | CLAP between cfg7 and cfg12 (dCLAP 0 to +0.012); clipping at most cfg12's on the same seed | rescale shrinks the guided output's std toward the conditional's, so it lowers the effective scale |
| compose7 (caption + tags, score space) | dCLAP between -0.03 and 0; dKL >= 0; about 1.5x s/clip (about 24 s) | tag guidance alone hurts; composition dilutes the caption's share of the scale |
| apg12_projection (projection only) | within 0.005 CLAP of cfg12; clipping close to cfg12 | the parallel fraction is about 0.2 early, so projection removes about 4% of the update |
| apg12_cap (projection + cap, no momentum) | CLAP between apg12 and cfg12 | momentum, not the cap, lowers the effective scale |
| apg12_eta1 (cap + momentum, no projection) | within 0.005 CLAP of apg12 | same reason as apg12_projection |
| MQAScore, cfg3 / cfg12 | cfg3 < cfg7 (0.546) < cfg12 | guidance strengthens attribute adherence |
| MQAScore, tags | at least cfg7 - 0.02, so far less of a penalty than its CLAP gap | the tags are the same ones MQAScore asks about (circular) |
| MQAScore, best-of-3 | above cfg7 by +0.01 to +0.04 | CLAP selection partly transfers to an independent judge |
| MQAScore, interval_late | below cfg7 | late guidance carries detail (A1) |
| MQAScore, vocal attribute | stays the lowest attribute in every run | A6 |

## Update 11-15 11:35: interval_early, and two more predictions

- **Result:** `interval_early` (guidance off for sigma > 110, the first 21 of 100 steps):
  - CLAP 0.299, dCLAP +0.000 [-0.003, +0.005]
  - KL 1.51
  - FD-VGG 3.96 vs 4.02
  - FD-MERT 43.8 vs 43.5
  - clip 0.085%
  - 14.7 s/clip, 9% faster than cfg7
- **Prediction (dCLAP -0.03 to -0.08, FD-MERT lower): wrong.** Dropping the earliest guidance is free, and FD does not change either.
- **Revised A1:** guidance does nothing at the highest noise (sigma > 110), even though its relative norm is largest there (about 0.22). It matters at low noise, where the norm is smallest (dropping sigma <= 2.6 costs -0.027; dropping sigma <= 12 costs -0.075). So the guidance norm is a poor guide to where guidance matters, in both directions. A likely mechanism: the SDE solver adds fresh noise each step, so whatever early guidance does is washed out before the content settles.
- **Schedule facts** (SAO, 100 steps): sigma > 110 covers 21 steps, > 60 covers 29, > 30 covers 38, > 12 covers 50, > 2.6 covers 71.
- **New predictions** (written 11:35, before the runs; queued as `dev__interval_early30` and `dev__interval_early12`, ahead of the MQAScore block):

| run | prediction | reason |
|---|---|---|
| interval_early30 (off for sigma > 30, the first 38 steps; close to ETTA's first 40%) | dCLAP between -0.02 and 0; s/clip about 13 (-19%) | between free (sigma > 110) and harmful; ETTA reports harm for its 40% window |
| interval_early12 (off for sigma > 12, the first 50 steps) | dCLAP between -0.05 and -0.01; s/clip about 12 (-25%) | 12 < sigma <= 30 is where guidance starts to shape content |

- **Possible method:** if `early30` is free, "no guidance for sigma > 30" is a quality-neutral 19% speedup. It would be the guidance-interval result for the report, and it combines with any scale or APG setting.

## Paired FD-VGG against cfg7 (11-15 12:00, `scripts/evaluation/paired_fd.py`, `results/analysis/paired_fd_dev.json`)

The same resampled prompt ids are used for both runs, against all MusicCaps; cfg7 itself scores 4.024. The CPU checks passed: a run against itself gives exactly 0, and each point FD matches its `metrics.json` value.

| run | dFD-VGG [95%] |
|---|---|
| cfg3 | +0.14 [-0.32, +0.52] |
| cfg12 | +0.36 [-0.02, +0.66] |
| apg7 | +0.13 [-0.25, +0.39] |
| apg12 | +0.36 [+0.11, +0.60] |
| apg12_r2 | +0.37 [+0.09, +0.60] |
| tags | +0.90 [+0.24, +1.59] |
| tags_caption | +0.23 [-0.30, +0.70] |
| tags_oracle | +1.14 [+0.51, +1.91] |
| interval_late | +0.16 [-0.09, +0.37] |
| interval_late12 | +0.13 [-0.45, +0.86] |
| cfg7 seed 1 | +0.08 [-0.38, +0.55] |
| cfg7 seed 2 | +0.46 [-0.09, +1.04] |
| best-of-3 | +0.17 [-0.28, +0.59] |

- **Significantly worse than cfg7 on FD-VGG:** apg12, apg12_r2, tags and tags_oracle. No run is significantly better. No PaSST FD difference excludes 0.
- **Reading:**
  - APG12 reaches about the same FD-VGG as cfg12 (+0.36 each), but only APG12's interval excludes 0. Its CLAP gain comes with the same FD cost as raising the CFG scale, which supports A3 (APG moves along the CFG curve).
  - Best-of-3 shows no FD cost.
- **FD-MERT not paired:** only the MusicCaps MERT embeddings are cached, not per-run embeddings. Pairing it would need a GPU re-embed, about 1 min per run.
- **What the selection rule would see** (applied only at the check-in with the user):
  - cfg12 passes the FD check by a narrow margin.
  - apg12 fails it.
  - best-of-3 passes it, and its MQAScore is still pending.

## Update 11-15 12:20: rescale12, extra metrics, one correction

- **Correction:** the test MQAScore vocal attribute (0.03) rests on ONE item. Only 1 test caption has a vocal tag, and 3 of 100 test captions mention vocals at all. It is not evidence; drop it. The vocals finding (A6) stands on dev:
  - MQAScore vocal 0.22 generated vs 0.49 real, 50 items.
  - PaSST vocal presence on vocal captions 0.063 for cfg7 vs 0.173 for the real clips (0.019 vs 0.052 on the other captions).
- **rescale12** (CFG 12, rescale phi 0.7):
  - CLAP 0.326, dCLAP +0.028 [+0.017, +0.039]
  - KL 1.42, dKL -0.084 [-0.170, +0.003]
  - FD-VGG 4.13; FD-MERT 41.0, below all three cfg7 seeds (43.4-45.7)
  - clip 0.068%; RMS -18.0 dBFS (cfg12 -16.7, real -18.5)
  - FD-CLAP (paired) 0.731, the lowest of any generated run
  - **Prediction (dCLAP 0 to +0.012): wrong.** It beats cfg12 (+0.015) and keeps or improves the diversity-side metrics. This is the off-the-trade-off-curve result APG did not deliver. The paired FD-VGG check is still to run.
- **Extra metrics** (`scripts/evaluation/extra_metrics.py`, `results/analysis/extra_metrics.csv`, 9b0b92f). Seed noise is noted per metric, since cfg7 seeds 0-2 give the spread at one setting.
  - **CLAP retrieval R@1** separates real from generated, where mean cosine could not:
    - real 0.40 vs cfg7 0.21 (median rank 2 vs 3);
    - seed spread 0.21-0.28;
    - best-of-3 0.30, rescale12 0.28.
  - **Paired output change vs cfg7** (CLAP-audio cosine; a different seed gives 0.67). This confirms the washout mechanism of A1: the early-guidance output barely changes.
    - interval_early 0.98: the early guidance is washed out.
    - interval_late 0.92: a small change, yet the metrics got worse.
    - tags 0.56: as different as a new seed.
  - **Precision/recall** (PANN, against the 100 paired real clips):
    - precision is stable, about 0.8 (seeds 0.80-0.82);
    - recall is noisy (seeds 0.24-0.41), so its differences are weak evidence;
    - higher CFG scale lowers recall (cfg12 0.32 vs cfg7 0.41);
    - APG12 keeps recall at 0.41 at its higher CLAP, matching the APG paper's recall claim;
    - rescale12 has 0.46 recall and 0.84 precision.
  - **Loudness:** higher CFG is louder (cfg3 -18.8, cfg7 -17.4, cfg12 -16.7 dBFS). APG12 (-17.6) and rescale12 (-18.0) remove most of that, a cleaner saturation signal than clip fraction. Generated audio is also brighter than real (centroid about 2600 vs 2200 Hz).
  - **Coverage:** real 0.088 vs generated 0.011-0.022 against all MusicCaps. With 100 samples the ceiling is the real value.
- **Plan consequence:** rescale12 becomes a headline candidate. The useful next dev runs are rescale12 seeds 1-2 (seed noise of the new leader, plus best-of-3 at rescale12) rather than cfg12 seeds. A phi or scale sweep around it is optional.

## Consequences for the plan

- **Guidance interval:** demoted from main new method to a clear negative result with a mechanism. The report keeps it (speed vs quality, and "why the norm ratio misleads").
- **Headline method candidates:**
  - APG at high scale: the CLAP gain at cfg7 clipping, if per-clip paired clipping supports it.
  - Best-of-3 selection, if MQAScore confirms it.
  - Tag prompts move to "did not work, and why".
- **Gap for the selection rule:** the paired FD-difference bootstrap is not implemented yet (`evaluate.py` gives per-run FD intervals only). It is needed before the selection check-in (about 17:30).
- **New analysis to add:**
  - per-clip paired clipping for APG vs CFG on the same seed;
  - real-vs-generated discrimination, CLAP vs MQAScore (slide 17);
  - MQAScore per attribute, with vocals as SAO's main failure.
