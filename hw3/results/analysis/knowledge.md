HW3 knowledge sheet, rewritten 11-16 14:26 by scripts/analysis/knowledge.py

## Predictions (docs/predictions.toml): MISSED ones are where the strategy must change

met    dev__interval_early30 vs dev__cfg7, clap in [-0.020, +0.000] (written 11-15 11:35): got -0.003 [-0.012, +0.005]
       next: if the interval includes 0 (free): keep dev__interval_early12 to find the boundary. The guidance-interval story becomes 'the first 38 steps need no guidance'.
MISSED dev__cfg7_best3 vs dev__cfg7, mqa in [+0.010, +0.040] (written 11-15 11:00): got -0.007 [-0.036, +0.019]
       next: if at or below 0: best-of-3 is CLAP-circular; submit a single run and report best-of-3 as the circularity study. If above +0.04: same as met.
met    dev__cfg12 vs dev__cfg7, mqa in [+0.000, +0.060] (written 11-15 11:00): got +0.007 [-0.010, +0.023]
       next: MQAScore agrees with CLAP on the scale trend.
met    dev__cfg3 vs dev__cfg7, mqa in [-0.080, +0.000] (written 11-15 11:00): got -0.031 [-0.052, -0.011]
       next: as above.
met    dev__rescale12 vs dev__cfg7, mqa in [+0.000, +0.050] (written 11-15 11:58): got +0.007 [-0.009, +0.024]
       next: rescale12 is the single-run leader on both judges.
met    dev__rescale12_seed1 vs dev__cfg7_seed1, clap in [+0.010, +0.045] (written 11-15 11:58): got +0.032 [+0.020, +0.043]
       next: the rescale gain replicates on a second seed.
met    dev__rescale12_seed2 vs dev__cfg7_seed2, clap in [+0.010, +0.045] (written 11-15 11:58): got +0.023 [+0.007, +0.039]
       next: the gain holds on 3 of 3 seeds; rescale12 is the single-run pick over cfg12.
met    dev__rescale12_best3 vs dev__rescale12, clap in [+0.030, +0.060] (written 11-15 11:58): got +0.042 [+0.032, +0.053]
       next: selection and rescale add up; best-of-3 at rescale12 has the highest dev CLAP.
met    dev__rescale12_best3 vs dev__rescale12, mqa in [+0.000, +0.040] (written 11-15 11:58): got +0.011 [-0.004, +0.025]
       next: best-of-3 at rescale12 qualifies in the user's sense if KL is not worse beyond its interval.
met    dev__rescale18 vs dev__rescale12, clap in [-0.010, +0.012] (written 11-15 11:58): got +0.002 [-0.010, +0.015]
       next: the scale axis is closed at 12 for rescale; no further scale runs.
MISSED dev__rescale7 vs dev__cfg7, clap in [+0.000, +0.015] (written 11-15 11:58): got -0.003 [-0.012, +0.004]
       next: if above +0.015: rescale helps by itself, independent of scale; say so. If below 0: rescale needs a high scale.
MISSED dev__rescale12_phi1 vs dev__rescale12, clap in [-0.015, +0.005] (written 11-15 11:58): got +0.006 [-0.004, +0.015]
       next: if above +0.005 with an interval excluding 0: phi 1 is better; it is a report ablation only if the selection is already committed.
met    dev__rescale12_best3_c8998 vs dev__rescale12, clap in [+0.010, +0.035] (written 11-15 12:04): got +0.015 [+0.003, +0.027]
       next: part of the best-of-3 gain is real under a different ranker; report same-checkpoint minus cross-checkpoint as the circular share.
met    dev__rescale12_best3_c8998 vs dev__rescale12, mqa in [+0.000, +0.040] (written 11-15 12:04): got +0.014 [-0.003, +0.030]
       next: with KL not worse beyond its interval and the paired FD-VGG check passed, this is the only best-of-3 candidate for the submission.
met    dev__cfg7_best3_c8998 vs dev__cfg7, clap in [+0.015, +0.040] (written 11-15 12:04): got +0.031 [+0.019, +0.042]
       next: the circular share at cfg7 is +0.051 minus this value; one number for slide 15.
met    dev__real vs dev__cfg7, alm in [+0.030, +0.200] (written 11-15 16:10): got +0.109 [+0.078, +0.140]
       next: two judges agree that real beats generated, which CLAP cannot see (0.302 vs 0.299); one line on slide 17.
met    dev__rescale12 vs dev__cfg7, alm in [+0.000, +0.040] (written 11-15 16:10): got +0.030 [+0.009, +0.050]
       next: the rescale gain is not a CLAP-only effect; cite both judges on the method slide.
MISSED dev__cfg7 vs dev__real, vocal_db in [-50.000, -15.000] (written 11-15 16:10): got -12.742 [-18.419, -7.100]
       next: if above -15: SAO does make vocals and MQAScore's vocal failure is about their quality, not their presence; reword the vocal finding.
met    test__rescale12 vs test__cfg7, clap in [+0.010, +0.045] (written 11-15 17:37): got +0.010 [-0.000, +0.021]
       next: the CLAP gain carries from dev to test; report the dev-test agreement.
met    test__rescale12 vs test__cfg7, kl in [-0.100, +0.100] (written 11-15 17:37): got -0.021 [-0.101, +0.053]
       next: a flat test KL is expected, not a failure; the report says so next to the dev KL gain.
met    test__rescale12_best3_c8998 vs test__cfg7, clap in [+0.020, +0.070] (written 11-15 17:37): got +0.040 [+0.027, +0.051]
       next: the submission's CLAP gain holds on test.
met    test__rescale12_best3_c8998 vs test__cfg7, kl in [-0.250, +0.050] (written 11-15 17:37): got -0.067 [-0.170, +0.031]
       next: report the KL change with its interval; claim a gain only if the interval excludes 0.
met    dev__rescale12_valid vs dev__rescale12, clap in [-0.015, +0.015] (written 11-15 18:10): got -0.003 [-0.013, +0.008]
       next: the fix changes how rescale is measured, not what it buys in CLAP; report KL and FD-VGG as the deciding metrics.
met    dev__rescale12_valid vs dev__rescale12, kl in [-0.120, +0.040] (written 11-15 18:10): got -0.055 [-0.115, +0.007]
       next: the fix behaves like a stronger phi; the pilot slide says so.
met    dev__rescale12_best3_mqa vs dev__rescale12, clap in [+0.000, +0.020] (written 11-15 18:10): got +0.003 [-0.010, +0.016]
       next: a judge ranker lifts CLAP a little without being CLAP; report next to the cross-checkpoint ranker.
met    dev__rescale12_best3_mqa vs dev__rescale12, kl in [-0.150, +0.000] (written 11-15 18:10): got -0.041 [-0.119, +0.038]
       next: any reasonable ranker of 3 seeds lowers KL; the selection effect is generic, not specific to CLAP.
met    test__rescale12_best3_mqa vs test__rescale12, clap in [+0.000, +0.015] (written 11-15 18:10): got +0.011 [-0.002, +0.022]
       next: the judge ranker transfers like the CLAP rankers.
MISSED dev__rescale12_degraded vs dev__rescale12, kl in [-0.200, +0.000] (written 11-15 18:10): got +0.075 [+0.032, +0.118]
       next: if above 0: KL is not moved by recording conditions at this SNR; the metric is safer than feared.
met    dev__rescale12_degraded vs dev__rescale12, mqa in [-0.030, +0.010] (written 11-15 18:10): got -0.021 [-0.035, -0.009]
       next: with the KL result: the judge is robust to recording conditions where KL is not.
MISSED dev__rescale12_best3_probe vs dev__rescale12, clap in [+0.000, +0.015] (written 11-15 20:04): got -0.002 [-0.015, +0.011]
       next: if at or below 0: the probes see attributes that CLAP does not reward; report the per-group hit rates instead of a score gain.
met    dev__rescale12_best3_probe vs dev__rescale12, kl in [-0.120, +0.000] (written 11-15 20:04): got -0.040 [-0.119, +0.040]
       next: the ranker helps the reference-based metric too.
met    test__rescale12_best3_probe vs test__rescale12, clap in [+0.000, +0.012] (written 11-15 20:04): got +0.010 [-0.001, +0.022]
       next: the ranker transfers, at a smaller size, as expected from the caption style gap.
met    dev__rescale12_clean vs dev__rescale12, clap in [-0.010, +0.010] (written 11-15 20:15): got -0.003 [-0.016, +0.008]
       next: cleaning is neutral on CLAP: the dropped sentences were not steering the music.
met    dev__rescale12_clean vs dev__rescale12, kl in [+0.000, +0.080] (written 11-15 20:15): got +0.013 [-0.057, +0.089]
       next: KL partly rewards matching recording conditions (with the degrade probe).
met    dev__rescale12_clean vs dev__rescale12, mqa in [+0.000, +0.030] (written 11-15 20:15): got +0.004 [-0.007, +0.015]
       next: the judge rewards following the musical words; with KL, the metrics disagree on what 'better' means.
met    dev__rescale12_loop2 vs dev__rescale12_best3_probe, clap in [-0.005, +0.010] (written 11-15 23:31): got -0.001 [-0.009, +0.006]
       next: round 2 adds little beyond round 1: the probe gate and CLAP agree weakly.
met    dev__rescale12_loop2 vs dev__rescale12_best3_probe, kl in [-0.050, +0.030] (written 11-15 23:31): got -0.024 [-0.071, +0.018]
       next: KL is not moved by attribute emphasis.
MISSED dev__rescale12_loop2 vs dev__rescale12_best3_probe, kept_fraction in [+0.350, +0.650] (written 11-15 23:31): got +0.303 [+0.303, +0.303]
       next: if above 0.65: emphasis sentences reliably make the model render missing attributes; if below 0.35: they hurt.
MISSED dev__rescale12_ablate vs dev__rescale12, clap in [-0.015, -0.002] (written 11-16 00:40): got -0.025 [-0.041, -0.010]
       next: if above -0.002: CLAP barely weighs single instruments or voices in long captions.
MISSED dev__rescale12_ablate vs dev__rescale12, mqa in [-0.030, -0.003] (written 11-16 00:40): got +0.001 [-0.013, +0.015]
       next: if above -0.003: the judge does not penalize a missing asked instrument or voice at this size.
met    dev__rescale12_followed vs dev__rescale12, clap in [+0.000, +0.015] (written 11-16 01:02): got +0.002 [-0.010, +0.014]
       next: steering by followed words gives a small CLAP gain without changing the caption.
MISSED dev__rescale12_followed vs dev__rescale12, kl in [-0.050, +0.050] (written 11-16 01:02): got -0.093 [-0.164, -0.025]
       next: if below -0.05: the followed attributes also carry the reference tags; if above +0.05: the push moves clips off the reference style.
met    dev__rescale12_followed vs dev__rescale12, mqa in [+0.000, +0.020] (written 11-16 01:02): got +0.001 [-0.016, +0.017]
       next: the judge rewards the followed attributes; with CLAP, the finding produces a score change.
met    dev__rescale16 vs dev__rescale12, clap in [-0.010, +0.010] (written 11-16 01:04): got +0.005 [-0.005, +0.015]
       next: the extra weight alone does not move CLAP; any followed-word gain is from the words.
MISSED dev__rescale12_followed vs dev__rescale16, clap in [+0.000, +0.012] (written 11-16 01:04): got -0.003 [-0.017, +0.010]
       next: if below 0: the gain (if any) over rescale12 is just guidance strength.
MISSED dev__rescale12_followed_sum12 vs dev__rescale12, clap in [-0.005, +0.012] (written 11-16 01:05): got -0.006 [-0.018, +0.007]
       next: if below -0.005: the caption's other words (mood, texture) carry CLAP too and lose weight; if above +0.012: the followed words are what CLAP rewards.
met    dev__rescale12_followed_sum12 vs dev__rescale12, mqa in [-0.005, +0.020] (written 11-16 01:05): got +0.013 [-0.005, +0.029]
       next: the judge's attribute questions benefit from the re-weighting.

Pending:
  dev__interval_early12 vs dev__cfg7, clap in [-0.050, -0.010] (written 11-15 11:35)
  dev__apg12_projection vs dev__cfg12, clap in [-0.005, +0.005] (written 11-15 11:00)
  dev__apg12_eta1 vs dev__apg12, clap in [-0.005, +0.005] (written 11-15 11:00)
  dev__compose7 vs dev__cfg7, clap in [-0.030, +0.000] (written 11-15 11:00)
  dev__tags vs dev__cfg7, mqa in [-0.020, +0.100] (written 11-15 11:00)
  dev__interval_late vs dev__cfg7, mqa in [-0.100, +0.000] (written 11-15 11:00)
  dev__rescale12_best3_mqa vs dev__rescale12_best3_c8998, alm in [+0.000, +0.060] (written 11-15 18:10)
  dev__rescale12_best5_c8998 vs dev__rescale12_best3_c8998, clap in [+0.000, +0.008] (written 11-15 23:34)
  test__rescale12_best5_c8998 vs test__rescale12_best3_c8998, clap in [+0.000, +0.008] (written 11-15 23:34)
  dev__rescale12_best5_probe vs dev__rescale12_best3_probe, clap in [-0.005, +0.008] (written 11-15 23:34)
  dev__rescale12_best5_mqa vs dev__rescale12_best3_mqa, mqa in [+0.000, +0.020] (written 11-15 23:34)
  dev__rescale12_followed_sentence vs dev__rescale12_followed_sum12, clap in [+0.000, +0.012] (written 11-16 01:11)

## Selection board (dev, paired differences vs dev__cfg7, 95% bootstrap over prompts)

run | CLAP | dCLAP | dKL | dFD-VGG | dMQAScore | rule
dev__rescale12_best3 | 0.369 | +0.070 [+0.056, +0.083] | -0.22 [-0.32, -0.14] | +0.26 [-0.21, +0.51] | +0.017 [-0.004, +0.037] | analysis only (ranked by the scoring checkpoint)
dev__cfg7_best3 | 0.350 | +0.051 [+0.041, +0.062] | -0.08 [-0.18, +0.01] | +0.17 [-0.28, +0.59] | -0.007 [-0.036, +0.019] | analysis only (ranked by the scoring checkpoint)
dev__rescale12_best3_c8998 | 0.341 | +0.042 [+0.027, +0.057] | -0.21 [-0.32, -0.10] | -0.00 [-0.43, +0.36] | +0.020 [-0.001, +0.042] | candidate
dev__rescale12_phi1 | 0.332 | +0.033 [+0.021, +0.047] | -0.13 [-0.21, -0.04] | -0.20 [-0.60, +0.12] | pending | candidate
dev__rescale16 | 0.332 | +0.033 [+0.019, +0.046] | -0.17 [-0.29, -0.06] | +0.17 [-0.27, +0.46] | -0.009 [-0.031, +0.012] | candidate
dev__cfg7_best3_c8998 | 0.330 | +0.031 [+0.019, +0.042] | -0.04 [-0.14, +0.07] | +0.16 [-0.24, +0.63] | +0.006 [-0.013, +0.024] | candidate
dev__rescale12_best3_mqa | 0.329 | +0.031 [+0.014, +0.046] | -0.13 [-0.22, -0.03] | +0.04 [-0.40, +0.36] | pending | candidate
dev__rescale12_followed | 0.329 | +0.030 [+0.016, +0.045] | -0.18 [-0.28, -0.07] | +0.45 [-0.01, +0.76] | +0.007 [-0.011, +0.026] | candidate
dev__rescale18 | 0.329 | +0.030 [+0.015, +0.045] | -0.19 [-0.30, -0.09] | +0.14 [-0.37, +0.47] | pending | candidate
dev__rescale12 | 0.326 | +0.028 [+0.017, +0.039] | -0.08 [-0.17, +0.00] | +0.11 [-0.27, +0.36] | +0.007 [-0.009, +0.024] | candidate
dev__rescale12_seed1 | 0.324 | +0.025 [+0.007, +0.042] | -0.17 [-0.28, -0.06] | +0.19 [-0.29, +0.59] | +0.021 [-0.002, +0.043] | candidate
dev__rescale12_valid | 0.324 | +0.025 [+0.013, +0.037] | -0.14 [-0.22, -0.05] | -0.13 [-0.61, +0.20] | pending | candidate
dev__rescale12_degraded | 0.324 | +0.025 [+0.011, +0.038] | -0.01 [-0.10, +0.08] | +0.70 [+0.18, +1.05] | -0.015 [-0.033, +0.005] | out: worse FD-VGG
dev__rescale12_loop2 | 0.323 | +0.024 [+0.007, +0.041] | -0.15 [-0.27, -0.04] | +0.28 [-0.21, +0.65] | +0.013 [-0.006, +0.034] | candidate
dev__rescale12_clean | 0.323 | +0.024 [+0.009, +0.039] | -0.07 [-0.17, +0.02] | +0.18 [-0.22, +0.50] | +0.011 [-0.008, +0.031] | candidate
dev__rescale12_followed_sum12 | 0.321 | +0.022 [+0.007, +0.036] | -0.08 [-0.19, +0.02] | +0.27 [-0.08, +0.60] | +0.019 [-0.001, +0.039] | candidate
dev__rescale12_steps25 | 0.314 | +0.015 [+0.001, +0.029] | -0.01 [-0.09, +0.07] | +0.10 [-0.26, +0.42] | +0.003 [-0.013, +0.020] | candidate
dev__rescale12_seed2 | 0.314 | +0.015 [-0.002, +0.031] | -0.11 [-0.21, -0.01] | +0.37 [-0.13, +0.76] | -0.001 [-0.023, +0.021] | tied with cfg7 (default wins)
dev__cfg12 | 0.314 | +0.015 [+0.001, +0.028] | -0.05 [-0.13, +0.03] | +0.36 [-0.02, +0.66] | +0.007 [-0.010, +0.023] | candidate
dev__apg12 | 0.310 | +0.011 [+0.002, +0.020] | +0.01 [-0.06, +0.08] | +0.36 [+0.11, +0.60] | pending | out: worse FD-VGG
dev__apg12_r2 | 0.305 | +0.006 [-0.006, +0.017] | +0.01 [-0.06, +0.07] | +0.37 [+0.09, +0.60] | pending | out: worse FD-VGG
dev__rescale12_ablate | 0.301 | +0.003 [-0.014, +0.021] | +0.01 [-0.12, +0.13] | +0.13 [-0.33, +0.46] | +0.008 [-0.013, +0.030] | tied with cfg7 (default wins)
dev__interval_early | 0.299 | +0.000 [-0.003, +0.005] | +0.01 [-0.01, +0.02] | -0.07 [-0.12, +0.00] | pending | tied with cfg7 (default wins)
dev__tags_caption | 0.298 | -0.001 [-0.017, +0.013] | +0.03 [-0.08, +0.17] | +0.23 [-0.30, +0.70] | pending | tied with cfg7 (default wins)
dev__rescale7 | 0.296 | -0.003 [-0.012, +0.004] | +0.03 [-0.02, +0.09] | -0.01 [-0.26, +0.21] | pending | tied with cfg7 (default wins)
dev__interval_early30 | 0.296 | -0.003 [-0.012, +0.005] | +0.08 [+0.02, +0.15] | +0.16 [-0.06, +0.44] | pending | out: worse KL
dev__cfg7_seed1 | 0.293 | -0.006 [-0.025, +0.011] | -0.04 [-0.16, +0.07] | +0.07 [-0.38, +0.55] | pending | tied with cfg7 (default wins)
dev__cfg7_seed2 | 0.290 | -0.009 [-0.029, +0.012] | +0.04 [-0.08, +0.16] | +0.46 [-0.09, +1.04] | pending | tied with cfg7 (default wins)
dev__apg7 | 0.290 | -0.009 [-0.020, +0.002] | +0.04 [-0.03, +0.11] | +0.13 [-0.25, +0.39] | pending | tied with cfg7 (default wins)
dev__interval_late | 0.272 | -0.027 [-0.034, -0.019] | +0.16 [+0.08, +0.24] | +0.16 [-0.09, +0.37] | pending | out: worse KL
dev__cfg3 | 0.270 | -0.029 [-0.045, -0.013] | +0.19 [+0.06, +0.35] | +0.14 [-0.32, +0.52] | -0.031 [-0.052, -0.011] | out: worse KL
dev__tags_oracle | 0.258 | -0.041 [-0.064, -0.020] | +0.22 [+0.07, +0.39] | +1.14 [+0.51, +1.91] | pending | out: worse KL and FD-VGG
dev__tags | 0.248 | -0.051 [-0.074, -0.027] | +0.43 [+0.28, +0.63] | +0.90 [+0.24, +1.59] | pending | out: worse KL and FD-VGG
dev__interval_late12 | 0.224 | -0.075 [-0.094, -0.056] | +0.36 [+0.16, +0.57] | +0.13 [-0.45, +0.86] | pending | out: worse KL
dev__rescale12_best3_probe | 0.000 | +0.025 [+0.010, +0.042] | -0.12 [-0.23, -0.02] | pending | pending | FD-VGG pending

Rule: highest dev CLAP unless FD-VGG or KL is worse beyond its interval; ties go to cfg7. 'candidate' = passes and dCLAP excludes 0.
The choice is made only at the check-in with the user.

## Head-to-head among candidates (paired, row minus the next lower candidate)

dev__rescale12_best3_c8998 vs dev__rescale12_phi1: dCLAP +0.009 [-0.007, +0.025], win rate 0.57, dKL -0.08 [-0.19, +0.02] -> TIE (interval includes 0)
dev__rescale12_phi1 vs dev__rescale16: dCLAP +0.001 [-0.012, +0.013], win rate 0.48, dKL +0.04 [-0.05, +0.15] -> TIE (interval includes 0)
dev__rescale16 vs dev__cfg7_best3_c8998: dCLAP +0.002 [-0.012, +0.016], win rate 0.48, dKL -0.14 [-0.23, -0.03] -> TIE (interval includes 0)
dev__cfg7_best3_c8998 vs dev__rescale12_best3_mqa: dCLAP +0.000 [-0.012, +0.013], win rate 0.54, dKL +0.09 [-0.00, +0.17] -> TIE (interval includes 0)
dev__rescale12_best3_mqa vs dev__rescale12_followed: dCLAP +0.001 [-0.015, +0.017], win rate 0.49, dKL +0.05 [-0.04, +0.13] -> TIE (interval includes 0)
dev__rescale12_followed vs dev__rescale18: dCLAP +0.000 [-0.013, +0.014], win rate 0.52, dKL +0.01 [-0.08, +0.09] -> TIE (interval includes 0)
dev__rescale18 vs dev__rescale12: dCLAP +0.002 [-0.010, +0.015], win rate 0.44, dKL -0.10 [-0.18, -0.03] -> TIE (interval includes 0)
dev__rescale12 vs dev__rescale12_seed1: dCLAP +0.002 [-0.014, +0.019], win rate 0.49, dKL +0.08 [-0.02, +0.20] -> TIE (interval includes 0)
dev__rescale12_seed1 vs dev__rescale12_valid: dCLAP +0.001 [-0.016, +0.017], win rate 0.51, dKL -0.03 [-0.14, +0.08] -> TIE (interval includes 0)
dev__rescale12_valid vs dev__rescale12_loop2: dCLAP +0.000 [-0.016, +0.015], win rate 0.43, dKL +0.01 [-0.09, +0.11] -> TIE (interval includes 0)
dev__rescale12_loop2 vs dev__rescale12_clean: dCLAP +0.000 [-0.018, +0.017], win rate 0.37, dKL -0.08 [-0.19, +0.03] -> TIE (interval includes 0)
dev__rescale12_clean vs dev__rescale12_followed_sum12: dCLAP +0.002 [-0.015, +0.018], win rate 0.61, dKL +0.01 [-0.09, +0.11] -> TIE (interval includes 0)
dev__rescale12_followed_sum12 vs dev__rescale12_steps25: dCLAP +0.007 [-0.010, +0.022], win rate 0.51, dKL -0.08 [-0.17, +0.03] -> TIE (interval includes 0)
dev__rescale12_steps25 vs dev__cfg12: dCLAP +0.000 [-0.013, +0.013], win rate 0.53, dKL +0.04 [-0.03, +0.11] -> TIE (interval includes 0)

## Best-of-3 runs vs their own seed 0 (decision #2: better on one metric, not worse beyond its interval on any other)

dev__rescale12_best3 vs dev__rescale12: dCLAP +0.042 [+0.032, +0.053] (circular), dKL -0.14 [-0.22, -0.07], dMQAScore +0.011 [-0.004, +0.025] -> QUALIFIES
dev__cfg7_best3 vs dev__cfg7: dCLAP +0.051 [+0.041, +0.062] (circular), dKL -0.08 [-0.18, +0.01], dMQAScore -0.007 [-0.036, +0.019] -> QUALIFIES
dev__rescale12_best3_c8998 vs dev__rescale12: dCLAP +0.015 [+0.003, +0.027] (cross-checkpoint ranker), dKL -0.12 [-0.22, -0.03], dMQAScore +0.014 [-0.003, +0.030] -> QUALIFIES
dev__cfg7_best3_c8998 vs dev__cfg7: dCLAP +0.031 [+0.019, +0.042] (cross-checkpoint ranker), dKL -0.04 [-0.14, +0.07], dMQAScore +0.006 [-0.013, +0.024] -> QUALIFIES
dev__rescale12_best3_mqa vs dev__rescale12: dCLAP +0.003 [-0.010, +0.016] (circular), dKL -0.04 [-0.12, +0.04], dMQAScore pending -> MQAScore pending
dev__rescale12_best3_probe vs dev__rescale12: dCLAP -0.002 [-0.015, +0.011] (circular), dKL -0.04 [-0.12, +0.04], dMQAScore pending -> MQAScore pending

