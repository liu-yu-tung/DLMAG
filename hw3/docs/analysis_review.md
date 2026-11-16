# HW3 analysis review (review agent, 11-15 evening)

Scripts are in `scripts/analysis/review_checks/` (copied from the session scratch folder by main, 11-15 18:11; `.out` files are their printed results). The checks first run inline while plan mode blocked writes are now saved and re-run (11-15 evening, check-and-fix worker): `listening_check.py`, `mqa_bias.py`, `metric_agreement.py`, `paired_design.py`, `trajectory_windows.py`, `hub_kl.py`, `difficulty.py`, `pick_ties.py`, `setting_rule.py`, `apg_check.py`. Every number below comes from a run record or one of these. "Recorded" = the value in findings.md, status.md or knowledge.md.

## Verdict

- The main method result holds and is stronger than recorded: pooled over the 3 seed pairs, rescale12 beats cfg7 on CLAP and on KL. The KL gain is significant, where findings.md calls it n.s.
- Two recent claims do not hold. "rescale12's KL gain comes only from vocal captions" shows up on seed 0 only. "CLAP cannot separate real from generated" is true of mean cosine only.
- Best-of-3: the cross-checkpoint ranker gives the same independent-metric gain as the same-checkpoint one. Its CLAP gain on top of that is mostly a caption-independent shift.
- Submission risks: reproducing the best-of-3 picks (README, TF32 drift) and the wording on the test-time ranking rule. No rule breach found; the ranking is a grey area to state openly.

## Holds up

| claim | recorded | re-derived (script) |
|---|---|---|
| rescale12 dCLAP vs cfg7, seed 0 | +0.028 [+0.017, +0.039] | same; 1e5 resamples [+0.016, +0.039], sign-flip p < 1e-4, survives Holm over 48 tests (`stats_robust.py`) |
| rescale12 gain on seeds 1, 2 | +0.032, +0.023 | +0.032 [+0.020, +0.043], +0.023 [+0.007, +0.039]; pooled per prompt over 3 seed pairs +0.028 [+0.020, +0.035] |
| rescale lowers FD-MERT | 41.0 vs 43.4-45.7 | rescale12 seeds 41.0 / 41.2 / 41.5, all below every cfg7 seed (43.4 / 43.5 / 45.7) (metrics.json) |
| rescale lowers clipping | 0.068% vs 0.087% | lower on 3 of 3 seed pairs: 0.068 / 0.034 / 0.110 vs 0.087 / 0.196 / 0.289% (gen_info.json) |
| tags worse | dCLAP -0.051, dKL +0.43 | same; both survive Holm; paired FD-VGG +0.90 excludes 0 on 5 of 5 bootstrap seeds (`fd_robust.py`) |
| late guidance not free | -0.027, -0.075 | same; CLAP and KL survive Holm |
| early guidance (sigma > 110) free | +0.000 [-0.003, +0.005] | same; paired CLAP-audio cosine to cfg7 0.98 |
| apg12 worse FD-VGG | +0.36 [+0.11, +0.60] | excludes 0 on 5 of 5 bootstrap seeds |
| best3_c8998 vs rescale12 seed 0 | +0.015 [+0.003, +0.027] | same (`best_of_n.py`) |
| MQAScore separates real from generated | 0.646 vs 0.546 | paired +0.101 [+0.064, +0.136]; real higher on 78% of prompts |
| it is not only the vocal attribute | (vocal split) | real - cfg7 without vocal tags: +0.096 vocal group, +0.070 others (vocal_split_dev.csv) |
| SAO misses vocals | vocal attribute 0.22 vs 0.49 | real - cfg7 vocal attribute +0.271 [+0.180, +0.363], n = 50 |
| rescale12 CLAP gain equal in both vocal groups | +0.028, +0.027 | equal on every seed; interaction permutation p 0.71-0.99 (`vocal_interaction.py`) |
| guidance norm ratio | 0.22 early, < 0.02 late | cfg7 median: 0.228 (steps 0-20), 0.006 (steps 71-99); parallel fraction 0.19 early, 0.03 late |
| listening counts | s1 9 / 5 / 2, s2 9 / 6 / 1, B 10 of 14 and 10 of 15 | same; sign test p 0.42 and 0.61 |
| scheduler adds fresh noise each step | stated as mechanism | true: CosineDPMSolverMultistepScheduler, SDE variant with a Brownian-tree noise sampler (diffusers source). The washout itself is not tested |

## Does not hold or is overstated

1. **"rescale12's KL gain is n.s." (findings 12:20: -0.084 [-0.170, +0.003]).**
   - Seed 0 alone is fragile: its interval excludes 0 on 8 of 20 bootstrap seeds.
   - Seeds 1 and 2: -0.125 [-0.210, -0.038], -0.146 [-0.244, -0.051].
   - Pooled over 3 seed pairs: **-0.119 [-0.170, -0.068]**, sign-flip p 5e-5.
   - **Use:** "rescale12 improves KL-PaSST by 0.12 [0.07, 0.17] over 3 seed pairs."
2. **"rescale12's KL gain sits only in vocal captions" (status 17:40; it is the reason given for the 17:37 test KL prediction).**
   - Seed 0: vocal -0.188, others +0.019, interaction -0.206 [-0.371, -0.042], p 0.021.
   - Seed 1: -0.134 vs -0.117 (p 0.85). Seed 2: -0.133 vs -0.160 (p 0.79).
   - Pooled over 3 seeds: vocal -0.151, others **-0.086 [-0.161, -0.015]**, interaction p 0.22.
   - A corrected split (+39, 40, 6, 59, -94) gives p 0.10, a keyword split p 0.07.
   - **Use:** "the KL gain is somewhat larger on vocal captions, but it is present on the others too; the seed-0 pattern did not replicate." The test KL prediction's range may still be met, but not for the stated reason.
3. **"Vocal captions score lower on real clips too."**
   - True for MQAScore (-0.171, p < 0.001). The whole drop is the instrument attribute (-0.332, p < 0.001); genre, mood and context do not move. cfg7 shows the same instrument drop (-0.309).
   - Not true for CLAP: real vocal minus other -0.005 (p 0.83).
   - Vocal captions carry 2.6 more tags.
   - The bigger real-vs-generated gap on vocal captions (0.13 vs 0.07) is borderline: interaction +0.060, p 0.10 (+0.080, p 0.03 with the corrected split).
   - **Use:** "on vocal captions the judge doubts the listed instruments even on real clips; it is a judge-and-caption effect, not only a generation failure."
4. **The vocal split has extraction errors.**
   - Ids 39 and 40 ask for rap and singing but have no vocal tag.
   - Ids 6 and 59 mention female vocal samples (borderline).
   - Id 94 is an instrumental tagged "female vocal".
   - The keyword rule finds 61 captions, but 7 of its 12 extra hits are negations ("no voices").
   - Conclusions barely move with the corrected split (above). Report the split as "tag-based, about 3-5 of 100 misassigned".
5. **"CLAP cannot separate real from generated on dev."**
   - True of the mean own-caption cosine only (+0.004 [-0.023, +0.034]).
   - Generated audio is more generic. Its mean cosine to the other 99 captions is 0.061 against 0.040 for the real clips (paired -0.021 [-0.034, -0.008]).
   - The caption-specific margin (own minus other) nearly separates them: +0.024 [-0.001, +0.054]. R@1 does separate them (real 0.40 vs 0.21-0.28 over the cfg7 seeds, extra_metrics.csv; no interval computed).
   - **Use:** "CLAP's absolute cosine is inflated by generic audio; its caption-specific margin and retrieval rank do separate real from generated" (`clap_margin` check).
6. **"MQAScore is the more discriminative text-match metric."**
   - It does separate real from generated. The gap comes from genre +0.244, mood +0.172, vocal +0.271 and context +0.055; instrument is +0.025 [-0.025, +0.073], n.s.
   - It has a caption-length bias: Spearman with tag count -0.24 to -0.34 in every run, the real clips included (all p <= 0.017).
   - It is not loudness-biased (|rho| with RMS, peak, clipping <= 0.15, all n.s.).
   - **Use:** "more discriminative between real and generated, mainly on genre, mood and vocals; scores fall with caption length."
7. **"Clipping is about 3x higher at scale 12" (findings, "met" item 1).**
   - cfg12 0.281% sits inside the cfg7 seed range (0.087-0.289%).
   - Loudness is the cleaner signal. cfg12 is -16.7 dBFS, louder than every cfg7 seed (-17.4 / -17.5 / -18.1).
8. **"APG12 keeps about 70% of cfg12's CLAP gain."**
   - apg12 - cfg12 is -0.004 [-0.018, +0.011], so they are not separable.
   - apg12's own gain over cfg7 (+0.011, p 0.020) does not survive Holm.
   - **Use:** "APG12's text match is not separable from CFG12's. Its output stays closer to cfg7 (CLAP-audio cosine 0.91 vs 0.81 for cfg12) at cfg7 loudness (-17.6 dBFS), and it fails the FD-VGG check."
9. **Single-test claims that do not survive a multiplicity correction** (48 tests vs cfg7, Holm): cfg12 dCLAP (raw p 0.034), apg12 dCLAP (0.020), cfg3 dKL (0.010), interval_early30 dKL (0.025), tags_oracle dKL (0.007), rescale12_phi1 dKL (0.003).
   - cfg12 dCLAP's interval excludes 0 on 19 of 20 bootstrap seeds.
   - cfg12's paired FD-VGG lower bound is -0.030 to 0.000 over 5 bootstrap seeds (excludes 0 on 1 of 5), so its FD pass is a coin toss.
   - **Use:** state "95% intervals, no multiplicity correction" once, and mark which claims survive Holm.
10. **"The tie-break by 'improve' makes rescale12 the dev pick" (user 16:15).**
    - Applied as written at the setting level, rescale18 improves on rescale12: CLAP +0.002, KL -0.105 [-0.180, -0.029], FD-VGG +0.035 [-0.296, +0.287].
    - rescale12_phi1 also improves on rescale12: CLAP +0.006, KL -0.045, FD-VGG -0.305 [-0.529, -0.013], which excludes 0 on 5 of 5 bootstrap seeds.
    - The submission itself (best3_c8998) does improve on both under the rule: vs phi1, dCLAP +0.009 and dKL -0.08 (point), FD-VGG +0.199 [-0.217, +0.554]; vs rescale18, FD-VGG -0.142 [-0.459, +0.250].
    - **Use:** "rescale12 was kept because only it had 3 seeds and the best-of-3 pool; rescale18 and phi 1 are single-seed ablations that look at least as good." Do not write that the rule alone picks rescale12.
11. **Listening, session 1 KL agreement.**
    - The pairs were drawn at random (`rng.sample`), not by metric, so this is not a sampling artefact.
    - Against the marginals it is above chance: 13 of 14 agree vs 7.9 expected under independence, Fisher p 0.005; KL-PANNs 14 of 14, p 0.0005.
    - In session 2 every metric is at chance (Fisher p >= 0.49).
    - It is one listener, 8 metric-session tests, and clips loudness-matched to -18 LUFS.
    - **Use:** "exploratory: in the cfg7 vs pick session the listener's choices tracked KL to the real clip, which they never heard; in the seed vs best-of-3 session nothing tracked."
    - Side bias: B was chosen 20 of 29 decided pairs (p 0.06). The second run was chosen 6 of 7 times when on B and 3 of 7 when on A (session 1). Sides were balanced, so this adds noise, not bias.

## Risks for the submission

1. **Test-time ranking vs "Do not train or tune on validation or test examples"** (Homework_3.pdf).
   - The ranking uses no test reference and no training. It was fixed on dev before the test runs, and the ranker is a different checkpoint.
   - But it picks per test item by a near-copy of a reference-free test metric. Per-item correlation between the checkpoints is 0.49-0.65, and picks agree on 53% vs 33% chance.
   - The test reference clips were also retrieved with the 90.14 checkpoint, so test KL is not independent of CLAP either.
   - **Action, no change to the submission:** in the report, (a) show test seed 0 of rescale12 (no selection) next to best-of-3 as the method result; (b) call the best-of-3 CLAP gain "selection-inflated" and show the margin split (missed insight 1); (c) describe it as inference-time reranking with the ranker, its checkpoint and its citation; (d) say which test metrics it can touch. If the user reads the rule strictly, test seed 0 is the fallback and is already generated.
2. **Reproducing the picks.**
   - `docs/readme_draft.txt` does not describe the submission path: rescale phi 0.7 at scale 12, seeds 0-2, `clap_rescore.py` with the 89.98 checkpoint, `select_best_of_n.py --score clap8998_cosine`.
   - `docs/clap_rescore_state.txt` records CPU fp32 vs GPU TF32 score differences of up to 0.022. On dev, 31 of 100 picks have a top-2 gap under 0.022 (10 under 0.01, minimum 0.0021).
   - **Action:** ship the test `selection.csv` in the release; give the exact command chain in the README; state that rescoring on other hardware can flip near-tie picks; cite the 89.98 checkpoint.
3. **Selection wording** (Does not hold, item 10).
   - **Action:** state both definitions with their times: "improve" 11-15 11:50, tie broken by "improve" 16:15. Both were set after dev results were seen.
4. **Test KL prediction.** Its reason (vocal-only KL gain) is wrong.
   - **Action:** if it is met, do not present a flat test KL as "expected because the gain is vocal-only". Report the pooled dev no-vocal KL gain (-0.086) instead.
5. **MQAScore judge and MusicCaps.** Whether Music Flamingo saw MusicCaps clips or captions in training is not checked. If it did, the real-clip score is inflated.
   - **Action:** one line, "not checked", next to the real-vs-generated result.

## Missed insights (ranked by value for the report vs cost)

1. **What the CLAP gain is made of** (value high; done, figure about 15 min; `clap_margin` check). Split each CLAP change into own-caption cosine, mean cosine to the other 99 captions, and the margin between them.
   - rescale12 vs cfg7, pooled over 3 seeds: own +0.028, margin **+0.015 [+0.009, +0.021]**. About half of the gain is caption-specific.
   - best3_c8998 vs rescale12: own +0.015, other +0.012, margin **+0.003 [-0.009, +0.015]**. The submission's extra CLAP is mostly generic.
   - Same-checkpoint best3 vs rescale12: margin +0.030 (selection bias).
   - cfg12 vs cfg7: margin +0.012 of +0.015.
   - real vs cfg7: other -0.021, margin +0.024.
2. **Pooled seeds** (value high; done; `stats_robust.py`, `vocal_interaction.py`): the rescale12 KL gain is real (-0.119 [-0.170, -0.068]), and the vocal interaction is a seed-0 artefact (above).
3. **Both rankers give the same independent gain** (value high; done; `best_of_n.py`). rescale12, against the seed mean (= a random pick):

   | ranker | dCLAP90.14 | dCLAP89.98 | dKL-PaSST |
   |---|---|---|---|
   | same-checkpoint | +0.047 | +0.025 | -0.104 [-0.156, -0.050] |
   | cross-checkpoint | +0.020 | +0.047 | -0.088 [-0.149, -0.027] |

   - MQAScore vs seed 0: +0.011 and +0.014.
   - Each checkpoint gets back about half of the other's self gain, which matches the slope of the seed-to-seed differences (0.54).
   - The extra +0.027 CLAP of same-checkpoint ranking is pure selection bias.
   - No seed clustering: picks per seed 39 / 37 / 24 (chi-square p 0.14).
   - Report best-of-N against the seed mean, not seed 0.
4. **Hub references make test KL easier** (value medium-high; done; `hub_kl` check).
   - 39 test captions have a reference clip that serves 3 or more captions. Their KL is 0.82 / 0.65 / 0.75 vs 1.04 / 1.07 / 1.05 for the rest (cfg7 seeds 0-2).
   - Spearman of KL with the clip's k-occurrence: -0.25 / -0.37 / -0.33 (p <= 0.012).
   - Test KL partly measures how generic the retrieved reference is.
5. **MQAScore biases and per-attribute gap** (value medium-high; done; `mqa_bias` check): length bias, instrument doubt on vocal captions, gap from genre, mood and vocals (above).
6. **Shared initial noise is worth about 1.9x the prompts** (value medium, design slide; done; `paired_design` check).
   - Same-seed pairs vs different-seed pairs: SD of the per-prompt difference 0.065 vs 0.090 for CLAP (0.73x), 0.46 vs 0.59 for KL (0.79x).
   - Per-prompt correlation cfg7 vs rescale12: 0.80 with the same seed, 0.64 with a different seed.
7. **No "helps hard prompts" effect** (value medium; done; `difficulty` check).
   - rescale12's per-prompt gain correlates -0.39 with cfg7 seed-0 CLAP (shared noise, regression to the mean), but -0.01 with cfg7 seeds 1-2.
   - Prompt explains 67% of per-item CLAP variance over 6 runs; mean rank correlation between runs is 0.62.
   - Generated per-prompt CLAP does not track the real clip's (Spearman +0.08). MQAScore does (real vs cfg7 0.58).
   - 6 of the 10 hardest prompts are vocal captions; length and quality words do not matter.
8. **Metrics agree on means, not per prompt** (value medium; done; `metric_agreement` check).
   - Within a run, Spearman CLAP vs MQAScore is +0.16 to +0.28.
   - On per-prompt changes between runs, CLAP vs MQAScore is +0.09 to +0.32 and CLAP vs KL is -0.02 to -0.28.
9. **phi 1 looks at least as good as phi 0.7** (value medium; done).
   - Single seed: CLAP +0.006, KL -0.045, FD-VGG -0.305 [-0.529, -0.013], FD-MERT 39.7 vs 41.0, clip 0.038 vs 0.068%.
   - Its prediction missed. Report it as "the best phi is likely 1; not adopted because the pick was fixed".
10. **Rescale's std includes the padding** (value medium; 5 min to state, a GPU run to test, not now).
    - `rescale_to_conditional` takes the std over all 1024 latent frames. Only 216 (21%) carry the 10 s clip.
    - The guidance difference sits entirely in those frames: valid share 1.00 at every step (trajectory.jsonl).
    - So the std ratio is diluted toward 1, and rescale acts weaker than phi says. That is a candidate reason phi 1 is not too strong. Untested.

## Suggested slide changes

- **Rescale slide:** use the 3-seed pooled numbers (dCLAP +0.028 [+0.020, +0.035], dKL -0.119 [-0.170, -0.068]), the FD-MERT seed ranges and the phi 1 result.
- **Best-of-N slide:** per ranker, show own / other / margin CLAP and KL vs the seed mean. The message: "same-checkpoint excess = selection bias; cross-checkpoint gain mostly generic".
- **Vocal slide:** drop "KL gain only in vocal captions"; keep "CLAP gain equal across groups"; note the tag-split errors.
- **MQAScore slide:** per-attribute real-vs-generated gaps, length bias, the instrument drop on vocal captions; replace "CLAP cannot separate" with the margin and R@1 result.
- **Evaluation slide:** the shared-noise design (intervals 0.73x as wide); "48 tests, no correction", and which claims survive Holm.
- **Dev vs test slide:** the hub effect on test KL; test references come from the scoring checkpoint; show dev-test agreement as a scatter of paired differences with intervals, since a rank correlation over about 6 settings says little.
- **Selection slide:** both definitions with their times, and why rescale12 rather than rescale18 or phi 1.
- **APG and listening:** drop "70% of the gain". Listening: n.s. preference, side bias 20 of 29, KL agreement in session 1 marked exploratory.

## Test check (rescale12 test evals landed during the review; read only)

- rescale12 - cfg7, seed-matched on test:
  - CLAP +0.010 [-0.000, +0.021] / +0.007 [-0.004, +0.018] / +0.014 [+0.005, +0.023]
  - pooled **+0.010 [+0.005, +0.016]** (dev pooled +0.028)
  - KL-PaSST pooled **-0.023 [-0.059, +0.015]** (dev pooled -0.119)
  - caption-specific margin pooled +0.006 [+0.001, +0.011] (dev +0.015)
- So the gain transfers to test at about a third of its dev size, and KL does not move.
- The seed-0 CLAP prediction sits at the low edge of its range (+0.010). The KL prediction is likely met, but its stated reason is not why.
- **Use:** "the rescale gain is smaller on LLM-style test captions with retrieved references."
- test best3_c8998 per_prompt.csv had not been written yet: not checked.

## Not checked

- Test best-of-3 results and test ablations (pending in the queue when the review ended).
- Paired FD-MERT (no per-run MERT cache).
- Music Flamingo training data.
- The washout mechanism.
- Rescale with a valid-frame std.
- MQAScore of rescale18 / phi1 / seeds 1-2 (not run).

## Errata from re-running the checks as scripts (main, 11-15)

The inline checks were saved as scripts in `scripts/analysis/review_checks/` (commit 906fb0c) and re-run. Four numbers above differ; use these values:

- **Late guidance norm ratio (cfg7, steps 71-99):** median 0.004, not 0.006 (0.006 is the median over steps 65-99).
- **Mean rank correlation between the 6 runs:** 0.65 (range 0.53-0.86), not 0.62.
- **Hardest 10 prompts that are vocal captions:** 7 by mean CLAP over the 6 runs; 6 only when ranked by the real clips' CLAP.
- **MQAScore vs tag count:** Spearman -0.24 to -0.41 over all 9 runs (best-of-3 runs -0.30, -0.35, -0.41; the other six -0.24 to -0.33); all p <= 0.017.

Wording: "prompt explains 67%" is the two-way ICC(3,1) (plain share of variance in prompt means: 71%); the real-vs-generated CLAP +0.08 is a Pearson correlation with the 6-run mean (Spearman +0.07; vs cfg7 alone -0.04).

New: on test, 41 of 100 best-of-3 picks have a top-2 gap under 0.022 and 24 under 0.01 (minimum 0.0002); the two checkpoints pick the same seed on 53% of items, as on dev (`pick_ties.out`).
