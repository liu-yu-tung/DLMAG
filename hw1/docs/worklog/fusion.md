# Layer weighting and late fusion

Main session, 2026-09-29. Script: `scripts/bench_fusion.py`. Outputs:
- `results/fusion.json`, `results/tables/fusion.csv`
- `results/mert_layer_weights.png`
- `results/fusion_{A,B}_cm_norm.png`

## Method

- **Components**, each trained on train only:
  - hand-crafted logreg (30 s; for A also 10 s chunks);
  - MERT-v2 all-layer average: the mean log-prob of 24 per-layer logreg probes, so no layer is selected;
  - MERT-v2 learned layer weights (`hw1.layermix`): softmax layer weights + linear head, 300 full-batch AdamW epochs, dropout 0.3, weight decay 1e-2, averaged over 5 seeds;
  - CLAP audio-embedding logreg.
- **Fusion:** an equal-weight mean of log-probabilities. No weights or components are fitted on validation. The single best MERT layer is shown only as a validation-selected reference.

## Results (validation S; noise is about ±0.05)

| Method | A | B |
|---|---:|---:|
| MERT-v2 all-layer average + hand-crafted (A: 10 s chunks, B: 30 s) | **0.974** | 0.971 |
| MERT-v2 all-layer average | 0.947 | 0.990 |
| MERT-v2 single best layer (val-selected, reference) | 0.947 (L9) | 1.000 (L14) |
| MERT-v2 learned layer weights | 0.924 | 0.990 |
| MERT-v2 all-layer average + CLAP | 0.924 | **1.010** |
| hand-crafted 10 s chunks | 0.875 | not run |
| hand-crafted 30 s (logreg) | 0.792 | 0.618 |
| CLAP audio embedding | 0.561 | 0.603 |

## Findings

- **Averaging all 24 layer probes matches the best single layer** (A 0.947 = 0.947; B 0.990 vs 1.000) without choosing a layer on validation. This is the robust MERT choice.
- **Learned layer weights stay close to uniform** (0.03-0.05 each). They lean toward the upper half on B (layers 12-24 at about 0.05 vs 0.033 below), which matches the sweep: market information sits in the higher layers. They don't beat simple averaging on this data size.
- **Hand-crafted features add to MERT on A** (+0.027 with 10 s chunks) but not on B. The production cues seem complementary for decade; for market MERT already covers what they carry.
- **CLAP audio embeddings are weak** (0.56/0.60, above chance 0.417 but well below the hand-crafted features). Adding CLAP helps nowhere beyond noise. The lecture notes that CLAP aligns global semantics and misses fine detail. It's also possible the transformers 5.17 issue that breaks the CLAP text tower affects the audio tower too; not verified.
- **All differences among the top methods are within validation noise.** Pick by simplicity and prior reasoning, not by the last 0.01.

## Candidates for the submission (to confirm later)

- **A:** MERT-v2 all-layer average + hand-crafted 10 s chunks.
- **B:** MERT-v2 all-layer average; revisit after the language-ID feature.

## Rerun after the rotary fix (2026-09-29)

The tables above used the broken MERT features. After re-extraction (`bench_fusion.py`, same seeds):

| Method | A | B |
|---|---:|---:|
| MERT-v2 all-layer average | 0.943 | 1.034 |
| MERT-v2 learned layer weights | 0.939 | 1.020 |
| MERT-v2 all-layer average + hand-crafted 10 s (A) / 30 s (B) | 0.947 | 0.971 |
| MERT-v2 all-layer average + CLAP | 0.924 | 1.005 |
| MERT-v2 all-layer average + CLAP + hand-crafted | 0.974 | 0.946 |
| best single layer (val-selected, reference) | 0.920 (L9) | 1.039 (L21) |

- **B improved** (all-layer average 0.990 to 1.034); **A is about the same** (0.947 to 0.943). Learned weights on B now clearly favor layers 12-24 (about 0.05 vs 0.033 below).
- **The submission recipes are unchanged** under the selection rule in the plan (simplest recipe within 0.03 of the best): A = MERT average + hand-crafted 10 s (0.947; + CLAP 0.974 is within 0.03 and adds a weak, partly broken model), B = MERT average (1.034). `results/final_validation.json` matches these numbers.

## Task A ordinal check (2026-09-29, `scripts/ordinal_a.py`, `results/ordinal_A.json`)

- **Error structure (current A recipe):** mean absolute error 0.88 decades out-of-fold (5-fold on train, n=1026) and 0.82 on validation; a random guess gives 1.94. Of the wrong top-1 picks, 62% (OOF) and 61% (validation) are one decade off. Only 57-62% of top-3 sets are three adjacent decades.
- **Neighbour smoothing** of the probabilities, p'[k] = (1-2e) p[k] + e (p[k-1] + p[k+1]), strength chosen on the out-of-fold set: S rises from 0.892 (e=0) to 0.913 (e=0.25-0.3), then collapses above 0.35 (top-1 drops to 0.43). Validation at e=0.25: 1.000 (top-1 0.561, top-3 0.879) against 0.947.
- **Forced-adjacent top-3** (best run of three neighbouring decades): OOF 0.900 (+0.008), validation 0.955; no clear gain.
- **Rounded expected decade** as top-1: 0.423, worse than argmax (0.471).
- **Reading:** the OOF gain (+0.02) is under the 0.03 rule and the validation gain (+0.05) is inside its noise, so smoothing is not adopted yet. Recheck on the fused pool with the cross-validated selection.

## Cross-validated fusion selection (2026-09-29, `scripts/cv_fusion.py`, `results/cv_fusion.json`, `results/tables/cv_fusion_{A,B}.csv`)

- **Method:** 11 components (MERT all-layer average on mixture, vocals, accompaniment, drums, bass, other; hand-crafted; CLAP; mix balance; language ID from vocals and from mixture). Each gets 5-fold out-of-fold log-probs on train (1,026 clips for A, 798 for B) plus a train-only fit for validation. Every fusion of up to 4 components (equal-weight log-prob mean) is scored on the out-of-fold set. Rule: fewest components within 0.03 of the best out-of-fold S. Validation is reported only. Cached in `features/cv_{A,B}.npz`.
- **B:** the rule picks MERT mixture + language (mixture): out-of-fold S 1.023 against 0.964 for MERT alone, a paired difference of +0.059 (95% bootstrap CI 0.038 to 0.081). Validation agrees in sign (1.069 vs 1.034). Language alone scores 0.989 out-of-fold, higher than MERT alone (0.964). This overturns the earlier validation-only reading ("no gain beyond noise"): on 798 clips the language feature adds real signal.
- **A:** the rule picks MERT mixture alone (out-of-fold 0.905). The current recipe, MERT + hand-crafted 10 s, scores 0.892 out-of-fold, so hand-crafted adds nothing (difference for mixture alone: +0.013, CI -0.006 to 0.031). The best 4-component subsets reach only 0.916, inside the noise.
- **Stacker** (logistic regression on all 11 components' out-of-fold log-probs, C=0.01, cross-validated): A 0.945 out-of-fold and 0.966 validation, clearly above every equal-weight subset (0.916); B 1.006 and 1.039, below the B subset. Worth carrying into the next round for A.
- **Not yet decided:** the submission recipes are unchanged. Next: check the A stacker with a proper nested fit, refit choices (train only vs train+val), and add the CNN and MuQ components tonight.

## Stacker robustness check (2026-09-29, `scripts/check_stacker.py`; numbers are out-of-fold / validation S)

- **B, MERT + language (mixture) holds under every fusion method:** equal weight 1.023/1.069, temperature-calibrated 1.033/1.039, stacker 0.99-1.01. MERT alone is 0.964/1.034. Adopt.
- **A, most of the stacker gain is not fusion:** a stacker on MERT mixture alone (a 6-to-6 linear remap of its log-probs) gives 0.922/1.023 at C=0.01, against 0.905/0.943 without it. This is a class-bias and neighbour-decade correction, the same effect as the ordinal smoothing (+0.02 out-of-fold).
- **A, the all-11 stacker is sensitive to C:** 0.932 (C=0.001), 0.945 (0.01), 0.922 (0.1) out-of-fold. It would need Demucs, Whisper, CLAP and hand-crafted features at inference. Not adopted without a nested check.
- **Temperatures:** fitted on out-of-fold data, the MERT components need T of about 0.4 (over-confident after averaging 24 layer log-probs); language needs about 1.0-1.1. This is why equal-weight fusion lets MERT dominate.

## Nested remap check and recipe change (2026-09-29, `scripts/nested_remap.py`, `results/nested_remap.json`)

- Outer 5-fold on train; the remap (logistic regression on component log-probs, C picked by inner CV from 0.001/0.01/0.1) is fit on inner out-of-fold log-probs only.
- **A:** MERT alone 0.906 -> remap 0.932 (+0.026, CI -0.003 to 0.056). With hand-crafted: 0.892 -> 0.934. The non-nested stacker (0.945) was optimistic. C was always 0.1, the grid edge.
- **B:** MERT + language 1.023 equal weight vs 1.020 remap; no gain.
- **Recipe change (selection rule):** A = MERT all-layer average alone (hand-crafted dropped; remap under the 0.03 bar). B = MERT + Whisper language (mixture), equal weight. Validation: A 0.943, B 1.069. `predict.py` from audio matches the cached-feature predictions on all 234 test clips.

## Ordinal soft labels for the A MERT probe (2026-09-29, `scripts/ordinal_probe.py`, `results/ordinal_probe_A.json`)

- Soft-label cross-entropy via sample weights (each clip also counted as its neighbouring decades with weight eps). Same 5 folds; eps 0.1 fixed in advance.
- Out-of-fold / validation S: baseline 0.906 / 0.943; eps 0.1: 0.916 / 1.011 (diff CI -0.012 to 0.032); eps 0.05: 0.924 / 0.989 (CI 0.002 to 0.034); eps 0.2: 0.918 / 1.000.
- Mean error falls from 0.88 to about 0.81 decades. The gain is small (+0.01 to +0.02 out-of-fold) but has the same sign for every eps and on validation, and costs nothing at inference (same model shape).
- Not adopted yet: under the 0.03 bar. Recheck together with the CNN folds and the remap. Out-of-fold log-probs saved in `features/cv_A_ordinal.npz`.

## Signal outside MERT (2026-09-29, `scripts/non_mert.py`, `results/non_mert.json`; cached out-of-fold log-probs, no training)

- **Every non-MERT component is above chance** (S 0.417; all bootstrap CIs exclude it). A: hand-crafted 0.730, mix balance 0.675, language (mixture) 0.584, CLAP 0.567, language (vocals) 0.540. B: language (mixture) 0.989, language (vocals) 0.972, hand-crafted 0.576, mix balance 0.517, CLAP 0.516.
- **Language on A is above chance although almost every clip is English.** The 100-dim Whisper distribution must be reading recording or vocal character, not the language itself.
- **Unique contribution inside the non-MERT pool** (leave-one-out from the calibrated full fusion): A: hand-crafted +0.071 (CI 0.046 to 0.097) and mix balance +0.026 (0.004 to 0.049) are unique; CLAP and language are redundant. B: only language (mixture) is unique (+0.031).
- **Complementarity with MERT** (clips MERT gets wrong; uniform rank of the true class is 3.5, top-1 chance 0.167):
  - A (529 clips): every component ranks the true class better than uniform (2.95-3.26, CIs exclude 3.5), but top-1 is only 0.17-0.22. They order MERT's misses slightly better without flipping them, so equal-weight fusion barely moves S.
  - B (363 clips): language ranks the true class at 2.68 with top-1 0.31-0.33, twice chance; this is why MERT + language works. Hand-crafted, mix balance and CLAP are at uniform there: their signal is the part MERT already has.
- **Reading:** the non-MERT signals are real but redundant with MERT, except language on B. On A, hand-crafted and mix balance carry production information the other non-MERT features lack, but it overlaps MERT's. Use the "rank of the true class on MERT's misses" test as the cheap screen for new components (CNN, Demucs latent, Qwen).

## Demucs U-Net encoder features and CNN folds in the screen (2026-09-29, `scripts/cv_demucs.py`, `results/cv_demucs.json`, `scripts/non_mert.py`)

- **Demucs encoder features** (week 4 p.70 idea; 11 encoder layers of Hybrid Demucs, mean+std pooled, per-layer probe, all-layer average): A 0.852 out-of-fold / 0.939 validation, the best non-MERT component; deepest spectral layers fe4/fe5 0.764 alone. B 0.684: weak.
- **Short-Chunk CNN, 5 folds (seed 0):** A 0.813 out-of-fold.
- **They subsume the hand-crafted and mix-balance signal:** with Demucs in the pool, hand-crafted's unique contribution falls from +0.071 to about 0; Demucs keeps +0.029 (CI 0.009 to 0.050).
- **On MERT's A misses** they are the best of all components (true-class rank 2.77 Demucs, 2.80 CNN, against 3.5 uniform; top-1 0.23), yet MERT + either, calibrated, is not better than MERT alone (0.902 and 0.899 vs 0.906). MERT + Demucs + CNN: 0.910 (CI -0.022 to 0.032). Language on A lowers MERT (CI below zero).
- **Reading:** on A, three independent learned models (MERT, Demucs encoder, CNN) reach 0.81-0.91 and agree too much to help each other by averaging. The ceiling looks like label or data limits (clip-level decade ambiguity) more than missing features.

## CNN ordinal folds and A stacking check (2026-09-29, 16:10)

- **CNN full-train seeds (validation S):** A 0.826 / 0.754 / 0.799 (seeds 0-2), B 0.618 / 0.652 / 0.608. Five-fold out-of-fold: A 0.813, B 0.657.
- **CNN with ordinal soft targets (eps 0.1), 5 folds on A:** 0.793 alone (0.813 without). It ranks MERT's misses slightly better (true-class rank 2.70 vs 2.80, top-1 0.253 vs 0.233), but MERT + it is 0.896 (MERT alone 0.906).
- **Stacking the small A gains** (calibrated equal-weight, out-of-fold, CI of the difference against MERT alone): ordinal MERT 0.916 (-0.012 to 0.032); + Demucs 0.921 (-0.011 to 0.041); + Demucs + CNN 0.919; + Demucs + ordinal CNN 0.912. None clears the 0.03 rule or has a CI above zero. A stays MERT alone.
- **B:** the CNN and Demucs lower MERT (CIs below zero); only language helps.

## CNN without gain augmentation (2026-09-29 16:45)

- 5 A folds without the random +-6 dB gain: 0.824 out-of-fold alone (0.813 with it); MERT + it, calibrated, 0.917 (MERT alone 0.906; CI of the difference -0.014 to 0.038; with the gain-augmented CNN it was 0.899).
- Consistent with absolute level (mastering loudness) being a decade cue that the gain augmentation removes, but inside the noise; not adopted.

## MuQ-large (2026-09-29 17:00, `scripts/extract_muq.py`, `scripts/cv_muq.py`, `results/cv_muq.json`)

- **Porting:** muq 0.1.0 builds a transformers Wav2Vec2Conformer encoder from an EasyDict config. On transformers 5 it needed (1) `_attn_implementation = "eager"` set on that config and (2) hidden states rebuilt with forward hooks, because the transformers 5 encoder no longer returns them (layout kept as in transformers 4: input, layers 1..11, layer-normed layer 12). Features are finite and identical across two loads. Other transformers-5 changes inside the encoder cannot be ruled out; the scores below are well above chance, so the features are meaningful.
- **Out-of-fold / validation S (all-layer average probe):** A 0.867 / 0.883 (MERT 0.906 / 0.943); B 0.824 / 0.833 (MERT 0.964 / 1.034).
- **Per-layer validation S:** A 0.78-0.91 (flat, like MERT); B rises from 0.65 (input) to 0.88 at layer 6, then 0.72-0.82.
- **Fusion (calibrated, CI of the difference against the current recipe):** A MERT + MuQ 0.898 (-0.027 to 0.011); B current + MuQ 1.032 (-0.020 to 0.018). No gain; not adopted.
- Reading: in our setup MuQ is weaker than MERT-v2 on both tasks and redundant with it.
