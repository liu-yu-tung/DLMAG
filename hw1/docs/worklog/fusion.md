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
