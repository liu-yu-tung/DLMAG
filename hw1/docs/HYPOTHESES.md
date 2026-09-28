# Hypotheses

Guesses to test on train and validation only. No outside lookups about the dataset (see DECISIONS.md).
Status: open, supported, rejected, or unclear, with the evidence.

| ID | Hypothesis | How to test | Status |
|---|---|---|---|
| H1 | Decade is carried mostly by production: dynamic range, bass energy, brightness. | Per-decade plots of hand-crafted features on train; hand-crafted baseline accuracy on validation. | supported (hand-crafted): the energy group alone reaches validation S 0.761 vs 0.792 for all features. RMS rises about 3.5 dB after the 1980s, dynamic range drops 1.7 dB, the sub-60 Hz share rises from 2.3% to 8.5%, and brightness peaks in the 1980s-90s. See `worklog/handcrafted.md`. |
| H2 | Decade errors fall mostly on neighboring decades. | Share of validation errors within ±1 decade, from the confusion matrix. | partly supported (hand-crafted): 47.6% of errors are ±1 decade vs 33% for random wrong guesses. The main blocks are 2000s/2010s and 1960s/1970s, but the mean error distance is still 1.93 decades. |
| H3 | Market is carried mostly by the sung language. | Whisper language-ID feature alone vs MERT on validation; language vs market cross-table on train. | supported (9/29): Whisper language log-probs alone reach validation S 0.980 on B (MERT 1.034; chance 0.417); per-market vocal-stem top-1 language: Brazil pt 117/133, Italy it 57, Spain es 68, Germany de 22 (most German clips read as en). Fusing with MERT adds nothing beyond noise (1.069). Earlier hint: MERT-v2 lifts B from 0.681 (hand-crafted) to 0.92-1.00 in layers 9-24, while its early, acoustic layers (1-5) stay at the hand-crafted level (about 0.67). This fits the market signal living in higher-level content such as vocals and language, but language isn't isolated yet. |
| H4 | US vs UK is the main Task 2 confusion (both English). | Task 2 confusion matrix. | supported one way (hand-crafted): 9/17 US clips go to UK and US is never predicted. But Spain and Germany are also mostly sent to UK or Brazil, so UK absorbs several classes. |
| H5 | The dataset may contain different versions (covers) of the same song, so melody and harmony are weak cues for both tasks. Guess based only on the dataset's name "Discogs-VI"; not verified. | Compare chroma/harmony-only features with timbre features on validation; look for near-duplicate melodies across classes within train. | open |
| H6 | A class dominated by a few artists is easier to fit on train but generalizes worse (artist confound). | Spread within each class vs per-class validation accuracy. | open |

## H7: mixing conventions separate decades (and maybe markets)

- **Claim:** per-stem level, dynamics and brightness (vocal level over the band, drum and bass prominence, per-stem compression) change by decade, and these are easier to read from Demucs stems than from the mixture.
- **Test:** probe the 29 mix-balance features and per-stem MERT embeddings on train, score on validation; compare mixture vs vocals vs accompaniment inputs (the HW's optional experiment).
- **Status (9/29):** partly supported. On A, drums share rises from 0.08 (1960s) to 0.21 (1980s-90s), the `other` stem's crest factor drops about 1.5 dB in the 2000s-2010s, and bass level variability falls from 4.9 to 3.6-3.9 dB. But the 29 features alone reach only S 0.640 (A) and 0.637 (B), and adding them to the recipe gives 0.947 (A, no change) and 1.073 (B, +0.039, inside the noise). Stem inputs to MERT: the accompaniment carries A (0.947), the vocals carry B (1.015). Details in `docs/worklog/stems.md`.
