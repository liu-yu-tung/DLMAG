# Hypotheses

Guesses to test on train and validation only. No outside lookups about the dataset (see DECISIONS.md).
Status: open, supported, rejected, or unclear, with the evidence.

| ID | Hypothesis | How to test | Status |
|---|---|---|---|
| H1 | Decade is carried mostly by production: dynamic range, bass energy, brightness. | Per-decade plots of hand-crafted features on train; hand-crafted baseline accuracy on validation. | open |
| H2 | Decade errors fall mostly on neighboring decades. | Share of validation errors within ±1 decade, from the confusion matrix. | open |
| H3 | Market is carried mostly by the sung language. | Whisper language-ID feature alone vs MERT on validation; language vs market cross-table on train. | open |
| H4 | US vs UK is the main Task 2 confusion (both English). | Task 2 confusion matrix. | open |
| H5 | The dataset may contain different versions (covers) of the same song, so melody and harmony are weak cues for both tasks. Guess based only on the dataset's name "Discogs-VI"; not verified. | Compare chroma/harmony-only features with timbre features on validation; look for near-duplicate melodies across classes within train. | open |
| H6 | A class dominated by a few artists is easier to fit on train but generalizes worse (artist confound). | Spread within each class vs per-class validation accuracy. | open |
