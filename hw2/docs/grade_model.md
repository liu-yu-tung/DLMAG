# Listening-grade predictor

A small linear model that predicts which of two pieces the listener prefers, from piano-roll features of the MIDI. It does not replace listening tests: it explains grades for the report, screens candidate sets, and tracks training checkpoints between tests.

## Data

- **Blind A/B pairs:** every `results/listening/<session>.csv` with `source_a`, `source_b`, `pref` (written by `listening.py reveal`). `pref < 0` means A is better; `pref == 0` pairs are not used for fitting.
- **Pilot 1-5 scores:** every session CSV with `source`, `file`, `overall` (`candidates`, `ckpt`, `ckpt2`, `planfill`). Clips are paired within their session when the scores differ by 1 or more, weighted so each clip counts about once. The pilot rendering was flawed, so the pilot scores weigh less than their count suggests.
- **Tags:** A/B clips and the `candidates` session; used only to check the features (`results/grade_tags.csv`).

## Features (`src/hw2/grade_features.py`)

All tracks are merged first: single-stage pieces and real songs are one piano track while plan-fill has two, so a per-track feature would only identify the system. Bars come from the MIDI downbeats; the first 32 bars count. The clues came from the full piano rolls of every graded clip (2026-10-17):

| Clue in the rolls | Features |
|---|---|
| Second half changes texture or collapses into a one-note or one-bar loop | `half_pc_sim`, `half_rhythm_sim`, `half_register_shift`, `half_density_change`, `loop_run`, `one_note_bars` |
| Bass stops, notes drift above MIDI 84 | `bass_dropout` (bars whose lowest note is MIDI 60 or higher), `high_share` |
| Scattered notes in bars 1-4 | `opening_high_share`, `opening_density_ratio`, `skyline_leap` |
| Gaps, held notes that fade to silence | `longest_onset_gap_beats`, `onset_gaps_1bar`, `empty_bars` |
| Wrong notes | `key_fit` (share of note time in the best diatonic set), `key_fit_worst8` |
| Key change away from the opening or prompt key, clashing chords (T3 comments) | `opening_key_fit_min4`, `semitone_clash_share`, `restrike_share` |
| Others | `notes_per_bar`, `density_cv`, `mean_chord`, `velocity_spike_share`, `tempo`, `tempo_changes`, `duration_s` |

## Model (`scripts/grade_model.py`)

- **Score:** weighted sum of standardized features; P(A preferred) = sigmoid(score A − score B). Fit by L2-penalized logistic regression on feature differences (no intercept, so the A/B order does not matter).
- **Three feature sets:** `v1` was chosen from the rolls before any fit. `v2` swaps in the strongest single features from the first screen (`half_pc_sim`, `one_note_bars`); on T3, scored before refitting, it got 3 of 9 (`v1` 5 of 9). `v3` (built after T3) is `v1` plus the two harmony checks from the T3 comments: `opening_key_fit_min4` (fit of the opening key, or the prompt key in TASK B, in the weakest 4-bar window) and `semitone_clash_share` (share of onsets with two sounding notes a semitone apart). On T4, scored before refitting, `v3` got 6 of 7 (`v1` 5, `v2` 4), so `predict` uses `v3` since 2026-10-18 (`grade_pred_*` files without a variant suffix written before that are `v1`). It switches again only if a later session shows another set does better before refitting.
- **TASK B pieces:** features cover the 24 generated bars only (`first_bar` from the set's `gen_info.json`: a prompt file or folder means 8 prompt bars); the prompt sets the reference key.
- **Left out of the model:** `tempo_changes` and `velocity_spike_share` mostly tell real songs and the two systems apart.
- **`win_vs_real`:** sigmoid(score − mean score of the real anchors in `outputs/ref/`), the predicted chance that a piece beats a typical real opening.

## Checks

- **Leave-one-out:** each A/B pair is predicted by a model fit on everything else; each pilot session by a model fit without it (Spearman of score vs `overall`). Baselines: coin flip, "more notes wins". `results/grade_model_cv.csv`.
- **New session, before it is fit:** `grade_model.py check --session <id>` scores the new pairs with the models as they were, appends to `results/grade_model_check.csv`, then `fit` refits. This is the only unbiased test of `v2`.
- **Single features and tags:** `results/grade_single_features.csv` (how often the A/B winner has the higher value; mean within-session Spearman), `results/grade_tags.csv` (AUC of each feature for clips with vs without each tag).

## Where it runs in the pipeline

1. **After each graded session:** `listening.py import`, `listening.py reveal --session <id>`, `grade_model.py check --session <id>`, `grade_model.py fit` (`make grade-check SESSION=<id> grade-fit`).
2. **After each generated set:** `make grade-predict DIRS="outputs/<set>/*" NAME=<set>`, next to the H1/H4/GS and silence stats.
3. **Training and loss evaluation:** after a run, sample each loss-level checkpoint and score it. Fill model: the same 20 plans for every checkpoint (`make grade-sweep-fill RUN=<run>`); plan model: one fill checkpoint (`make grade-sweep-plan RUN=<run>`); main model: the existing sweep in `outputs/exp2/`. Output: `results/grade_pred_<run>.csv`, `_sets.csv`, `results/figures/grade_vs_loss_<run>.png` (predicted grade vs step and vs train and held-out loss).
4. **Listening test design:** pairs where the predictor is unsure or disagrees with the other metrics are the most informative to grade next.

## Limits

- About 13 decisive A/B pairs and 56 pilot clips; a leave-one-out accuracy on 13 pairs has an error bar of about ±25 points.
- The features see the MIDI, not the audio: rendering problems (a fading long note, loudness) are only partly covered by the gap features.
- The checkpoint decision rules stay with the listening tests; predicted grades are extra evidence. Picking submitted pieces by predicted grade would be cherry-picking and would have to be stated in the report.
