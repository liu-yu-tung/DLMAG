# HW2 status

As of 2026-10-17 01:00. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages) is 100% of the grade. Details behind each point: `results/findings.md` (evidence), `docs/listening_issues.md` (listening comments), README (pipeline).

## Systems and checkpoints

- **TASK A, single-stage (`gpt2_ppb16`):** not chosen. Val loss is lowest at step 7000 (1.41) and rises after. `loss_0.40` (step 28,658, val 2.73) was the pick from the pilot listening; `step_10000` (val 1.50) matches real songs as well or better on H1/H4/GS, silence and copy rate and was never listened to. `step_6000` at t1 p1 leaves gaps (1.4 empty bars per piece vs 0.04-0.06 real).
- **TASK A, plan-fill (`plan_rep/best` step 3600, val 1.187; `fill_rep/best` step 6000, val 1.354):** melody-first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8` (with or without `--plan_repeat_bias 4`), t1 p0.95. The candidate outputs are invalid until regenerated (repeat-guide bug below).
- **Checkpoint rules differ:** the main model is picked far past its val minimum, plan and fill at theirs. Any comparison of the two systems should pick both by the same benchmark (sampled pieces vs real songs, then listening).
- **TASK B:** must use the TASK A checkpoint (spec page 16). `outputs/task_b` used `step_6000`, so it is regenerated after the decision. Plan-fill TASK B runs exist; the repeat-bias ones need regenerating.
- **BONUS TASK:** `abl_melody/step_7000`, ablations `abl_lookahead`, `abl_chords`. Done.

## Findings that matter

1. **H1/H4/GS do not separate anything:** every model and setting lands at the real level (H4 2.5-2.7, GS 0.95-0.97); shuffling a real piece's bars leaves them unchanged. Required in the report, but structure needs other measures.
2. **Loss vs listening:** more training lowers held-out likelihood but makes samples closer to real songs at t1 p1. Training longer adds no 4/8-bar repetition; no main checkpoint has it.
3. **POP909 traits:**
   - Intros are sparse: bars 1-4 have 14.4 notes per bar vs 20.8 after, and 30% of songs start with an empty or pickup bar. Generated pieces copy this; the old real anchors (mid-song crops) did not have it.
   - Melody enters around bar 8 (median).
   - Melody repeats at lags 2/4/8/16. The melody comes back exactly while the piano varies.
4. **Plan-fill:**
   - The fill model follows the planned chords (0.85, same as real) and density bins.
   - It ignores repeat labels when sampling (writes the asked repeat 4.3% of the time) although it copies under teacher forcing.
   - The repeat bias brought lag 2/8 melody repeats to the real level; those numbers need a re-run.
5. **Rendering:**
   - The same pitch in MELODY and PIANO at once played 6 dB louder; fixed for all WAVs (`hw2.render.one_keyboard` in `midi_to_wav`).
   - Long notes fade because the piano decays (middle C -35 dB in 4 s); not a bug.
6. **Bar-end silence:** plan-fill at truncated sampling has 4.5-5% silent share vs 1.4% real, concentrated before barlines. It is in the notes, not the audio. Cause unknown.
7. **The four pilot listening sessions are a pilot only:** flawed rendering, 2-4 clips per source, the rubric changed between sessions. The plan-fill clips carried loud doubled notes and note clusters, so they are biased against plan-fill. Large gaps still hold (real songs above every system; early checkpoints far below).

## Bugs fixed 2026-10-16/17, outputs not redone

- **Repeat-guide note clusters (`RepeatGuide`):** re-pitched notes were never used up, so the melody stacked up to 32 notes on one onset. Fixed, plus a rule that a part cannot strike the same position and pitch twice in a bar. CPU test: at most 6 notes on one onset vs 32 before. To redo: every run with repeat bias 8 or more (4-17 of 20 pieces had clusters).
- **Doubled notes across tracks:** fixed in rendering. To redo: existing WAVs (rendered before 2026-10-16 23:00).
- **Piano roll ahead of the audio on the listening page:** fixed and published.

## Decisions (hands-off mode 10-16 to 10-20, fixed rules in the approved plan)

- Single-stage checkpoint: `loss_0.40` (T1 2026-10-17: 3 wins vs 1 for `step_10000`, 1 same).
- TASK A system (single-stage or plan-fill): T2, after `plan_rep_long` / `fill_rep_long`.
- Sampling fixes 2026-10-17: no `Pos_` past the meter, one opening tempo (`docs/listening_issues.md` section 11).
- Checkpoint defaults in the Makefile and `generate_plan_fill.py`: set after T2.

## Plan (proposed; the user has midterms on 10-17/18)

1. **Design doc** `docs/design.md`: POP909 traits, hypotheses, what each stage is responsible for, evaluation ladder, status of each hypothesis.
2. **Regenerate** the repeat-bias runs with the fixed guide, re-run the stats, update the `findings.md` tables. About 7 minutes of GPU per 20 pieces.
3. **Loss analysis** on existing checkpoints:
   - Count only sampled tokens, per type (`Pos`, `Pitch`, `Vel`, `Dur`) and per part.
   - Use all 45 held-out songs.
   - Measure the gap between true and shuffled labels (does the fill model use them).
4. **Longer training:** plan about 15 minutes for 3x the steps; fill about 1.8 h to reach the main model's training-loss level (estimate from 0.22 s/step).
5. **Stage tests:** the fill model on real held-out plans (never done; separates plan errors from fill errors); melody-first vs piano-first with the current models.
6. **Find the bar-end silence cause.**
7. **Scorecard:** one table per folder with H4, GS, silent share, notes per bar, melody exact match at lag 4/8, theme return, chord match, label hit rate, copied 8-note runs, each next to the real value.
8. **One clean A/B listening session:** system decision and main checkpoint, picked by the same benchmark, with real song openings as anchors.
9. **Final generation:** TASK A 20 pieces plus WAV; TASK B 3 prompts x 3 configs with the TASK A checkpoint; re-render WAVs; prepare the submission branch.
10. **Report.** The outline was planned for about 10-21.
