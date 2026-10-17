# HW2 status

As of 2026-10-17 11:00. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages) is 100% of the grade. Evidence: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, single-stage (`gpt2_ppb16`):** checkpoint `loss_0.40` (T1 2026-10-17: 3 wins, 1 loss, 1 same vs `step_10000`). Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK A, plan-fill:** plan `plan_rep/best` or `plan_rep_long/loss_0.60`, fill `fill_rep_long/loss_0.40` or `loss_0.60`; melody-first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`. Six candidate sets of 20 are in `outputs/cand_a/` (3 single-stage, 3 plan-fill) with the fixed guide; stats in `results/cand_a_*.csv`.
- **TASK A decision (T2, 2026-10-17, `results/listening/t2_openings.csv`):** plan-fill won 3 of 6 pairs against single-stage (2 losses, 1 same), below the 5-of-6 rule, so single-stage `loss_0.40` is submitted; plan-fill stays an extra experiment. Best config `t1 p1` (`outputs/cand_a/single_t1_p1`, the other two configs `t1 p0.95`, `t0.9 p1` are in `outputs/cand_a/`).
- **TASK B:** must use the TASK A checkpoint (spec page 16); final set `outputs/task_b_final` (3 prompts x 3 configs, 8 + 24 bars). **T3 decision (2026-10-17, `results/listening/t3_continuations.csv`):** `t1 p0.95` won both of its pairs on every prompt (6 of 6), so it is the best config for all three prompts (`outputs/task_b_final/t1_k0_p0.95`).
- **BONUS TASK:** `abl_melody/step_7000`, ablations `abl_lookahead`, `abl_chords`. Done.
- **Held-out songs:** 45 of 896 are held out; the report must say so.

## Findings that matter

1. **H1/H4/GS do not separate anything:** every model and setting lands at the real level (H4 2.5-2.7, GS 0.95-0.97); shuffling a real piece's bars leaves them unchanged. Required in the report, but structure needs other measures.
2. **Loss vs listening:** more training lowers held-out likelihood but makes samples closer to real songs at t1 p1. Training longer adds no 4/8-bar repetition.
3. **POP909 traits:** sparse intros (bars 1-4: 14.4 notes per bar vs 20.8), melody enters around bar 8, melody repeats at lags 2/4/8/16 (exact melody returns, varied piano).
4. **Plan-fill:** follows planned chords (0.80-0.82 vs 0.85 real) and density bins; writes an asked repeat in 14-22% of bars even with the repeat bias.
5. **Rendering:** one-keyboard rendering fixed the doubled-note loudness; long notes fade because the piano decays (not a bug). Melody stem vs piano stem loudness matches real songs (`results/stem_gap.csv`).
6. **Bar-end silence:** plan-fill with some plan models has more silent share and empty bars than real; `oldplan_fill040` is closest to real.
7. **Loss review:** the fill model reads its labels (true 1.16 / removed 1.28 / shuffled 1.72 melody loss); `loss_0.40` held-out loss is 2.70 vs 1.44 at `step_6000`.
8. **The four pilot listening sessions are a pilot only** (flawed rendering, few clips, rubric changed).

## Open issues

- Loud sparse or high clip, empty bar 2, "broken piano" re-listen (`docs/listening_issues.md` section 11).
- Sudden loud notes in plan-fill (8.7% of notes vs 5.0% single-stage, 6% real); the `min_gap` melody lift in the listening build raised it to 9.6%. Keep or remove the lift: decide before final WAV rendering.
- The submitted TASK A WAVs are rendered without the real-velocity map; whether they need it is part of the volume decision above.

## Done 2026-10-16/17

- Repeat-guide note clusters fixed; fixed-guide re-run in `outputs/plan_fill_rep_fix/*`.
- Longer training: `plan_rep_long` (12k steps), `fill_rep_long` (30k steps).
- Candidate sets regenerated (`outputs/cand_a/*`); T1 graded; T2 built.

## Plan

1. T3 graded 2026-10-17 17:50: TASK B best config `t1 p0.95` for all prompts (see Systems). Metrics in `results/task_b.csv`, `task_b_fade.csv`; element marks in `results/listening/t3_continuations_elements.csv`.
2. Final TASK A is `outputs/cand_a/single_t1_p1` (20 MIDI + 20 WAV + metrics).
3. Plan-fill matters in the report even though it is not submitted (user, 2026-10-17): keep testing and improving it.
   - Longer training (user, 2026-10-17 15:10): `plan_rep_24k` (24k steps) and `fill_rep_60k` (60k steps, the same length and schedule as `gpt2_ppb16`), done 19:18.
   - Grade sweeps done 21:03 (`make grade-sweep-fill RUN=fill_rep_60k`, `make grade-sweep-plan RUN=plan_rep_24k`); findings section "Predicted grade vs loss for the longer plan-fill runs".
   - T4: the last checkpoint of each new run against `pf_oldplan_fill040` (2 pairs each; the fill pairs share the plan), and the three single-stage configs against each other (3 pairs).
   - Fill model on real held-out plans (separates plan errors from fill errors; never done).
   - Cause of the sudden loud notes in plan-fill (see open issues).
   - Melody-repeat following (14-22% of asked repeats written) and bar-end silence.
4. Listening-grade predictor (user, 2026-10-17; `docs/grade_model.md`): fit on T1, T2 and the pilot; after each graded session run `grade_model.py check` before `fit` (T3: `v1` 5 of 9, `v2` 3 of 9; `v3` built after T3, first tested on T4); score each new set; predicted grade vs loss for `exp2` (main model) and the new plan-fill runs.
5. Extras: NLL matching of sampling settings, loss vs listening figure (partly covered by item 4), SI feasibility.
6. Packaging: checkpoint defaults in the `Makefile` and `generate_plan_fill.py`, `submission` branch, fresh-venv README test. Code, checkpoints and README go on the cloud drive (the user does this).
7. Report (PPT-style PDF, about 10 pages; name and ID on the cover).
