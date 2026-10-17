# HW2 status

As of 2026-10-17 11:00. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages) is 100% of the grade. Evidence: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, single-stage (`gpt2_ppb16`):** checkpoint `loss_0.40` (T1 2026-10-17: 3 wins, 1 loss, 1 same vs `step_10000`). Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK A, plan-fill:** plan `plan_rep/best` or `plan_rep_long/loss_0.60`, fill `fill_rep_long/loss_0.40` or `loss_0.60`; melody-first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`. Six candidate sets of 20 are in `outputs/cand_a/` (3 single-stage, 3 plan-fill) with the fixed guide; stats in `results/cand_a_*.csv`.
- **TASK A decision (T2, 2026-10-17, `results/listening/t2_openings.csv`):** plan-fill won 3 of 6 pairs against single-stage (2 losses, 1 same), below the 5-of-6 rule, so single-stage `loss_0.40` is submitted; plan-fill stays an extra experiment. Best config `t1 p1` (`outputs/cand_a/single_t1_p1`, the other two configs `t1 p0.95`, `t0.9 p1` are in `outputs/cand_a/`).
- **TASK B:** must use the TASK A checkpoint (spec page 16); `outputs/task_b` used `step_6000`, so it is regenerated after the decision (3 prompts x 3 configs, 8 + 24 bars; best per prompt by T3 wins, tie to the lower temperature).
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

1. T3 (`t3_continuations`, TASK B: 3 prompts x 3 configs, 9 pairs, with the new element panel); `outputs/task_b_final` (configs `t0.9 p1`, `t1 p0.95`, `t1 p1`, metrics in `results/task_b.csv`, `task_b_fade.csv`); best config per prompt by wins.
2. Final TASK A is `outputs/cand_a/single_t1_p1` (20 MIDI + 20 WAV + metrics).
3. Plan-fill matters in the report even though it is not submitted (user, 2026-10-17): keep testing and improving it.
   - T4: more plan-fill vs single-stage pairs, and the three single-stage configs against each other.
   - Fill model on real held-out plans (separates plan errors from fill errors; never done).
   - Cause of the sudden loud notes in plan-fill (see open issues).
   - Melody-repeat following (14-22% of asked repeats written) and bar-end silence.
4. Extras: NLL matching of sampling settings, loss vs listening figure, SI feasibility.
5. Packaging: checkpoint defaults in the `Makefile` and `generate_plan_fill.py`, `submission` branch, fresh-venv README test. Code, checkpoints and README go on the cloud drive (the user does this).
6. Report (PPT-style PDF, about 10 pages; name and ID on the cover).
