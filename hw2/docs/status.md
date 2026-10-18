# HW2 status

As of 2026-10-18. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages) is 100% of the grade. Evidence: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, single-stage (`gpt2_ppb16`):** checkpoint `loss_0.40` (T1 2026-10-17: 3 wins, 1 loss, 1 same vs `step_10000`). Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK A, plan-fill:** plan `plan_rep/best`, fill `fill_rep_60k/last` (T4 2026-10-18: 2 of 2 against `fill_rep_long/loss_0.40` on the same plans; `plan_rep_24k/last` tied 1-1 with `plan_rep/best`); melody-first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`. Set: `outputs/sweep/fill_rep_60k/last` (the 20 plans of `pf_oldplan_fill040`). Six older candidate sets of 20 are in `outputs/cand_a/` (3 single-stage, 3 plan-fill); stats in `results/cand_a_*.csv`.
- **TASK A decision (T2, 2026-10-17, `results/listening/t2_openings.csv`):** plan-fill won 3 of 6 pairs against single-stage (2 losses, 1 same), below the 5-of-6 rule, so single-stage `loss_0.40` is submitted; plan-fill stays an extra experiment. Config `t1 p1` (`outputs/cand_a/single_t1_p1`, the other two configs `t1 p0.95`, `t0.9 p1` are in `outputs/cand_a/`). T4: `t0.9 p1` won both its pairs, but side B won every config pair (6 of 7 in the session), so `t1 p1` stays until T5 replays the three matchups with the sides swapped; decision on T4 + T5: most wins, tie to the lower temperature.
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
- Sudden loud notes: resolved 2026-10-18 (findings section "Sudden loud notes"): mostly melody over piano, as in real openings (7.7%); the `min_gap` lift stays for plan-fill audio.
- Final TASK A and TASK B WAVs: velocities as written, loudness to −18 LUFS like the rated clips (now −38 to −30); a packaging step.

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
   - T4 graded 2026-10-18 03:00 (findings section "T4"): new fill checkpoint `fill_rep_60k/last`, plan stays `plan_rep/best`, config decision moved to T5.
   - T5: the three config matchups with the sides swapped (3 pairs), and plan-fill with the new fill against single-stage `t1 p1` (6 pairs, sides alternating; the 5-of-6 rule, comparable with T2).
   - Fill model on real held-out plans: done 21:28 (findings section "Fill model on real plans").
   - Melody-repeat following (14-22% of asked repeats written) and bar-end silence.
4. Listening-grade predictor (user, 2026-10-17; `docs/grade_model.md`): fit on T1, T2 and the pilot; after each graded session run `grade_model.py check` before `fit` (T3: `v1` 5 of 9, `v2` 3 of 9; T4: `v1` 5 of 7, `v2` 4 of 7, `v3` 6 of 7, so `predict` uses `v3` since 2026-10-18); score each new set; predicted grade vs loss for `exp2` (main model) and the new plan-fill runs.
5. Extras: NLL matching of sampling settings, loss vs listening figure (partly covered by item 4), SI feasibility.
6. Packaging: checkpoint defaults in the `Makefile` and `generate_plan_fill.py`, final TASK A and TASK B WAVs at −18 LUFS, `submission` branch, fresh-venv README test. Code, checkpoints and README go on the cloud drive (the user does this).
7. Report (PPT-style PDF, about 10 pages; name and ID on the cover).
