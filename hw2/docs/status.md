# HW2 status

As of 2026-10-25 04:20. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages, name and ID on the cover) is 100% of the grade. Measured results: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, submitted: single-stage `gpt2_ppb16/loss_0.40`, config `t1 p1`, cut: 40 bars generated, the first 8 dropped (`generate_uncond.py --skip_bars 8`)**, set `outputs/cand_a/single_t1_p1_skip8`. Checkpoint picked by T1 (3 wins, 1 loss, 1 same vs `step_10000`); cut picked by T5 (4 of 4 against uncut, same seeds); plan-fill lost the 5-of-6 rule in T2 (3 of 6) and T5 (4 of 6). Allowed by the spec if the inference code does it and the report says so. Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK B:** must use the TASK A checkpoint (spec page 16); set `outputs/task_b_final` (3 prompts x 3 configs, 8 + 24 bars); best config `t1 p0.95` for all three prompts (T3, 6 of 6).
- **Plan-fill (extra system for the report):** plan `plan_rep/best`, fill `fill_rep_60k/last` (T4: 2 of 2 against `fill_rep_long/loss_0.40`), melody first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`; set `outputs/sweep/fill_rep_60k/last` (the 20 plans of `pf_oldplan_fill040`).
- **Plan-fill, chord-relative:** `fill_rel_60k/last` (pitches as `Rel_k`, semitones from the chord root; `docs/plan_fill.md`), set `outputs/cand_a/pf_fillrel60k`: melody on chord tones at the real level, more minor-2nd rubs (findings "Chord-relative fill, trained"). T5: 1-1 against the current fill, so the current fill stays (tie rule); reported as the harmony attempt.
- **Plan-fill, melody-lead (2026-10-20):** melody model `melody_20k` plus BONUS piano, set `outputs/cand_a/pf_melody20k`: the piano does not follow the melody (80% of pieces past the real line), not a listening candidate (findings "Melody model plus BONUS piano"). No-chord fill `fill_nochord_60k/last`, set `outputs/cand_a/pf_fillnochord60k`: also 80% past the line (findings "Fill trained without chord labels"). All three melody-lead options failed; none goes to listening. The no-training test (`--drop chords`) failed (findings "Melody-lead tests").
- **BONUS TASK:** `abl_melody/step_7000`, ablations `abl_lookahead`, `abl_chords`. The `abl_chords` row is invalid: training transposed the notes but not the chord roots (findings "BONUS piano against the melody"); fix and retrain (about 45 min GPU) or report the bug: user to decide.
- **Held-out songs:** 45 of 896 are held out; the report must say so.

## Schedule (draft 2026-10-25 04:20; deadline Mon 2026-10-26 23:59)

1. **Done 04:20:** T5 graded (findings "Listening tests", `listening_issues.md` section 13), predictor refit.
2. **Sun 04:30-05:45, packaging code:** checkpoint defaults in the `Makefile`, `generate_uncond.py` (cut on), `generate_plan_fill.py` and the README examples; `--plan_tok2idx` (without `runs/` the plan vocabulary falls back to the fill one, which breaks with a `Rel_` fill model); final TASK A (cut) and TASK B (`t1 p0.95`) WAVs at −18 LUFS (now −38 to −30).
3. **T6 `t6_select` (built 2026-10-25 by about 07:00; rate when back, 10:00-13:00): one named challenger against the TASK A cut.** Fresh pieces nobody has heard. Challenger fixed before building: **chord-relative fill + cut (`--skip_bars 8`, form `W8I4A8B8A8E4`) + semitone penalty (`--rub_penalty`)**, the only plan-fill with a winning record against single-stage (T5: 4 wins, 1 loss, 1 same), with the two fixes aimed at its flaws (clashes, openings). One challenger, not three, so a lucky switch stays unlikely.
   - Sets: single-stage cut seeds 100+ (`outputs/cand_a/single_t1_p1_skip8_s100`); 30 new plans with warm-up (seed 200), filled by the chord-relative fill at penalty 0, 2, 4, 8 and by the current fill at the chosen penalty, all on the same plans (`outputs/cand_a/pf_*_cut_*s200`).
   - **Penalty strength, chosen by measurement before the session is built:** the smallest of 2, 4, 8 with piano-side clash time at most 1.25x real (real 0.005 of melody time), melody chord-tone time (plan labels, pooled) at most 0.01 below penalty 0, and silent share at most 0.005 above penalty 0; if none meets the clash bound, the lowest piano-side clash time among those meeting the other two.
   - 22 pairs, sides alternate within each matchup, no plan used twice outside the same-plan pairs: challenger vs cut single-stage x10; challenger vs the same plans at penalty 0 x4; challenger vs current fill at the same penalty and plans x4; cut single-stage vs real opening x2; challenger vs real opening x2.
   - **Rules, fixed before rating:** (a) TASK A switches to the challenger only with at least 8 wins of its 10 pairs against the cut (ties count as not won; if the two were equal, 8 of 10 happens about 1 time in 20); otherwise the cut stays. (b) The penalty pairs, the fill pairs and the real pairs are for the report only.
   - If (a) switches TASK A to plan-fill, TASK B must switch too (spec page 16: TASK B uses the TASK A model): a new TASK B plan-fill set and a check, about 1-2 h on Monday.
   - The uncut sets made first for T6 (`pf_fill60k_s100`, `pf_fillrel60k_s100`) are kept for the record and not used.
   - Later, after packaging: T7, the submitted TASK A pieces and TASK B continuations as rendered, flag broken ones only.
4. **Sun 06:30-11:00, report:** slide outline (about 10 pages) and the figure and table list from findings, then drafts.
5. **Sun after 11:00 to Mon evening:** finish slides, `submission` branch, fresh-venv README test, code, checkpoints and README to the cloud drive (user), PDF with name and ID on the cover; Mon 20:00 buffer.

## Open issues

- **Plan-fill melody vs chords:** 20-35% of pieces fall past the real worst-10% line for melody time on chord tones (findings "Harmony"); the chord-relative model is the attempt.
- **Plan-fill silence:** above the real level, and it comes from the fill model (findings "Silence").
- **Plan-fill repeats:** asked repeats written in 14-22% of bars with the repeat bias; the plans ask for too many (46% of bars vs 30%).
- **Single-stage openings:** messy starts; the cut fixed them in T5 (bars 1-8 won 4 of 5). Late key moves remain (uncut 06: D to G at bar 17).
- **Re-listen requests:** loud sparse or high clip, empty bar 2, "broken piano" (`docs/listening_issues.md` section 11).

## Stopped or replaced (kept for the record)

- **`--chord_bias` (removed 2026-10-18, never used in a kept set):** a bonus for chord tones on strong beats. The chord labels are triads, so it would also push out 7ths and 9ths. Replaced by the chord-relative encoding.
- **Plan-fill cut (`outputs/cand_a/pf_fill60k_skip8`):** dropped (user: no cut for plan-fill if the harmony gets fixed). In that set's `plans.json`, piece 04 lost a key change made inside the dropped warm-up (fixed in code since), so analyses misread that piece.
- **Unrated `t5_openings`:** replaced by the T5 above before anyone rated it.
- **Side-swapped single-stage config re-test:** dropped by the user for the cut test.
- **Superseded models:** `plan_v1`, `fill_v1`, `fill_rep`, `plan_rep_long`, `fill_rep_long`, `plan_rep_24k` (each beaten or tied in listening by the current ones, or replaced before testing).
- **Superseded rating forms:** the 1-5 score page (pilot sessions) and the A/B issue tags and element strips (before `t5`); see `docs/listening_rubric.md`.
- **Old repeat guide:** stacked melody notes before the fix of 2026-10-17; runs made with it are not used.
