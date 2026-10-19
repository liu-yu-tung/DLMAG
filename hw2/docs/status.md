# HW2 status

As of 2026-10-18 evening. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages, name and ID on the cover) is 100% of the grade. Measured results: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, submitted: single-stage `gpt2_ppb16/loss_0.40`**, config `t1 p1`, set `outputs/cand_a/single_t1_p1`. Picked by T1 (3 wins, 1 loss, 1 same vs `step_10000`) and T2 (plan-fill 3 of 6 against it, below the 5-of-6 rule). The other configs (`t1 p0.95`, `t0.9 p1`) are in `outputs/cand_a/`; T4 favoured `t0.9 p1` but was confounded with the side, so `t1 p1` stays. Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK A cut, under test:** 40 bars generated, the first 8 dropped (`generate_uncond.py --skip_bars 8`), sets `outputs/cand_a/single_*_skip8`. Allowed by the spec if the inference code does it and the report says so.
- **TASK B:** must use the TASK A checkpoint (spec page 16); set `outputs/task_b_final` (3 prompts x 3 configs, 8 + 24 bars); best config `t1 p0.95` for all three prompts (T3, 6 of 6).
- **Plan-fill (extra system for the report):** plan `plan_rep/best`, fill `fill_rep_60k/last` (T4: 2 of 2 against `fill_rep_long/loss_0.40`), melody first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`; set `outputs/sweep/fill_rep_60k/last` (the 20 plans of `pf_oldplan_fill040`).
- **Plan-fill, chord-relative:** `fill_rel_60k/last` (pitches as `Rel_k`, semitones from the chord root; `docs/plan_fill.md`), set `outputs/cand_a/pf_fillrel60k`: melody on chord tones at the real level, more minor-2nd rubs (findings "Chord-relative fill, trained"). In T5.
- **Plan-fill, melody-lead (2026-10-20):** melody model `melody_20k` plus BONUS piano, set `outputs/cand_a/pf_melody20k`: the piano does not follow the melody (80% of pieces past the real line), not a listening candidate (findings "Melody model plus BONUS piano"). No-chord fill `fill_nochord_60k/last`, set `outputs/cand_a/pf_fillnochord60k`: also 80% past the line (findings "Fill trained without chord labels"). All three melody-lead options failed; none goes to listening. The no-training test (`--drop chords`) failed (findings "Melody-lead tests").
- **BONUS TASK:** `abl_melody/step_7000`, ablations `abl_lookahead`, `abl_chords`. Done.
- **Held-out songs:** 45 of 896 are held out; the report must say so.

## Next

1. T5 grades (`t5_sections`, built 2026-10-20 00:20): import, reveal, `grade_model.py check --session t5_sections` before `fit`, then apply the rules below. Melody-lead sets: generate on the same plans, `harmony_stats.py --chords_from_piano` plus the usual stats; a listening test only if one comes near real.
2. **T5 rules** (session `t5_sections`, 12 pairs, section-duel form, sides balanced), fixed before rating:
   - single-stage `t1 p1` cut vs uncut, same seeds, 4 pairs: the cut is submitted if it wins more than it loses (tie: uncut)
   - chord-relative vs current plan-fill on the same plans (`sweep/fill_rep_60k/last`), 2 pairs: kept for plan-fill if it wins more (tie: current)
   - chord-relative plan-fill vs single-stage, 3 against cut and 3 against uncut: plan-fill is submitted only with at least 5 of 6
3. Packaging: checkpoint defaults in the `Makefile` and `generate_plan_fill.py` (and the README examples), a `--plan_tok2idx` option (without `runs/` the plan vocabulary falls back to the fill one, which breaks with a `Rel_` fill model), final TASK A and TASK B WAVs at −18 LUFS (now −38 to −30; velocities as written), `submission` branch, fresh-venv README test. Code, checkpoints and README go on the cloud drive (the user does this).
4. Report.

## Open issues

- **Plan-fill melody vs chords:** 20-35% of pieces fall past the real worst-10% line for melody time on chord tones (findings "Harmony"); the chord-relative model is the attempt.
- **Plan-fill silence:** above the real level, and it comes from the fill model (findings "Silence").
- **Plan-fill repeats:** asked repeats written in 14-22% of bars with the repeat bias; the plans ask for too many (46% of bars vs 30%).
- **Single-stage openings:** the listener hears messy starts; the cut is the attempt (T5).
- **Re-listen requests:** loud sparse or high clip, empty bar 2, "broken piano" (`docs/listening_issues.md` section 11).

## Stopped or replaced (kept for the record)

- **`--chord_bias` (removed 2026-10-18, never used in a kept set):** a bonus for chord tones on strong beats. The chord labels are triads, so it would also push out 7ths and 9ths. Replaced by the chord-relative encoding.
- **Plan-fill cut (`outputs/cand_a/pf_fill60k_skip8`):** dropped (user: no cut for plan-fill if the harmony gets fixed). In that set's `plans.json`, piece 04 lost a key change made inside the dropped warm-up (fixed in code since), so analyses misread that piece.
- **Unrated `t5_openings`:** replaced by the T5 above before anyone rated it.
- **Side-swapped single-stage config re-test:** dropped by the user for the cut test.
- **Superseded models:** `plan_v1`, `fill_v1`, `fill_rep`, `plan_rep_long`, `fill_rep_long`, `plan_rep_24k` (each beaten or tied in listening by the current ones, or replaced before testing).
- **Superseded rating forms:** the 1-5 score page (pilot sessions) and the A/B issue tags and element strips (before `t5`); see `docs/listening_rubric.md`.
- **Old repeat guide:** stacked melody notes before the fix of 2026-10-17; runs made with it are not used.
