# HW2 status

As of 2026-10-25 04:20. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages, name and ID on the cover) is 100% of the grade. Measured results: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, submitted: single-stage `gpt2_ppb16/loss_0.40`, config `t1 p1`, cut: 40 bars generated, the first 8 dropped (`generate_uncond.py --skip_bars 8`)**, set `outputs/cand_a/single_t1_p1_skip8`. Confirmed by T6 (2026-10-26): the plan-fill challenger won 6 of 10, below the 8-of-10 rule. Checkpoint picked by T1 (3 wins, 1 loss, 1 same vs `step_10000`); cut picked by T5 (4 of 4 against uncut, same seeds); plan-fill lost the 5-of-6 rule in T2 (3 of 6) and T5 (4 of 6). Allowed by the spec if the inference code does it and the report says so. Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK B:** must use the TASK A checkpoint (spec page 16); set `outputs/task_b_final` (3 prompts x 3 configs, 8 + 24 bars); best config `t1 p0.95` for all three prompts (T3, 6 of 6).
- **Plan-fill (extra system for the report):** plan `plan_rep/best`, fill `fill_rep_60k/last` (T4: 2 of 2 against `fill_rep_long/loss_0.40`), melody first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`; set `outputs/sweep/fill_rep_60k/last` (the 20 plans of `pf_oldplan_fill040`).
- **Plan-fill, chord-relative:** `fill_rel_60k/last` (pitches as `Rel_k`, semitones from the chord root; `docs/plan_fill.md`), set `outputs/cand_a/pf_fillrel60k`: melody on chord tones at the real level, more minor-2nd rubs (findings "Chord-relative fill, trained"). T5: 1-1 against the current fill, so the current fill stays (tie rule); reported as the harmony attempt.
- **Plan-fill, melody-lead (2026-10-20):** melody model `melody_20k` plus BONUS piano, set `outputs/cand_a/pf_melody20k`: the piano does not follow the melody (80% of pieces past the real line), not a listening candidate (findings "Melody model plus BONUS piano"). No-chord fill `fill_nochord_60k/last`, set `outputs/cand_a/pf_fillnochord60k`: also 80% past the line (findings "Fill trained without chord labels"). All three melody-lead options failed; none goes to listening. The no-training test (`--drop chords`) failed (findings "Melody-lead tests").
- **BONUS TASK:** `abl_melody/step_7000`, ablations `abl_lookahead`, `abl_chords`. The `abl_chords` row is invalid: training transposed the notes but not the chord roots (findings "BONUS piano against the melody"); fix and retrain (about 45 min GPU) or report the bug: user to decide.
- **Held-out songs:** 45 of 896 are held out; the report must say so.

## Schedule (deadline Mon 2026-10-26 23:59; state at 18:45)

Decided: TASK A = single-stage cut (T5 4-0, T6 challenger 6-4 < 8 of 10); TASK B `t1 p0.95`; BONUS = `bonus_ppb16/last`, `--grammar --top_p 0.95` (`outputs/bonus/t1_p0.95_grammar`; user 2026-10-26); `abl_chords` bug stated in the report, not retrained; publish like HW1 (user 2026-10-26).

1. **Packaging (in progress):** done: README and Makefile final commands (`fba4d32`), `--plan_tok2idx`, `tokens_to_midi` time signatures (`782580f`); `dist/release/*.tar` (5 checkpoints) and `dist/checkpoints.sha256`; `dist/results/hw2_generation_results/` (TASK A 20, TASK B 3 prompts x 3 configs, BONUS 10; WAV copies at -18 LUFS). To do: zip the results and add a note naming the best TASK B config per prompt; branch `hw2-submission` (HW2 inference files only at the root plus `data/tokens/*` vocabularies, prompts and bins; the `submission` branch stays HW1's); release `hw2` (the tars, the results zip); fresh-clone test from the release.
2. **Report:** Slides artifact type as HW1 (`hw1/report/deck/`, same look); outline `docs/report_outline.md`; cover "Liu, Yu-Tung · b10901114", links `github.com/liu-yu-tung/DLMAG/tree/hw2-submission` and `.../releases/tag/hw2`; PDF to `dist/cool/b10901114_report.pdf`.
3. **Optional, user to decide:** plan-model listening test (findings "Plan model length"), only if the user listens while slides are written.
4. User review, PDF upload to NTU COOL.

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
