# HW2 status

As of 2026-10-25 04:20. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF, about 10 pages, name and ID on the cover) is 100% of the grade. Measured results: `results/findings.md`. Listening comments: `docs/listening_issues.md`. Pipeline and commands: `README.md`. File map: `CLAUDE.md`.

## Systems and checkpoints

- **TASK A, submitted: single-stage `gpt2_ppb16/loss_0.40`, config `t1 p1`, cut: 40 bars generated, the first 8 dropped (`generate_uncond.py --skip_bars 8`)**, set `outputs/cand_a/single_t1_p1_skip8`. Confirmed by T6 (2026-10-26): the plan-fill challenger won 6 of 10, below the 8-of-10 rule. Checkpoint picked by T1 (3 wins, 1 loss, 1 same vs `step_10000`); cut picked by T5 (4 of 4 against uncut, same seeds); plan-fill lost the 5-of-6 rule in T2 (3 of 6) and T5 (4 of 6). Allowed by the spec if the inference code does it and the report says so. Sampling fixes: no `Pos_` past the meter (`--max_beats`), one opening tempo.
- **TASK B:** must use the TASK A checkpoint (spec page 16); set `outputs/task_b_final` (3 prompts x 3 configs, 8 + 24 bars); best config `t1 p0.95` for all three prompts (T3, 6 of 6).
- **Plan-fill (extra system for the report):** plan `plan_rep/best`, fill `fill_rep_60k/last` (T4: 2 of 2 against `fill_rep_long/loss_0.40`), melody first, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`; set `outputs/sweep/fill_rep_60k/last` (the 20 plans of `pf_oldplan_fill040`).
- **Plan-fill, chord-relative:** `fill_rel_60k/last` (pitches as `Rel_k`, semitones from the chord root; `docs/plan_fill.md`), set `outputs/cand_a/pf_fillrel60k`: melody on chord tones at the real level, more minor-2nd rubs (findings "Chord-relative fill, trained"). T5: 1-1 against the current fill, so the current fill stays (tie rule); reported as the harmony attempt.
- **Plan-fill, melody-lead (2026-10-20):** melody model `melody_20k` plus BONUS piano, set `outputs/cand_a/pf_melody20k`: the piano does not follow the melody (80% of pieces past the real line), not a listening candidate (findings "Melody model plus BONUS piano"). No-chord fill `fill_nochord_60k/last`, set `outputs/cand_a/pf_fillnochord60k`: also 80% past the line (findings "Fill trained without chord labels"). All three melody-lead options failed; none goes to listening. The no-training test (`--drop chords`) failed (findings "Melody-lead tests").
- **BONUS TASK:** `bonus_ppb16/last` (30k steps), `--top_p 0.95 --grammar`, set `outputs/bonus/t1_p0.95_grammar` (CA 0.275, below the octave-down baseline 0.30; OA above both baselines). Ablations `abl_melody`, `abl_lookahead`, `abl_chords` (12k-step runs); the `abl_chords` row is invalid (chord roots not transposed with the notes), stated in the report, not retrained.
- **Held-out songs:** 45 of 896 are held out; the report must say so.

## Schedule (deadline Mon 2026-10-26 23:59; plan at 19:05)

Decided: TASK A = single-stage cut (T5 4-0, T6 challenger 6-4 < 8 of 10); TASK B `t1 p0.95`; BONUS = `bonus_ppb16/last`, `--grammar --top_p 0.95` (`outputs/bonus/t1_p0.95_grammar`; user 2026-10-26); `abl_chords` bug stated in the report, not retrained; publish like HW1 (user 2026-10-26); plan-model listening test skipped, stays a limitation (user 19:00); the report states plan-fill vs cut single-stage as close (T5 + T6 8-5, below the rule) and releases chord-relative fill + `plan_rep_24k/loss_0.40` samples for others to try (user 19:00).

Required by the spec: code, README, `requirements.txt`, the checkpoints behind the submitted files (`gpt2_ppb16/loss_0.40`, `bonus_ppb16/last`), the generation results; not training data, preprocessed data (so the TASK B prompts are tokenized by the README step, not shipped), other people's models or caches. The plan-fill checkpoints are extras.

State at 19:50:

| # | Step | State / ETA |
|---|---|---|
| 1 | Results zip with `README.txt`; `plan_rep_24k_loss_0.40.tar` + checksum; README (TASK B tokenize step, models table, plan-fill "try" command, extras MIDI + MP3) | done (`e889ac4`, `ed37876`) |
| 2 | Chord-relative + cut + penalty 2 under `plan_rep_24k/loss_0.40` (`outputs/cand_a/pf_fillrel60k_cut_pen2_plan24k040_s200`, 20 pieces, not listened to); `dist/release/hw2_extra_plan_fill.zip` (that set + 20 challenger pieces, MIDI + MP3 at -18 LUFS, 97 MB) | done |
| 3 | Orphan branch `hw2-submission` pushed (`e98c404`, `7de14d0`; 210 files; worktree in the session scratchpad `hw2-submission/`); listening page screenshots, blank state (`results/figures/listening_page_*.png`, `78e8ab0`) | done |
| 4 | Release `hw2` is a **draft** (only the owner sees it). 8 assets uploading direct, about 0.26 MB/s in total (route cap, not stream count): 6 tars, `hw2_generation_results.zip`, `hw2_extra_plan_fill.zip`. When all 8 are up: `gh release edit hw2 --repo liu-yu-tung/DLMAG --draft=false`, then check logged out (`curl -sI .../releases/download/hw2/plan_rep_best.tar` gives 302) | uploads ETA 20:50 |
| 5 | Fresh-clone test from the published release: TASK A n=1, TASK B (tokenize step), BONUS | 21:10 |
| 6 | Report deck, 16 slides, Slides type with HW1's look: https://claude.ai/artifact/YLnJeXb6ftHLLR33zgCFqm (scratch copy in the session scratchpad `deck_hw2/`) | first draft done 19:47 |
| 7 | User review and fixes | from 19:50, done by 21:30 |
| 8 | PDF to `dist/cool/b10901114_report.pdf`, first (safe) upload to NTU COOL with the link in the comment; later re-upload only if COOL allows it | safe upload by 22:15, last by 23:30 |

Upload notes: `gh` token had expired (user re-logged in 19:09). Measured: direct 0.26 MB/s whatever the stream count; WARP full tunnel 0.22; direct + WARP proxy 0.51 (worked once, then the proxy timed out after switching modes). WARP is back as it was: proxy mode, disconnected; `warp-svc` left running (started by the user).

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
