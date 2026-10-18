# Listening issues

From every comment, tag and session note in sessions `ckpt`, `ckpt2`, `planfill` and `candidates` (2026-10-14 to 10-16). Clip labels are `session clip (source)`; `ref` is a real held-out crop. Status: **fixed**, **built** (in the next session), **open**.

Song `NN` is `NN.mid` / `NN.wav` in the source folder: `ref` = `outputs/ref/heldout_crop32` (POP909 song number), `loss_0.40` / `loss_1.00` / `last` = `outputs/exp2/<name>`, `task_a_plan` = `outputs/task_a_plan_form_var_pbias4_bias8/t1_k0_p0.95`, `form_IABAE` = `outputs/plan_fill_v2/form_IABAE`, `free_pbias4_bias8` = `outputs/plan_fill_rep/free_pbias4_bias8`, `melody_first` / `piano_first` = `outputs/plan_fill/<name>`.

## 0. Sudden loud notes and silent long notes (loud notes built; silent notes explained)

- **Said:** sudden loud single notes ("only certain notes significantly louder and very disturbing"), and long notes on the roll with no sound (`candidates clip_15`, `task_a_plan` song `15`, "happened to a lot of audio").
- **Loud notes, cause tested in FluidSynth:** the same pitch in the MELODY and PIANO tracks at once plays two voices, 6 dB louder than one. The same pitch twice on one track is only 1.4 dB louder. So only two-track pieces (plan-fill) have it: 7-12 such notes per rated plan-fill clip, none in single-stage or real clips.
- **Silent long notes, cause tested:** the FluidR3 piano decays like a real piano. A held middle C (MIDI 60) falls 35 dB in 4 s and 45 dB by 7 s; C5 (72) falls 37 dB in 4 s. A note the roll shows for 6 s is close to inaudible after 3-4 s. Notes over 3 s: 0.3-1.2% of notes in every source, real songs included.
- **Note clusters, a bug in `RepeatGuide` (`--repeat_bias`, found 2026-10-17):** on a `Rep_rhythm` bar (and a `Rep_var` bar whose pitch changes) the guide counts a source note as used only when the written pitch matches, so it keeps boosting the same onset and the melody stacks notes there up to the 32-note cap (one bar of `free_bias8` song 02: 32 notes on beat 1). Pieces with a cluster (4 or more melody notes on one onset): `free`/`free_bias4` 0/20, `free_bias8` 4/20, `free_bias16` 17/20, both `TASK A` candidate sets 4-6/20 per config, `TASK B` with bias 1/3. Rated clips: `candidates clip_15` (`task_a_plan` song `15`, the "abruptly loud ... especially annoying" one) has 2 clusters with 18 notes; `clip_05` and `clip_09` (`free_pbias4_bias8`) have 5 and 1. Fixed in code 2026-10-17 (a note written at a source onset uses up the nearest source note there; and within a bar a part can no longer strike the same position and pitch twice); the runs above are not regenerated yet.
- **Not a cause:** a same-pitch note struck inside a held note on the same track ends the held note at its own note-off. pretty_midi (the page's piano roll) and FluidSynth pair note-on/off the same way, so roll and audio agree there.
- **Built:** `hw2.render.one_keyboard` plays each piece as one keyboard before rendering: a strike doubled across tracks is dropped (keep the louder, melody on ties), a held note ends where the next strike on that key starts. It runs in `midi_to_wav`, so every WAV (listening pages and submission renders) gets it. Generated MIDI, tokens and metrics are unchanged; WAVs rendered before 2026-10-16 23:00 do not have it.
- **Effect on the pilot ratings:** the loud notes hit only plan-fill clips (`planfill`, `candidates` sessions), so they bias those comparisons against plan-fill; the size is unknown (per-clip count vs score residual: Spearman -0.03, 56 clips). Decay affects all sources alike.
- **Open:** plan-fill writes the doubled notes (the piano part takes the melody's pitch at the same onset); fix it at generation so the MIDI itself is clean. Decay: the roll could fade long notes, or nothing (it is how a piano sounds).

## 1. Weak intros (open, cause found)

- **Said:** "every song has weak/awkward/random in the first few bar" (`planfill` session note); "terrible first 8 bar" (`candidates clip_01`, `form_IABAE` song `09`); "weak first 16 bar" (`candidates clip_05`, `free_pbias4_bias8` song `08`); "first 4 bar is not a song at all" (`candidates clip_11`, `form_IABAE` song `04`); "so many wrong note in melody at first 8" (`candidates clip_07`, `form_IABAE` song `19`); "too rush in the beginning" (`candidates clip_08`, `loss_0.40` song `07`); "very strange start" (`planfill clip_09`, `piano_first` song `16`); "misaligned ... in the start" (`planfill clip_10`, `melody_first` song `01`). Tag `abrupt_start` on 6 of 20 `candidates` clips.
- **Measured:** generated pieces have 25-35% fewer notes in bars 1-4 than in bars 5-32 (`loss_0.40` 15.8 vs 23.4 notes per bar, `task_a_plan` 14.8 vs 21.9). Real song openings show the same pattern: the first 32 bars of 896 POP909 songs average 14.4 vs 20.8, and 272 songs start with an empty or pickup bar. The real crops used as anchors are random mid-song windows (22.0 vs 23.0), so they never have an intro.
- **Cause:** the models start at `<BOS>` (song start) and imitate POP909 intros: sparse bars, pickups. The anchor is a mid-song crop, which makes the comparison unfair at the start.
- **Next:** use real song openings (first 32 bars of held-out songs) as the A/B anchor, so both sides start a song. For the report, compare bars 1-4 of generated pieces with real openings, not with crops.

## 2. Endings that stop or "give up" (open)

- **Said:** "end in rather abrupt way feel like it's cut off" (`ckpt clip_05`, `loss_1.00` song `12`); "end in rather incomplete progression" (`planfill` session note); "after bar 28 sounds like it's giving up" (`candidates clip_06`, `loss_0.40` song `14`); "it's crashing out and giving up" (`candidates clip_10`, `free_pbias4_bias8` song `04`). Tag `cut_off_ending` on 6 clips, 2 of them `ref`.
- **Cause (likely):** a fixed 32-bar budget with no notion of an ending; `ref` crops also end mid-song. Not checked further.

## 3. Silence, blank bars, missing melody (open)

- **Said:** "some half bar looks left out empty" (`ckpt2 clip_01`, `ref` song `338`); "some strange blank" (`ckpt2 clip_11`, `ref` song `559`); "2 or 3 bar when melody just not there" (`planfill clip_03`, `ref` song `522`); "a half blank" (`planfill clip_08`, `ref` song `364`); "weird silent in bar 12 to bar 13" (`candidates clip_19`, `ref` song `648`); "some blank part" (`planfill clip_01`, `melody_first` song `13`); "melody stop and only the piano stay in same note" (`planfill clip_02`, `piano_first` song `15`). Tag `silent_gap` on 9 `candidates` clips, `melody_missing` on 6.
- **Note:** half of these are real songs, so part of it is POP909 itself (melody rests). The rest overlaps the earlier bar-end silence work.
- **Said:** "some note too long there no sound in the audio (happened to a lot of audio)" (`candidates clip_15`, `task_a_plan` song `15`). The FluidR3 piano decaying on long held notes (measured, see section 0).

## 4. Volume (built)

- **Said:** "some notes abruptly loud ... need to fix in code" (`candidates clip_15`, `task_a_plan` song `15`); "the velocity is too aggressive sometimes" (`planfill clip_05`, `ref` song `237`); "a lot of song have bad audio balanced" (`candidates` session note); "only certain notes significant louder and very disturbing"; "melody too weak" (`planfill clip_04`, `loss_0.40` song `17`); good: "melody and piano are separated" (`candidates clip_03`, `free_pbias4_bias8` song `19`), "very good volume separation" (`candidates clip_17`, `task_a_plan` song `06`).
- **Measured:** clips ranged from -40.6 to -31.3 LUFS. Notes more than 20 above the local median velocity: 4.7% in `ref`, 4.7% `loss_0.40`, 1.3% `task_a_plan`, so spikes are in POP909 too. Real POP909 median velocity: `MELODY` 94, `BRIDGE` 87, `PIANO` 82.
- **Built:** `listening.py web --velocity real` quantile-maps each source's velocities per track onto POP909's (`MELODY`, `PIANO`+`BRIDGE`, or all tracks for single-track pieces), re-renders, and sets every clip to -18 LUFS (fixed gain, peak limiter). Generated MIDI files are unchanged.

## 5. Beat and tempo (open)

- **Said:** "the beat is kinda misaligned between melody and piano in first half" (`planfill clip_05`, `ref` song `237`); "misaligned beat at first few bar" (`candidates clip_19`, `ref` song `648`); "sudden speed change" (`planfill clip_01`, `melody_first` song `13`); "strange pace" (`ckpt2 clip_11`, `ref` song `559`). Tags `timing_off` on 9 clips, `odd_tempo` on 6.
- **Note:** most are real songs, so likely POP909 timing (tempo changes, pickups). Tempo tokens in the middle of generated pieces: see section 11 (fixed 2026-10-17).

## 6. Repetition and boredom (open)

- **Said:** "the chord almost the same in the whole song ... kinda repetitive" (`ckpt clip_07`, `loss_0.40` song `11`); "long and boring and loose" (`planfill clip_12`, `piano_first` song `08`); "nothing very bad just boring and not much going on" (`candidates clip_17`, `task_a_plan` song `06`); "the variation in the last few bar ... way too late" (`ckpt clip_06`, `last` song `06`). Tag `too_repetitive` on 4 clips.

## 7. Wrong notes and key (open)

- **Said:** "the key change is very weird in this song pls check what happened" (`candidates clip_16`, `form_IABAE` song `03` 03); wrong melody notes in the first 8 bars (`candidates clip_07`, `form_IABAE` song `19`). Tag `wrong_notes` on 6 clips, none `ref`.
- **Next:** check `form_IABAE/03` for a key or chord jump between sections.

## 8. Note length and articulation (open)

- **Said:** "too much on off between notes which sound rigid and not smooth" (`candidates clip_02`, `task_a_plan` song `16`); "the piano notes should hold longer or decrease the velocity" (`planfill clip_06`, `melody_first` song `12`); "1 bar substantially with same note too long" (`ckpt2 clip_11`, `ref` song `559`); "too many notes" (`candidates clip_04`, `ref` song `822`).

## 9. Structure (open)

- **Said:** "almost every song lack proper progression like we do AABA ... and 33 bar is kinda a weird number and it's not even full 33 bar" (`ckpt` session note).
- **Note:** some rolls show 33 bars (e.g. `candidates clip_06`, `loss_0.40` song `14`) for 32-bar pieces. Not checked.

## 10. Page and rendering (mixed)

- **Fixed:** piano roll ahead of the audio by up to 2.6 s on 6 `candidates` clips (`candidates clip_17`, `task_a_plan` song `06` "the sheet doesn't meet the audio AT ALL", `candidates clip_20`, `task_a_plan` song `11` "different start offset/lag"); `midi_to_wav` trims leading silence, the roll now shifts by the same amount.
- **Open:** "only heard in left channel" on the iPhone. All served clips are stereo with both channels within 4 dB; the cause is not in the files.
- **Done:** "really hard to identify whether each metrics stands for" (`ckpt` session note) led to the v2 rubric.

## 11. Session `t1_checkpoint` (2026-10-17; 7 A/B pairs, `results/listening/t1_checkpoint.csv`)

Result: `loss_0.40` 3 wins, `step_10000` 1, same 1 (5 pairs). Decision by the fixed rule: single-stage checkpoint = `loss_0.40`. Both lost to their real anchors (`pair_04`, `pair_06`). Sources: `loss040` = `outputs/exp2/loss_0.40` or `loss_0.40_seed1000`, `step10000` = `outputs/exp2/step_10000`.

- **"why A has 37 bar?"** (`pair_02` A, `step_10000` song `04`): **fixed.** The piece has 32 `Bar` tokens, but 10 bars hold `Pos_` 64-95 (beats 5-6), and the decoder stretches each such bar to hold its notes, so the piece lasts 36.75 four-beat bars. POP909's beat labels give 465 of 896 songs at least one 5- or 6-beat bar (17,756 of 1.5M `Pos_` tokens), and the model learned them: 3-14 of 20 single-stage pieces per checkpoint have such a bar; plan-fill has none. Fix: single-stage sampling bans `Pos_` at or past the piece meter (`--max_beats`, default 4; TASK B: the prompts' meter). Test: 8 of 8 pieces are exactly 32 four-beat bars.
- **"very large variations in tempo"** (`pair_05` A, `step_10000` song `02`): **fixed.** Its tokens set tempo 75, 110, 90 and 120 bpm (bars 1, 2, 3, 13). POP909: 33 of 896 songs (3.7%) jump 20+ bpm; generated: 3-4 of 20 pieces (15-20%). Also 8 of 20 `step_10000` pieces had no `Tempo_` at the start and played at the 120 bpm default (every real song opens `Bar Pos_0 Tempo_`). Fix: sampling requires one opening `Tempo_` and bans later ones (`--tempo_changes` turns it off).
- **"B is SO LOUD"** (`pair_07` B, `step_10000` song `14`): **open.** Both clips measure -18.1 LUFS on the page; B peaks at -0.9 dBFS (A -2.0), max momentary -11.5 vs -13.1 LUFS. B is sparse (5.3 notes/s vs 11.0) and high (90th percentile pitch 85 vs 71), so it gets about 5 dB more gain to reach the target. Not changed yet.
- **"full empty bar 2"** (`pair_06` A, `loss_0.40_seed1000` song `03`): **open.** Bars 1-2 hold 2 and 1 notes (sparse intro); see section 1 and 3.
- **"piano is broken with dropping frequencies ... notes too close side by side"** (`pair_02` A, `step_10000` song `04`): **open, not checked.** This is the stretched piece above; re-listen after the fix.
- **"A has very strange latter half"** (`pair_01` A, `loss_0.40_seed1000` song `11`), **"A kinda loose it in the later part"** (`pair_03` A, `loss_0.40` song `03`): drift in the second half, both `loss_0.40`; see section 9.
- **"B has well touching melodies ... melody and piano echoing each other"** (`pair_04` B, real song `889`, preferred over `step_10000` song `03`).

- **"A seems to have same volume in melody and piano"** (`t2_openings` `pair_01` A, plan-fill): **fixed in the listening build 2026-10-17.** The pooled quantile map keeps each piece's own melody-piano gap; this piece had median velocities 90 vs 88 (the model's gap is 13.6 on average with sd 9.4, real songs 13.0 with sd 8.9). `apply_velocity_map(min_gap=...)` now lifts the melody of a piece until its median is at least the real median gap (94 - 82 = 12) above the piano. MIDI files are unchanged; clips of this session were rebuilt and re-synced.
- **Stem loudness check** (`scripts/stem_gap.py`, `results/stem_gap.csv`): melody stem minus piano stem in LUFS, real 299 songs mean -2.73 dB (sd 2.4); plan-fill sets after the real-velocity map -2.8 / -2.3 / -3.3 dB, with the `min_gap` lift -2.0 / -1.7 / -2.3 dB. The balance is at the real level; the lift moves it about 0.8 dB above. `pair_01` A measures -3.0 dB after the lift.
- **Sudden loud notes in plan-fill, resolved 2026-10-18** (`results/findings.md` section "Sudden loud notes"): most spikes are melody notes above the piano-set bar median, as in real songs; real openings are at 7.7%, plan-fill as played at 7.4-9.5%. The lift stays. First noted 2026-10-17 after a volume check: over all notes of 20 pieces, as played by the listening build (real-velocity map), the share of notes more than 20 above the local median velocity is 8.7% for `oldplan_fill040`, 9.6% with the `min_gap` lift, 5.0% for `single_t1_p1`, 6% for real songs (whole songs, every tenth). Note-level velocity quantiles (5/50/95): real 59/87/116, single-stage 57/85/114, plan-fill 58/84/115. The lift raised the spike share, so it may make volume worse; unresolved. The user's claim that single-stage volume was better was retracted, so the cause (two-role map, melody lift) is still to test.
