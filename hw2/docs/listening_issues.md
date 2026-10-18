# Listening issues

Every comment, tag and session note from the listening sessions, with what was measured and done about it. Status as of 2026-10-18: **fixed** (in code, used by every set made since), **built** (in the listening build or render), **open**, **done**. Sections 0-10 come from the pilot sessions `ckpt`, `ckpt2`, `planfill` and `candidates` (2026-10-14 to 10-16, flawed rendering); section 11 on are the A/B sessions. Measured results are in `results/findings.md`.

Song `NN` is `NN.mid` / `NN.wav` in the source folder: `ref` = `outputs/ref/heldout_crop32` (POP909 song number), `loss_0.40` / `loss_1.00` / `last` = `outputs/exp2/<name>`, `task_a_plan` = `outputs/task_a_plan_form_var_pbias4_bias8/t1_k0_p0.95`, `form_IABAE` = `outputs/plan_fill_v2/form_IABAE`, `free_pbias4_bias8` = `outputs/plan_fill_rep/free_pbias4_bias8`, `melody_first` / `piano_first` = `outputs/plan_fill/<name>`. Several of these sets are superseded (`docs/status.md`).

## 0. Sudden loud notes and silent long notes (built; one part open)

- **Said:** sudden loud single notes ("only certain notes significantly louder and very disturbing"), and long notes on the roll with no sound (`candidates clip_15`, `task_a_plan` song `15`, "happened to a lot of audio").
- **Loud notes, cause tested in FluidSynth:** the same pitch in the MELODY and PIANO tracks at once plays two voices, 6 dB louder than one; the same pitch twice on one track is only 1.4 dB louder. Only two-track pieces (plan-fill) have it: 7-12 such notes per rated plan-fill clip, none in single-stage or real clips.
- **Built:** `hw2.render.one_keyboard` plays each piece as one keyboard before rendering: a strike doubled across tracks is dropped (keep the louder, melody on ties), a held note ends where the next strike on that key starts. It runs in `midi_to_wav`, so every WAV gets it; WAVs rendered before 2026-10-16 23:00 do not have it.
- **Note clusters, fixed 2026-10-17:** the first `RepeatGuide` stacked up to 32 melody notes on one onset (`candidates clip_15` had 2 clusters with 18 notes). Fixed in code and re-run into `outputs/plan_fill_rep_fix/*`; the candidate sets of 2026-10-16 that had them are archived.
- **Silent long notes, explained:** the FluidR3 piano decays like a real piano; a held middle C falls 35 dB in 4 s. Notes over 3 s are 0.3-1.2% of notes in every source, real songs included. Not a bug.
- **Spikes in velocity, done 2026-10-18:** most spikes are melody notes above a bar median set by the piano, as in real openings (7.7%); plan-fill as played is at 7.4-9.5% (findings "Sudden loud notes").
- **Open:** plan-fill MIDI still contains the doubled notes (the render removes them).

## 1. Weak or messy openings (open)

- **Said (pilot):** "every song has weak/awkward/random in the first few bar", "terrible first 8 bar", "weak first 16 bar", "first 4 bar is not a song at all", "so many wrong note in melody at first 8", "too rush in the beginning", "very strange start", "misaligned ... in the start". Tag `abrupt_start` on 6 of 20 `candidates` clips.
- **Said (2026-10-18, user):** a lot of openings are random, messy and disturbing, not just sparse.
- **Done:** A/B sessions since T1 use real song openings (`outputs/ref/heldout_open32`) as anchors, so both sides start a song.
- **Superseded explanation:** the pilot notes put this down to the models copying POP909's sparse intros. The 2026-10-18 comment says sparseness is not the main problem, so that explanation is not used.
- **Next:** the cut test in T5 (40 bars generated, the first 8 dropped; `docs/status.md`).

## 2. Endings that stop or "give up" (open)

- **Said:** "end in rather abrupt way feel like it's cut off" (`ckpt clip_05`, `loss_1.00` song `12`); "end in rather incomplete progression" (`planfill` session note); "after bar 28 sounds like it's giving up" (`candidates clip_06`, `loss_0.40` song `14`); "it's crashing out and giving up" (`candidates clip_10`, `free_pbias4_bias8` song `04`). Tag `cut_off_ending` on 6 clips, 2 of them `ref`.
- Not measured. Later halves: "strange latter half" and "loses it in the later part" (T1), "first half and later half totally different" (T2), "weird key change in the last 8 bars" (T3), "weird chord in the latter 8 bars" (T4).

## 3. Silence, blank bars, missing melody (open for plan-fill)

- **Said:** "some half bar looks left out empty" (`ref` song `338`); "some strange blank" (`ref` song `559`); "2 or 3 bar when melody just not there" (`ref` song `522`); "a half blank" (`ref` song `364`); "weird silent in bar 12 to bar 13" (`ref` song `648`); "some blank part" (`melody_first` song `13`); "melody stop and only the piano stay in same note" (`piano_first` song `15`). Tag `silent_gap` on 9 `candidates` clips, `melody_missing` on 6.
- Five of the seven comments are about real songs. For plan-fill, the measured silent share is above the real level and comes from the fill model (findings "Silence").

## 4. Volume (built)

- **Said:** "some notes abruptly loud ... need to fix in code"; "the velocity is too aggressive sometimes" (`ref` song `237`); "a lot of song have bad audio balanced"; "melody too weak" (`loss_0.40` song `17`); good: "melody and piano are separated", "very good volume separation".
- **Measured:** clips ranged from -40.6 to -31.3 LUFS. Real POP909 median velocity: `MELODY` 94, `BRIDGE` 87, `PIANO` 82.
- **Built:** `listening.py web --velocity real` quantile-maps each source's velocities per track onto POP909's, re-renders, and sets every clip to -18 LUFS (fixed gain, peak limiter); `min_gap` lifts a plan-fill melody whose median is less than 12 above the piano (`t2_openings pair_01` "same volume in melody and piano"). Generated MIDI files are unchanged. The submitted WAVs still need the loudness step (`docs/status.md`).

## 5. Beat and tempo (fixed for generated pieces)

- **Said:** "the beat is kinda misaligned between melody and piano in first half" (`ref` song `237`); "misaligned beat at first few bar" (`ref` song `648`); "sudden speed change" (`melody_first` song `13`); "strange pace" (`ref` song `559`). Tags `timing_off` on 9 clips, `odd_tempo` on 6.
- Three of the four comments are about real songs. Generated tempo changes are fixed by one opening tempo (section 11).

## 6. Repetition and boredom (open)

- **Said:** "the chord almost the same in the whole song ... kinda repetitive" (`loss_0.40` song `11`); "long and boring and loose" (`piano_first` song `08`); "nothing very bad just boring and not much going on" (`task_a_plan` song `06`); "the variation in the last few bar ... way too late" (`last` song `06`). Tag `too_repetitive` on 4 clips.

## 7. Wrong notes and key (open for plan-fill)

- **Said:** "the key change is very weird in this song" (`form_IABAE` song `03`); wrong melody notes in the first 8 bars (`form_IABAE` song `19`); T3 and T4 comments on key changes and chords (checked in findings "Listening tests").
- **Measured 2026-10-18 (findings "Harmony"):** no key or transposition fault (plan-fill melodies 99.7-99.8% in key, real 99.5%); plan-fill melodies sit on chord tones less than real ones, and 20-35% of pieces fall past the real worst-10% line. Next: the chord-relative fill model (`docs/status.md`).

## 8. Note length and articulation (open)

- **Said:** "too much on off between notes which sound rigid and not smooth" (`task_a_plan` song `16`); "the piano notes should hold longer or decrease the velocity" (`melody_first` song `12`); "1 bar substantially with same note too long" (`ref` song `559`); "too many notes" (`ref` song `822`).

## 9. Structure and bar count (bar count fixed; structure open)

- **Said:** "almost every song lack proper progression like we do AABA ... and 33 bar is kinda a weird number" (`ckpt` session note).
- **Bar count, fixed:** checked 2026-10-18 on `loss_0.40` song `14`: 32 `Bar` tokens, one of them with positions past beat 4, so the MIDI stretches to 34 downbeats. The same fault as the 37-bar piece in section 11; `--max_beats` bans those positions for every set made since.
- **Structure:** the form rule (`--form`) and repeat bias are the plan-fill answer (findings "Form rule", "Repeat bias"); the single-stage model has none.

## 10. Page and rendering (done, one open)

- **Fixed:** piano roll ahead of the audio by up to 2.6 s on 6 `candidates` clips; `midi_to_wav` trims leading silence and the roll shifts by the same amount.
- **Done:** "really hard to identify whether each metrics stands for" led to the v2 rubric, then to A/B pages; on 2026-10-18 "the option for position didn't make sense" and "tags are all or nothing for the whole song" led to the section duel (`docs/listening_rubric.md`).
- **Open:** "only heard in left channel" on the iPhone. All served clips are stereo with both channels within 4 dB; the cause is not in the files.

## 11. Session `t1_checkpoint` (2026-10-17; 7 A/B pairs)

Result: `loss_0.40` 3 wins, `step_10000` 1, same 1. Both lost to their real anchors.

- **"why A has 37 bar?"** (`pair_02` A, `step_10000` song `04`): **fixed.** 10 of its 32 bars hold `Pos_` 64-95 (beats 5-6), and the decoder stretches each such bar. POP909's beat labels give 465 of 896 songs at least one 5- or 6-beat bar, and the model learned them. Single-stage sampling bans `Pos_` at or past the piece meter (`--max_beats`, default 4); 8 of 8 test pieces are exactly 32 four-beat bars.
- **"very large variations in tempo"** (`pair_05` A, `step_10000` song `02`): **fixed.** Its tokens set tempo 75, 110, 90 and 120 bpm; 8 of 20 `step_10000` pieces had no opening `Tempo_`. Sampling requires one opening `Tempo_` and bans later ones (`--tempo_changes` turns it off).
- **"B is SO LOUD"** (`pair_07` B, `step_10000` song `14`): **open.** Both clips measure -18.1 LUFS; B is sparse (5.3 notes/s vs 11.0) and high (90th percentile pitch 85 vs 71), so it gets about 5 dB more gain to reach the target.
- **"full empty bar 2"** (`pair_06` A, `loss_0.40_seed1000` song `03`): **open.** Bars 1-2 hold 2 and 1 notes.
- **"piano is broken with dropping frequencies ... notes too close side by side"** (`pair_02` A): **open**; the stretched piece above, so a re-listen of a fixed set is needed.
- **"B has well touching melodies ... melody and piano echoing each other"** (`pair_04` B, real song `889`).

## 12. Sessions `t2_openings` to `t4_openings` (2026-10-17/18)

Comments that were checked against the MIDI are in findings "Listening tests". On 2026-10-18 the user also asked how "a bit" and "clearly" count: both count as one win; strength is reported, not used in decisions (page hint added).
