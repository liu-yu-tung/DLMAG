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
- **Note:** most are real songs, so likely POP909 timing (tempo changes, pickups). Not checked: tempo tokens in the middle of pieces.

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
