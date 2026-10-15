# Findings for the report

Training-set statistics (896 songs in `data/POP909/training`), from `scripts/data_stats.py` (2026-10-15). Numbers below are copied from the CSVs next to this file.

## Notes heard longest give the chord (`chord_from_notes.csv`)

For each maj/min segment of `chord_midi.txt`, add up how long each pitch class sounds (dur) or how many times it is pressed (count), pick the major or minor triad with the largest total, compare with the label. Weighted by segment time. The true chord boundaries are given, so this is an upper bound for the rule.

| Notes from | Triad match, dur | Triad match, count | Longest pitch class = root, dur | Longest = root, count |
|---|---|---|---|---|
| all tracks | 94.2% | 90.6% | 70.5% | 55.2% |
| PIANO + BRIDGE | 95.4% | 92.6% | 76.4% | 61.7% |
| PIANO | 97.4% | 94.9% | 84.6% | 64.9% |
| BRIDGE | 33.8% | 31.0% | 32.4% | 28.6% |
| MELODY | 28.4% | 26.4% | 30.4% | 28.0% |

- Duration weighting beats press counting for every source; the gap is largest for "longest = root" (PIANO 84.6% vs 64.9%).
- PIANO carries the harmony; MELODY and BRIDGE alone say little about the chord with this rule. For `BONUS TASK`, melody to chord needs a learned stage, not this rule.
- Not checked yet: how POP909 produced its chord labels. If they were computed from these notes, the PIANO match is partly circular.

## Notes and mean duration per bar, by track (`track_stats.csv`)

Bars are downbeat to downbeat of `beat_midi.txt` (72748 bars). Mean duration = total sounding time / presses, in beats.

| Track | Notes per bar p10 / p50 / p90 | Mean duration (beats) p10 / p50 / p90 | Empty bars |
|---|---|---|---|
| PIANO | 7 / 13 / 20 | 0.48 / 0.80 / 1.50 | 1.6% |
| BRIDGE | 0 / 2 / 8 | 0.33 / 0.84 / 2.29 | 35.1% |
| MELODY | 0 / 4 / 8 | 0.29 / 0.55 / 1.50 | 24.5% |

The tracks behave differently, so bar-level features should be kept per track.

## Token budget vs the 2048 context (`token_budget.csv`)

Sections of the `bonus_chd` tokens: plan = `Bar Chd <Pos Chord pairs>`, melody = `Mel` + melody notes, piano = `Acc` + piano notes (`Mel`/`Acc` are the token tags; the section names in `token_budget.csv` are `melody`/`piano`) (BRIDGE not included). Windows are 32 bars, stride 8.

| Scope | Section | p50 | p90 | max | Share over 2048 |
|---|---|---|---|---|---|
| bar | plan | 6 | 6 | 18 | 0% |
| bar | melody | 17 | 33 | 181 | 0% |
| bar | piano | 53 | 81 | 641 | 0% |
| song | plan | 410 | 576 | 1154 | 0% |
| song | melody | 1430 | 1843 | 5607 | 3.9% |
| song | piano | 4250 | 5880 | 12890 | 98.9% |
| song | all | 6184 | 8000 | 15213 | 99.7% |
| 32 bars | plan | 170 | 194 | 244 | 0% |
| 32 bars | melody | 576 | 832 | 3832 | 0.1% |
| 32 bars | piano | 1716 | 2332 | 6752 | 24.0% |
| 32 bars | all | 2472 | 3262 | 8384 | 76.6% |

- A whole-song plan fits the context for every song; notes do not, so the fill stage works bar by bar with a sliding window.
- The basic-task tokens (`data/tokens/ppb16`, all three tracks, with tempo) are about 86 tokens per bar at the median, 6700 per song; the model sees about 24 bars at once.

## Chord label rule in use (`src/hw2/bonus.py`, `chord_token`)

Root as pitch class (enharmonics merged); quality collapsed to maj / min / dim / aug / sus; inversions and extensions dropped; N or unknown root = `Chord_N`. Share of song time: maj 54.3%, min 36.8%, sus 6.6%, N 1.5%, dim 0.7%, aug 0.1%. Detail dropped: inversions 7.4%, 7ths or extensions 7.8% of song time.

## Data split

The spec says to use the whole training set (896 songs). `train.py` holds out 45 (`--n_val 45`, seed 0) for validation loss, checkpoint selection and the real-music reference; models train on 851. No other data. State this in the report.

## Distributions of the plan features (`figures/data_distributions.png`, `progressions.csv`)

From `scripts/plot_data_stats.py`. Key per song by Krumhansl-Kessler profile match (no key file in this copy of POP909; key changes inside a song not handled yet). Chords are degrees in the relative-major frame (a minor key's tonic is vi).

- **Song length:** 77 bars median (p10 57, p90 112.5, range 12 to 193).
- **Tempo:** 74 BPM median (p10 60, p90 116); 4 songs under 50 and 8 over 140 BPM, possibly beat tracking at half or double speed.
- **Meter:** 94.6% of bars have 4 beats, 3.4% have 6, 1.3% have 2; by most common bar, 868 songs are 4-beat and 28 are 6-beat.
- **Key:** 63.3% major, 36.7% minor; most common C (70 songs), G (65), D (54).
- **Chords, share of song time:** vi 17.7%, I 17.2%, V 14.8%, IV 13.4%, ii 8.5%, iii 7.4%; everything else under 3% each. Six diatonic chords cover about 79%.
- **Chord types per song:** 11 median (p10 7, p90 18).
- **Chord changes per bar:** 0: 8.7%, 1: 47.4%, 2: 38.3%, 3: 5.1%, 4+: 0.5%.
- **Top progressions:** 2-chord V->I 6.2%, IV->V 5.5%, V->vi 3.7%, vi->IV 3.6%; 4-chord vi-IV-V-I 1.0%, vi-ii-V-I 0.9%, V-vi-IV-V 0.7%. Long progressions are spread thin (top 4-chord pattern is 1%).
- **Register:** PIANO mean pitch per bar peaks below middle C (about 56), MELODY and BRIDGE overlap around 72. BRIDGE sits in the melody's register, not the accompaniment's.
- **PIANO notes per bar** has a spike at 16 (steady 16th-note patterns over 4 beats).

## Key changes (`key_changes.csv`, `figures/key_changes.png`)

From `scripts/key_changes.py`. Local key per bar by Viterbi over the 12 major scales (a minor key counts as its relative major, the frame of the relative chords). Bar cost = 1 - share of sounding time on the scale; a change costs a switch penalty. No ground truth in our copy of POP909, so the result is checked against the chord labels and by hand.

| Penalty | Songs with a change | Changes | Interval +1 | Interval +2 | Median position in song | Median bars after |
|---|---|---|---|---|---|---|
| 2 | 132 (14.7%) | 160 | 78 | 37 | 0.75 | 17 |
| 4 (default) | 107 (11.9%) | 111 | 74 | 24 | 0.77 | 17 |
| 8 | 69 (7.7%) | 70 | 62 | 4 | 0.77 | 16 |

- The pattern is stable across penalties: about 8-15% of songs change key, almost always once, mostly up 1 or 2 semitones, about three quarters of the way in, for the last ~17 bars. This is the pop "key up for the last chorus".
- At penalty 4, 80% of changes come after 70% of the song; 103 songs change once, 4 twice, none more.
- Share of chord time whose triad fits the scale: 88.1% with one key per song, 90.9% with the local key.
- Spot checks match the chord labels: song 010 G to Ab at bar 70 (Eb, Ab, Fm, Db after), 012 Eb to E at bar 53 (B as the pivot, then A, Abm, F#m, E), 025 A to C at bar 78 (G, C, Am, Em, F).
- For the plan: one key token per song plus an optional key-change token (+1, +2, ...) at the section where it happens covers nearly all songs.

## Silent starts in rendered audio (2026-10-16)

Many POP909 songs start with a bar that holds only a pickup at its end, and both models learned it: the first note of a generated piece can come 1.5-5.7 s after the start (listening notes: "no sound at the start"). `hw2.render.midi_to_wav` now cuts the WAV to 0.2 s before the first sound above -50 dBFS; the MIDI keeps all 32 bars. Existing WAVs were trimmed with `scripts/trim_wavs.py`: 239 files, 325.6 s in total (TASK B prompts and the real held-out crops needed none over 0.5 s).

## Silence, 8-bar windows and start/end return (`silence.csv`, `silence_windows.csv`, `silence_closure.csv`, `scripts/silence_stats.py`)

All on the TA tick grid, 16th-note slots. Real reference: 32-bar crops of POP909, from the song start and from inside (every 8 bars), split by the train.py val split. Generated: 20 pieces per folder, 32 bars.

| Source | Silent share | Gaps >= 1 bar / piece | Empty bars / piece | Lead >= 1 bar | Weakest 8-bar window, sounding share |
|---|---|---|---|---|---|
| real, song start | 1.4-1.6% | 0.02 | 0.04-0.06 | 1-2% | 0.93-0.94 |
| real, inside | 1.1-1.3% | 0-0.005 | 0.02 | 0-0.4% | 0.97 |
| main `step_6000` | 6.0% | 0.15 | 1.40 | 10% | 0.88 |
| main `loss_0.40` | 1.0% | 0 | 0.05 | 5% | 0.92 |
| plan-fill v1 melody-first | 2.8% | 0.20 | 0.55 | 15% | 0.89 |
| plan-fill v2 `gaps` | 2.1% | 0 | 0 | 0% | 0.93 |
| plan-fill v2 `gaps_intro_cadence` | 1.7% | 0 | 0 | 0% | 0.94 |

- Main model: the silence is a `step_6000` problem (under-trained); `loss_0.40` and `last` are as dense as real songs.
- Plan-fill v1: 10 of 11 empty bars were planned empty (`PDen_0` and `MDen_0`). The plan model writes fully empty bars in 1.7% of bars vs 0.17% in POP909, and put one in bar 1 of 5 of 20 plans; POP909 has none in bar 1 (0 of 117). The fill model follows the plan.
- Fix, sampling time only (`PlanRules`, `generate_plan_fill.py --min_piano_bin 1 --tempo_range 60,125`): every bar has piano notes (POP909: 0.4% of bars have none), tempo inside the POP909 5th-95th percentile (60-125). Output: `outputs/plan_fill_v2/gaps`. `--melody_by_bar 5 --cadence` adds a melody entry deadline and a V-I (or III-vi) ending: `outputs/plan_fill_v2/gaps_intro_cadence`.
- Melody entry in real songs is late: median bar 8, only 30% by bar 5, so long intros are POP909 style, not a model error. Melody gaps after entry also match POP909 (longest gap 3.2 bars in v1 vs 1.7-2.4 at song starts and 4.1-4.4 inside songs, where interludes sit).
- 8-bar windows, wrapping past bar 32 (offset 4: the last window is bars 29-32 then 1-4), so every offset covers every bar once. Contrast between windows is flat across offsets in real songs (0.029 at every offset for inside crops) and in generated pieces: note density does not mark an 8-bar section grid. The only position effect is the intro: bars 1-8 are thinner in real songs (bar 1: 10-12 notes and 66-75% sounding, a pickup bar) and in generated pieces.
- Plan-fill sections build up: notes per bar rise from about 19.7 at the first bar of each 8-bar section to 22.2 at the last (bars 9-32); real songs are flat (21.4 to 21.4 inside songs).
- Start/end return (pitch-class content per bar, cosine, bars aligned): in whole POP909 songs bars 1-8 vs the last 8 bars score 0.50-0.53, below the mean pair of windows (0.59). With any start in the last 16 bars the best match is 0.68 vs 0.82 for the best match in the middle; the end beats the middle in 15% of songs. POP909 songs reuse their opening mid-song more than at the end.
- Repetition is the larger form gap: the best-matching pair of 8-bar windows in a real 32-bar crop scores 0.79-0.81, in generated pieces 0.64-0.70. Real crops contain a near-repeated section; generated pieces do not.

## Theme return and repetition period (`return.csv`, `return_lags.csv`, `figures/return_lags.png`, `scripts/return_stats.py`)

Intro = bars before the first melody note, outro = bars after the last; body = the rest. Whole POP909 songs: intro median 6-7 bars (p25 4, p75 9, p90 11-12), outro median 4, body median 66 bars.

- Theme (first 8 body bars) vs later 8-bar windows, pitch-class cosine per bar, return = 0.85 or more: 80-84% of whole songs bring the theme back (about 2 times per song, for all notes, melody and piano alike), mostly mid-body (best match at 0.43-0.47 of the body); the end beats the middle in 0-6% of songs. In the first 32 bars, 39-54% of real songs already have a return; plan-fill 0% (all notes, piano), 6-8% (melody).
- At theme returns (melody cosine >= 0.85, 1437 cases): melody notes repeat exactly (same start and pitch) at median 72% (42% of returns >= 80%); piano at median 35% (18% >= 80%) while its pitch-class cosine stays at 0.90. Pop practice: the melody comes back, the accompaniment keeps the harmony and varies the notes. Random windows: 3% (melody), 7% (piano).
- Lag profile (bar i vs bar i+L, body only): real songs peak at lags 4 and 8 (also 2, 16, 24) and dip at odd lags. Exact note match at lag 4 / 8 vs lag 3 / 6: melody 0.13 / 0.15 vs 0.04 / 0.05; piano 0.16 / 0.16 vs 0.08 / 0.09. Train and held-out curves coincide.
- Main model (`step_6000`, `loss_0.40`, `last`, all notes): no peaks; exact match decays with lag (loss_0.40: 0.09 at lag 1 to 0.07 at 24).
- Plan-fill: pitch-class cosine of all notes and piano peaks weakly at 4, 8 and 16 (piano 0.53 / 0.57 vs real 0.63 / 0.65), carried by the plan chords; exact note match is flat (about 0.10 at every lag); the melody shows no peak in either measure.
- Plan chords (chord pair per bar, same at lag L): real 0.22 at lag 4 and 0.26-0.28 at lag 8 vs 0.05 at lags 1 and 3; plans 0.12-0.13 and 0.18-0.19. So the plan has about half to two thirds of the real periodicity, and the fill model does not reuse notes when the chords repeat.

## Melody repeat kinds and repeat labels (`hw2.plan.repeat_label`, `data/tokens/{plan,fill}_rep`)

Melody bars of POP909 (54968), each against the bars 2, 4, 8 and 16 back:

- Exact repeat (80% or more of the notes match in start and pitch): 24.5%. Velocity is identical in 59% of exact repeats; median change per note 0, p90 8 (two velocity steps).
- Varied repeat: same rhythm with some pitches changed 8.4%, same pitches with timing changed 4.8% (inside varied repeats, median 45% of notes match exactly).
- Same rhythm, new pitches: 8.1%. New material: 50.9%.
- Overlapping views of the same bars: earlier bar kept plus added notes 5.7%, subset of the earlier bar 4.1%, transposed copy 3.3% (mostly by a fourth, fifth or whole step). Repeats that only match after a time shift inside the bar: 0.3%.
- Tempo inside a song is near constant: beat-to-beat variation median 0 (p90 1.7%); last vs first quarter of the song differs by more than 5% in 0.9% of songs. Not modelled.

Repeat label per bar (plan and fill prefix, right after `Bar`): `Rep_exact_L`, `Rep_var_L`, `Rep_rhythm_L` (L = 2, 4, 8, 16; strongest kind wins, then the shorter lag) or `Rep_none`. Thresholds: exact = (start, pitch) Jaccard >= 0.8; var = pitch multiset Jaccard >= 0.6 and pitch-class cosine >= 0.85; rhythm = start Jaccard >= 0.8 with exact < 0.5. Added/removed-note and transposed kinds are left for later; they fall partly under var and rhythm.

## Form rule with exact melody copy (`--form`, `--return_as copy`; `outputs/plan_fill_v2/form_*`)

Reference runs for the form rule: the repeated section copies the plan and the melody notes, the fill model writes new piano. Flags `--min_piano_bin 1 --tempo_range 60,125 --cadence`, seed 0, 20 pieces each; the I4 intro puts the melody entry at bar 5, as `--melody_by_bar 5` does in `gaps_intro_cadence`.

- `form_IABAE` (`I4A8B8A8E4`, A returns 16 bars later): the theme comes back in 100% of pieces by melody (by construction), 79% by all notes, 58% by piano (real first 32 bars: 39-54%). Bar pairs 16 apart (8 of 12 are A to A'): melody exact match 0.62; piano cosine 0.79 with exact match 0.20, so the piano keeps the harmony and changes the notes, as at real returns (piano cosine 0.90, exact 35%).
- `form_IABCAE` (`I4A8B8C4A6E2`, A returns 20 bars later for 6 bars): melody theme-to-end cosine 0.78; a full 8-bar return (>= 0.85) in 50% of pieces, since the last window also holds the 2 E bars.
- Short periods stay missing: at lags 4 and 8 the melody exact match is 0.06 / 0.06 (real 0.13 / 0.15) and piano cosine 0.49 / 0.55 (real 0.63 / 0.66). The rule gives one long return, not the 4- and 8-bar repeats inside sections.
- Silence and density: silent share 0.017 (real song starts 0.016), no empty bars, longest melody gap 1.6-1.9 bars (real starts 1.7, `gaps_intro_cadence` 5.2). Notes per bar 18.3 (real starts 19.9); contrast in notes between 8-bar windows 5.2-6.4 (real starts 9.1, real inside 6.7).
