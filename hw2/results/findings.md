# Findings for the report

Data statistics (896 songs in `data/POP909/training`), analyses of generated pieces, and the decisions they led to. Each heading names the CSV next to this file and the script that writes it; numbers are copied from those CSVs.

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

## Chord label rule in use (`src/hw2/pop909.py`, `chord_token`)

Root as pitch class (enharmonics merged); quality collapsed to maj / min / dim / aug / sus; inversions and extensions dropped; N or unknown root = `Chord_N`. Share of song time: maj 54.3%, min 36.8%, sus 6.6%, N 1.5%, dim 0.7%, aug 0.1%. Detail dropped: inversions 7.4%, 7ths or extensions 7.8% of song time.

## Data split

The spec says to use the whole training set (896 songs). `train.py` holds out 45 (`--n_val 45`, seed 0) for validation loss, checkpoint selection and the real-music reference; models train on 851. No other data. State this in the report.

## Distributions of the plan features (`figures/data_distributions.png`, `progressions.csv`)

From `scripts/plot_data_stats.py`. Key per song by Krumhansl-Kessler profile match (no key file in this copy of POP909; key changes inside a song are in the next section). Chords are degrees in the relative-major frame (a minor key's tonic is vi).

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
- The plan uses this: one `Key_k` token per song and `KeyShift_k` (key relative to the first key) on the bar where the key changes (`src/hw2/plan.py`).

## Silent starts in rendered audio (2026-10-16)

Many POP909 songs start with a bar that holds only a pickup at its end, and both models learned it: the first note of a generated piece can come 1.5-5.7 s after the start (listening notes: "no sound at the start"). `hw2.render.midi_to_wav` now cuts the WAV to 0.2 s before the first sound above -50 dBFS; the MIDI keeps all 32 bars. Existing WAVs were trimmed once in place (a one-off script, removed afterwards): 239 files, 325.6 s in total (TASK B prompts and the real held-out crops needed none over 0.5 s).

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
- Repetition is the larger form gap: the best-matching pair of 8-bar windows in a real 32-bar crop scores 0.79-0.81, in generated pieces 0.64-0.70. Real crops contain a near-repeated section; generated pieces do not (before the form rule and repeat labels, see the sections below).

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

## Plan adherence (`adherence.csv`, `scripts/adherence.py`)

The written notes of each plan-fill piece, labelled again by the training rules and compared with the plan, bar by bar. Chord: the maj/min triad the piano holds longest in each half bar vs the planned chord (real held-out songs vs their `chord_midi.txt` labels give the reference, 0.85). Bins and repeat labels of real songs match by construction (1.0).

| Source | Chord match | PDen exact / within 1 | MDen exact / within 1 | Melody bars with an exact / varied / rhythm repeat |
|---|---|---|---|---|
| real held-out | 0.85 | 1 / 1 | 1 / 1 | 21.5% / 16.8% / 14.0% |
| v1 melody-first | 0.84 | 0.58 / 0.96 | 0.75 / 0.98 | 3.7% / 11.8% / 15.5% |
| v2 `gaps_intro_cadence` | 0.85 | 0.55 / 0.95 | 0.70 / 0.96 | 2.4% / 11.2% / 11.6% |
| v2 `form_IABAE` (melody copy) | 0.82 | 0.57 / 0.95 | 0.66 / 0.97 | 32.0% / 8.6% / 7.7% |
| rep `free` | 0.81 | 0.61 / 0.95 | 0.81 / 0.98 | 1.6% / 6.6% / 14.6% |

- The fill model follows the chords as well as real piano parts follow their labels, and density bins within one step 94-98% of the time.
- Generated melodies almost never repeat exactly (2-4% of melody bars vs 21.5%); varied and rhythm repeats are closer to real.

## Repeat labels in generation (`outputs/plan_fill_rep/*`, models `plan_rep`, `fill_rep`)

Models trained 2026-10-16 on `data/tokens/{plan,fill}_rep`: `plan_rep` best val 1.187 (step 3600 of 4000), `fill_rep` best val 1.354 (step 6000 of 10000, then overfits; `fill_v1` without labels: 1.325, not directly comparable since the label tokens are in the loss). Runs: `free` (labels from the plan model, same rules as `gaps_intro_cadence`), `free_drop_repeat` (same plans, labels left out of the fill prefix), `form_IABAE_var` (`--form I4A8B8A8E4 --return_as var`).

- The fill model ignores the labels. Bars the plan marks as a repeat (163 in `free`): the asked label is written 4.3% of the time, any repeat at the asked lag 13.5%. Without labels on the same plans: 21.3%. Bars marked `Rep_none` stay non-repeats 85% of the time.
- `form_IABAE_var`: the plan of A comes back (piano pitch-class cosine 0.80 at lag 16) but the melody does not (melody theme return 0%, exact match at lag 16 0.08 vs 0.62 with the copy rule).
- Lag profile unchanged: no peaks at 4 and 8 bars in `free` (melody exact 0.05 / 0.07 vs real 0.13 / 0.11-0.15).
- H1/H4/GS stay at the real level for every plan-fill run (`plan_fill_v2.csv`: H4 2.58-2.63, GS 0.95-0.97; real 32-bar crops 2.64 / 0.960), so these metrics do not see the missing repetition.
- `free` has more silence than `gaps_intro_cadence` (4.0% vs 1.7%, one 1-bar gap in 20 pieces); `free_drop_repeat` 3.0%.

### Why: the model copies under teacher forcing, sampling loses it (`repeat_probe.csv`, `scripts/repeat_probe.py`)

Teacher-forced loss on the melody tokens of the 45 held-out songs (bars with at least 16 bars of context), by the bar's true repeat kind:

| Model, labels | Exact | Varied | Rhythm | None | Pitch tokens: exact / none |
|---|---|---|---|---|---|
| `fill_rep`, true labels | 0.585 | 1.145 | 1.183 | 1.425 | 0.223 / 1.461 |
| `fill_rep`, all `Rep_none` | 0.715 | 1.156 | 1.183 | 1.428 | 0.482 / 1.465 |
| `fill_rep`, labels removed | 0.641 | 1.146 | 1.188 | 1.433 | 0.361 / 1.486 |
| `fill_v1` (no labels) | 0.554 | 1.126 | 1.178 | 1.421 | 0.274 / 1.485 |

- With the real earlier bar in context, both fill models predict exact repeats well (pitch loss 0.22-0.48 vs about 1.47 for new material), with or without labels: they copy from context.
- The label helps a little for exact repeats (0.22 with the true label vs 0.48 when told `Rep_none`) and not at all for varied and rhythm repeats.
- In generation the copy is lost: an exact bar is about 24 tokens, and once one early token is sampled differently the bar is no longer a repeat. Inspecting one labelled bar: the model put p = 0.999 on `Pos_0` and p ≈ 0 on the source bar's first onset, so the label did not steer even the first note (why is not checked).

### Fix at sampling time: `--repeat_bias` (`RepeatGuide` in `src/hw2/plan_generate.py`)

**Bug found 2026-10-17; the table below is from the old guide:** `RepeatGuide` never uses up a source note whose pitch the model changed, so `Rep_rhythm` and changed `Rep_var` bars stack melody notes on one onset (up to 32). Every run with β 8 or more has such clusters (4-17 of 20 pieces); β 4 has none. See `docs/listening_issues.md` section 0. Fixed in code the same day (plus a rule that a part cannot strike the same position and pitch twice in a bar); a CPU test with a random-weight model on a real plan gives at most 6 melody notes on one onset vs 32 with the old guide. Re-run 2026-10-17 into `outputs/plan_fill_rep_fix/*` (same plans and seeds); results in the next section.

On a melody bar whose plan label is `Rep_<kind>_<L>`, add a logit bonus β to the source bar's next note (bar b-L, moved to the bar's key), matched by position so one changed note does not shift the rest: exact = onset, pitch, velocity, duration; var = the same at β/2; rhythm = all but pitch. Same plans as `free` (`--plans_from`), 20 pieces each; `form_IABAE_var` plans with β 8.

| Run | Asked kind written at the lag | Any repeat at the asked lag | Melody bars: exact / var / rhythm | MDen within 1 | Melody exact match, lag 2 / 4 / 8 |
|---|---|---|---|---|---|
| real held-out | 1 | 1 | 21.5% / 16.8% / 14.0% | 1 | 0.095 / 0.127 / 0.112 |
| `free` (β 0) | 0.05 | 0.14 | 1.6% / 6.6% / 14.6% | 0.98 | 0.059 / 0.053 / 0.065 |
| `free_bias4` | 0.18 | 0.36 | 6.6% / 11.4% / 11.2% | 0.97 | 0.084 / 0.066 / 0.090 |
| `free_bias8` | 0.26 | 0.44 | 11.9% / 9.6% / 6.4% | 0.94 | 0.084 / 0.075 / 0.101 |
| `free_bias16` | 0.33 | 0.62 | 19.0% / 6.6% / 6.1% | 0.86 | 0.092 / 0.071 / 0.099 |

- The bonus works and scales with β: at β 8 the lag-2 and lag-8 melody repeats reach the real level; lag 4 stays low (0.075 vs 0.13).
- Lag 4 stays low, and the plan asks for it less often than real songs have it: repeat labels on melody bars by lag, plans of `free` vs real held-out first 32 bars: lag 2 13.5% vs 14.3%, lag 4 9.4% vs 15.7%, lag 8 10.8% vs 11.0%, lag 16 3.7% vs 2.9%, none 62.7% vs 56.2% (437 vs 1002 melody bars).
- Cost: the copy fights the planned density at large β (MDen within one bin 0.98 to 0.86 at β 16), and the model's own loss on its output rises (sampled NLL 1.32, 1.36, 1.41 for β 4, 8, 16). β 8 is the middle ground.
- `form_IABAE_var_bias8`: the melody of A now comes back with changes. Melody theme return 42% of pieces (real 32-bar crops 43%, β 0: 0%, exact copy rule: 100%); exact match at lag 16 0.47 (copy rule 0.62); melody bars 24.3% exact, 13.9% varied, 10.1% rhythm repeats.
- H1/H4/GS do not move (H4 2.59-2.64, GS 0.96-0.97).

### Re-run with the fixed guide (`outputs/plan_fill_rep_fix/*`; `adherence_fix.csv`, `return_stats_fix.csv`, `silence_stats_fix.csv`)

Same plans and seeds as the old runs; only `RepeatGuide` differs. Old -> fixed.

| Run | Max melody notes on one onset | Pieces with 4+ | Asked kind written at the lag | Any repeat at the asked lag | Melody bars exact / var / rhythm (fixed) | MDen within 1 (fixed) | Melody returns |
|---|---|---|---|---|---|---|---|
| `free_bias8` | 32 -> 2 | 4 -> 0 | 0.25 -> 0.39 | 0.44 -> 0.60 | 13.3% / 8.9% / 11.7% | 0.96 | 0.08 -> 0.08 |
| `free_bias16` | 32 -> 2 | 17 -> 0 | 0.31 -> 0.47 | 0.62 -> 0.82 | 18.5% / 5.0% / 14.2% | 0.88 | 0.15 -> 0.23 |
| `free_pbias4_bias8` | 32 -> 5 | 7 -> 1 | 0.33 -> 0.51 | 0.54 -> 0.76 | 14.5% / 8.0% / 16.3% | 0.98 | 0.07 -> 0.20 |
| `form_IABAE_var_bias8` | 32 -> 2 | 6 -> 0 | 0.17 -> 0.22 | 0.57 -> 0.73 | 27.8% / 12.7% / 14.7% | 0.96 | 0.42 -> 0.63 |

- The note clusters are gone, and more asked repeats are written at the asked lag in all four runs. Why adherence rose is not checked.
- Silent share does not change (2.6-3.9%).
- Real held-out: melody bars 21.5% / 16.8% / 14.0%, MDen within 1 = 1 by definition.

### Plan side: `--plan_repeat_bias` (`PLAN_REPEAT_SLOTS` in `src/hw2/plan_generate.py`)

In POP909 a melody repeat usually keeps its source bar's chords and melody bins; the plan model does not couple them, so the fill bias fights the plan. Share of labelled melody bars whose chords / melody bins equal the source bar's (real: held-out first 32 bars; plans: 20 each, seed 0-19):

| | Real | Plan β 0 | Plan β 4 | Plan β 8 |
|---|---|---|---|---|
| exact: chords / melody bins | 0.68 / 0.80 | 0.20 / 0.14 | 0.70 / 0.75 | 0.88 / 0.98 |
| var | 0.36 / 0.27 | 0.19 / 0.14 | 0.37 / 0.50 | 0.79 / 0.87 |
| rhythm | 0.15 / 0.40 | 0.18 / 0.19 | 0.10 / 0.85 | 0.18 / 0.93 |

Plan β 4 (bonus on chords + melody bins for exact, half for var, melody bins only for rhythm) brings exact and varied repeats to the real coupling. With fill β 8 (`free_pbias4_bias8`, new plans, same rules as `free`) vs fill β 8 alone (`free_bias8`):

- Asked kind written at the lag 0.34 vs 0.26, any repeat at the lag 0.54 vs 0.44; melody density within one bin 0.96 vs 0.94.
- Melody exact match at lags 2 / 4 / 8: 0.098 / 0.084 / 0.122 (real 0.095 / 0.127 / 0.112). Lags 2 and 8 are at the real level, lag 4 still low.
- Piano periodicity stays below real (pitch-class cosine at lag 4 0.51 vs 0.64): the plan's chord pairs repeat at lag 4 in 11-15% of bars vs 22% in POP909. Not addressed.
- H4 2.59, GS 0.959, sampled NLL 1.32.

## TASK B with plan-fill (`task_b_plan.csv`, `task_b_plan_fade.csv`, `outputs/task_b_plan/*`)

`make task-b-plan PLAN_CKPT=checkpoints/plan_rep/best FILL_CKPT=checkpoints/fill_rep/best PLAN_RULES="--min_piano_bin 1 --cadence"`: each prompt folder is labelled by the training rules, the plan model continues the 8-bar plan to 32 bars, the fill model writes bars 9-32 after the forced prompt notes. The prompt bars in the output match the TA prompt tokens note for note. One plan per prompt, filled with the same three configs as the main model's `TASK B`. Prompt 001 has no MELODY track; `--melody_by_bar 9` makes the melody enter in bar 9.

| Config | H4 | GS | Pitch-class similarity to the prompt, bars 9-32 | Main model `step_6000`, same config (`task_b_fade.csv`) |
|---|---|---|---|---|
| t0.8 p0.9 | 2.49 | 0.965 | 0.743 | 0.758 |
| t1.0 p0.95 | 2.50 | 0.966 | 0.753 | 0.747 |
| t1.2 p1.0 | 2.61 | 0.936 | 0.802 | 0.665 |

Real held-out continuations: 0.773. The plan keeps the key and harmony at t1.2, where the main model drifts. 3 pieces per config.

## Candidate sets for the plan-fill system (2026-10-16 night; `task_a_plan.csv`, `task_b_plan.csv`, `task_b_plan_fade.csv`)

Not chosen yet; made so the choice can be made by ear. Models `plan_rep` + `fill_rep`, `--repeat_bias 8`, one plan per piece shared by the three fill configs (the same configs as the main model's `TASK B`).

`TASK A`: `outputs/task_a_plan_form_var_bias8/<config>`, `--form I4A8B8A8E4 --return_as var --min_piano_bin 1 --tempo_range 60,125 --cadence`, 20 pieces each.

| Config | Melody theme return | Exact / varied repeats (melody bars) | Chord match | H4 | GS | Silent share |
|---|---|---|---|---|---|---|
| real 32-bar crops / held-out songs | 43% | 21.5% / 16.8% | 0.85 | 2.64 | 0.960 | 1.4% |
| t0.8 p0.9 | 84% | 39.5% / 11.5% | 0.90 | 2.55 | 0.980 | 5.5% |
| t1.0 p0.95 | 74% | 30.6% / 14.5% | 0.85 | 2.61 | 0.972 | 5.0% |
| t1.2 p1.0 | 11% | 10.9% / 11.3% | 0.69 | 2.69 | 0.917 | 1.8% |

- The configs span too repetitive (t0.8), close to real with more returns (t1.0 p0.95) and drifting off the planned chords (t1.2).
- The extra silence at t0.8 and t1.0 p0.95 is spread over all bars (92-97% sounding per bar vs 98-99% for real songs). Note counts per piece are the same as at t1.2 and mean durations only slightly shorter (11.6-11.7 vs 12.9 sixteenths), so the cause is not found yet. It sits at the end of the bar: silent share on beat 4 of bars 5-32 is 17% at t1.0 p0.95 vs 7.6% at t1.2 and 8% in `gaps_intro_cadence` (beats 1-3: 3-7% vs 1-2%); the last sixteenth of the bar is silent 50% vs 19% / 38% of the time (`pbias4_bias8` set, grid assumes 4 beats). Notes stop before the barline more often under truncated sampling; why is not checked.

Same with `--plan_repeat_bias 4` added (`outputs/task_a_plan_form_var_pbias4_bias8/<config>`, new plans): melody theme return 89% / 78% / 11%, exact repeats 37.9% / 30.6% / 16.5%, melody density within one bin 0.977 / 0.961 / 0.955 (vs 0.959 / 0.955 / 0.947), silent share 4.7% / 4.5% / 1.8%, H4 2.54 / 2.61 / 2.69, GS 0.981 / 0.970 / 0.918. With the form on, the plan bias changes little. In both sets at t1.0 p0.95 the melody exact match at lags 4 and 8 is at the real level (0.115-0.134 vs 0.127 / 0.112).

`TASK B`: `outputs/task_b_plan_bias8/<config>` (rules as `task_b_plan`). Pitch-class similarity to the prompt over bars 9-32: 0.749 / 0.769 / 0.773 for t0.8 / t1.0 / t1.2 (without the bias 0.743 / 0.753 / 0.802; real 0.773). H4 2.49 / 2.51 / 2.60, GS 0.965 / 0.965 / 0.928.
With `--plan_repeat_bias 4` as well (`outputs/task_b_plan_pbias4_bias8`): 0.743 / 0.758 / 0.778, H4 2.55 / 2.54 / 2.65, GS 0.968 / 0.966 / 0.915.

## Loss review (`loss_review.csv`, `scripts/loss_review.py`; 45 held-out songs, teacher forced)

Mean NLL per token on the held-out songs. Windows of whole bars; after the first window only bars with 16 bars of context count.

| Checkpoint | All / melody | Piano | Pitch | Pos | Dur | Vel |
|---|---|---|---|---|---|---|
| `gpt2_ppb16/step_6000` | 1.44 | | 1.32 | 0.57 | 2.17 | 1.70 |
| `gpt2_ppb16/step_10000` | 1.47 | | 1.16 | 0.59 | 2.43 | 1.71 |
| `gpt2_ppb16/loss_0.40` | 2.70 | | 1.33 | 1.00 | 4.99 | 3.55 |
| `fill_rep/step_2000` | 1.57 | 1.72 | 1.78 / 1.55 | 0.65 / 0.75 | 2.08 / 2.58 | 1.77 / 2.00 |
| `fill_rep/step_6000` | 1.16 | 1.39 | 1.10 / 1.07 | 0.38 / 0.53 | 1.67 / 2.23 | 1.48 / 1.73 |
| `fill_rep/step_10000` | 1.22 | 1.46 | 1.19 / 1.14 | 0.39 / 0.54 | 1.79 / 2.39 | 1.51 / 1.76 |

Fill columns with two values are melody / piano. Plan model (`plan_rep/best`, `last`): 1.19 on plan tokens either way.

- The main model's held-out loss at `loss_0.40` is almost twice that of `step_6000`. Most of the rise is in duration and velocity; pitch stays at 1.33.
- The fill model reads the plan. At `step_6000` the melody loss is 1.16 with the true labels, 1.28 with the labels removed and 1.72 with each song's labels shuffled between its bars (piano 1.39 / 1.45 / 1.67). Wrong labels cost more than no labels.
- Fill held-out loss is lowest near `step_6000` and rises by `step_10000`, the same pattern as the main model.
- No setting change for the longer runs: the fill model already uses its labels, so label dropout stays as it is.

## TASK A candidate sets with the fixed guide and the longer runs (2026-10-17; `outputs/cand_a/*`, `cand_a_*.csv`)

20 pieces of 32 bars per set, single-stage with `--max_beats` and one tempo, plan-fill with `--form I4A8B8A8E4 --return_as var --repeat_bias 8`. Plan-fill sets are named `<plan>_<fill>`: `newplan` = `plan_rep_long/loss_0.60`, `oldplan` = `plan_rep/best`, fill = `fill_rep_long/loss_0.40` or `loss_0.60`.

| Set | H1 | H4 | GS | silent share | empty bars | chord match | asked repeat written |
|---|---|---|---|---|---|---|---|
| real held-out | | | | 0.014 | 0.044 | 0.852 | 1.00 |
| `single_t1_p1` | 2.18 | 2.62 | 0.970 | 0.009 | 0 | | |
| `single_t1_p095` | 2.12 | 2.59 | 0.971 | 0.009 | 0 | | |
| `single_t09_p1` | 2.12 | 2.58 | 0.968 | 0.009 | 0 | | |
| `newplan_fill040` | 2.04 | 2.57 | 0.962 | 0.033 | 0.100 | 0.803 | 0.220 |
| `oldplan_fill040` | 2.11 | 2.65 | 0.958 | 0.016 | 0 | 0.820 | 0.205 |
| `oldplan_fill060` | 2.09 | 2.64 | 0.962 | 0.030 | 0 | 0.805 | 0.138 |

- **H1/H4/GS still separate nothing:** all six sets sit in the same range as before (H4 2.58-2.65, GS 0.958-0.971).
- **Copying:** `bar_copy` is 0 for every set; 8-note copy 1.2-2.8%.
- **Silence:** `newplan_fill040` has 1-bar gaps (20% of pieces) and empty bars (10%); `oldplan_fill040` is closest to the real silent share.
- **Repeat labels:** the fill model writes an asked repeat in 14-22% of the bars; the fix did not make it follow the labels.
- **Length:** plan-fill pieces are shorter in tokens (2160-2520) than single-stage (3180-3430).

## Stem loudness: melody vs piano (`stem_gap.csv`, `scripts/stem_gap.py`)

Melody stem minus piano stem, integrated LUFS of each rendered stem (ffmpeg `ebur128`, gated), real songs seconds 8-72.

| Source | pieces | mean dB | median dB | sd |
|---|---|---|---|---|
| real POP909 | 299 | -2.73 | -2.80 | 2.42 |
| `oldplan_fill040` (real-velocity map) | 20 | -2.80 | -4.00 | 2.73 |
| `newplan_fill040` | 20 | -2.31 | -2.00 | 2.15 |
| `oldplan_fill060` | 20 | -3.32 | -3.40 | 2.13 |
| same three with the `min_gap` lift | 20 each | -1.98 / -1.65 / -2.30 | -3.00 / -1.90 / -1.95 | 2.11 / 1.62 / 1.66 |

- The melody stem is quieter than the piano stem in real songs too (sparser notes). The generated sets match that level.
- The lift in the listening build moves plan-fill about 0.8 dB above the real mean; single-stage clips have no melody track and get no lift.

## Listening-grade predictor (2026-10-17)

Method and pipeline: `docs/grade_model.md`. Fit on T1, T2 (13 decisive A/B pairs) and the four pilot sessions (56 clips).

- **Leave-one-out, A/B pairs (`results/grade_model_cv.csv`):** `v1` 9 to 10 of 13 correct (0.69 at L2 0.1 and 1, 0.77 at 10); `v2` 10 of 13 at every penalty. Coin flip 0.5; "more notes wins" 0.31. On 13 pairs the error bar is about ±25 points.
- **Leave-one-session-out, pilot:** mean Spearman of score vs `overall` 0.22 to 0.33 (`v1`), 0.46 to 0.49 (`v2`, optimistic: its features were picked on this data).
- **Strongest single features (`results/grade_single_features.csv`):** `half_pc_sim` (first vs second half pitch-class similarity; the A/B winner has the higher value in 11 of 13 pairs, pilot Spearman 0.45), `key_fit` (9 of 13, 0.20), `one_note_bars` (pilot Spearman −0.45). `notes_per_bar` disagrees between A/B (winner has more notes in 4 of 13) and pilot (+0.45).
- **Tags (`results/grade_tags.csv`, AUC with vs without the tag):** `silent_gap` vs `longest_onset_gap_beats` 0.84; `wrong_notes` vs `key_fit` 0.27 (lower fit with the tag); `cut_off_ending` vs `density_cv` 0.71; `abrupt_start` vs `opening_high_share` 0.51 (not caught).
- **Loss evaluation:** predicted grade per checkpoint of the main model in `results/grade_pred_exp2_v1_sets.csv`, `results/figures/grade_vs_loss_main_v1.png` (and `_v2`); plan-fill runs follow after their training.

### Grade predictor on T3 (first held-out session, 2026-10-17)

`grade_model.py check` before refitting (`results/grade_model_check.csv`): `v1` 5 of 9 pairs, `v2` 3 of 9. Neither beats a coin flip on T3; the T3 pairs share their first 8 bars (the prompt) and differ only in the continuation, which the whole-piece features see only in part. After refitting on 22 decisive pairs, leave-one-out is 0.77 (`v1`) and 0.73 (`v2`) at L2 0.1, "more notes wins" 0.45. `v2` lost its edge on new data, so `predict` keeps `v1`. The predictor stays a descriptive tool for the report, not a stand-in for listening.

### T3: TASK B sampling configs (`results/listening/t3_continuations.csv`)

9 pairs (3 prompts x 3 configs, each piece in two pairs): `t1 p0.95` won 6 of 6 of its pairs, `t1 p1` beat `t0.9 p1` 2 of 3. Element marks (`t3_continuations_elements.csv`): `t1 p0.95` had the highest mean level on all six elements; `t0.9 p1` collected the most issue tags (wrong notes 4, cut-off ending 3). Best config for every prompt: `t1 p0.95`.

Comments, checked against the MIDI (`scripts` not changed; checks in this section only):

- **"Weird key change in the last 8 bars" (prompt 3, `t0.9 p1`, pairs 03 and 07):** bars 29-32 move to C# (fit of the prompt's A key 0.48 in that window); every other piece keeps the prompt key at 0.8 or more in every 4-bar window. The same piece still won pair 07 against `t1 p1`, so the judgment is relative.
- **"Weird key change dropping harmony degree" (prompt 3, `t1 p1`, pair 05):** no 4-bar window leaves the prompt key, so this is a chord-level problem (wrong scale degrees inside the key), not a modulation.
- **"Sounds broken", "chords broken in bars 9-24" (prompt 2, `t0.9 p1`, pairs 06 and 09, the same piece):** 2.1% of its onsets have two sounding notes a semitone apart, against 0.4% for the `t1 p0.95` piece on that prompt; it also re-strikes a sounding key 104 times (58). The re-strikes are cut at render time (`one_keyboard`, since 2026-10-16), and the decoder round trip passes, so no program or encoding fault was found; the clashes are in the generated notes.

Why the predictor missed T3: the pairs share the 8-bar prompt, so whole-piece features dilute the 24-bar difference; 5 of the 9 `v1` predictions sat between 0.46 and 0.54 (no signal). The deciding reasons were harmony (key change, clashing chords), which `key_fit` sees only over the whole piece; it got pair 03 right and pair 07 wrong (the same weird piece won against a worse one). `v2`'s `half_pc_sim` hurt here because the first half includes the shared prompt. Next version (`v3`, to be tested on the next session before fitting): features on the generated bars only, prompt-key fit per 4-bar window, semitone-clash share.

`v3` built (2026-10-17 18:20): `v1` plus `opening_key_fit_min4` and `semitone_clash_share`, TASK B features on the generated bars only. Fit on T1-T3 and the pilot; leave-one-out 0.68 on 22 pairs (in-sample design, not a test). First test: T4, scored before refitting.

## Predicted grade vs loss for the longer plan-fill runs (2026-10-17; `outputs/sweep/*`, `grade_pred_<run>*.csv`, `sweep_silence.csv`)

20 pieces per checkpoint. Fill sweep (`fill_rep_60k`): the 20 plans of `pf_oldplan_fill040` and the same seeds, so piece `NN` differs from `pf_oldplan_fill040/NN` only in the fill checkpoint. Plan sweep (`plan_rep_24k`): new plans, fill `fill_rep_long/loss_0.40`. Figures: `figures/grade_vs_loss_<run>.png` (`v1`) and `_v3`.

- **Predicted win vs a real opening (`win_vs_real_mean`):** `v1` 0.40-0.48 (fill), 0.40-0.49 (plan); `v3` 0.37-0.46 and 0.38-0.48. For comparison `pf_oldplan_fill040` 0.43, `single_t1_p1` 0.46 (`v1`, `grade_pred_cand_a_sets.csv`). The per-set standard error is about 0.02 (sd about 0.1 over 20 pieces), so no level stands out and there is no trend with training loss. In both runs the `best` checkpoint (lowest held-out loss, early: step 6000 fill, 3000 plan) has the lowest prediction.
- **Silence:** every new fill checkpoint has more silent share (0.020-0.028) and a longer longest gap (0.85-1.36 beats) than `pf_oldplan_fill040` (0.016, 0.69) on the same plans; real openings 0.014, 0.88. Plan sweep: silent share 0.019-0.040, empty bars in 0-0.15 of pieces.
- **T4 picks:** the last checkpoint of each run (the longest training, which is what the run was for): `fill_rep_60k/last` against `pf_oldplan_fill040` on the same plans, and `plan_rep_24k/last` (with the old fill) against `pf_oldplan_fill040`.

## Fill model on real plans (2026-10-17; `outputs/real_plan/fill_rep_long_loss_0.40`, `real_plan_*.csv`, `grade_pred_real_plan*.csv`)

The fill checkpoint of `pf_oldplan_fill040` (`fill_rep_long/loss_0.40`) fills the real plans (first 32 bars) of all 45 held-out songs (`generate_plan_fill.py --real_plans`, `--return_as var --repeat_bias 8`). Plan errors and fill errors separate: what stays wrong with a real plan is the fill model's.

| | real held-out | real plan + fill | `pf_oldplan_fill040` (generated plan) |
|---|---|---|---|
| chord match | 0.852 | 0.811 | 0.820 |
| piano density bin exact / within 1 | 1 / 1 | 0.597 / 0.958 | 0.547 / 0.941 |
| melody density bin exact / within 1 | 1 / 1 | 0.759 / 0.966 | 0.658 / 0.945 |
| bars with an asked repeat | 0.39 (whole songs) | 0.30 | 0.46 |
| asked repeat written | 1 | 0.440 | 0.205 |
| asked repeat at any lag | 1 | 0.743 | 0.486 |
| silent share (song start) | 0.014 | 0.024 | 0.016 |
| longest gap, beats | 0.88 | 1.43 | 0.69 |
| empty bars | 0.044 | 0.044 | 0 |
| predicted win vs real (`v1` / `v3`) | 0.51 / 0.52 (`heldout_open32`, 12) | 0.45 / 0.43 | 0.43 / 0.41 |

- **Chords are the fill model's limit:** the chord match is the same with real and generated plans (0.81-0.82, real 0.85).
- **Repeats are mostly a plan problem:** with real plans the fill writes the asked repeat twice as often (0.44 vs 0.21). Generated plans ask for a repeat in 46% of bars (real 30% in the first 32 bars). Why the fill model honors fewer of them is not checked yet.
- **Silence is a fill problem:** with real plans the silent share and the longest gap are above the real level (0.024 vs 0.014, 1.43 vs 0.88 beats); the empty bars match the real songs (empty intro bars in the plan).
- **Density bins** are followed better with real plans (melody exact 0.76 vs 0.66).
- 12 of the 45 songs are also in `outputs/ref/heldout_open32`, so a later test can pair a real opening with the fill of its own plan.

## T4: longer plan-fill training and the single-stage configs (2026-10-18; `results/listening/t4_openings.csv`, `t4_openings_elements.csv`)

7 pairs, rated 2026-10-18 02:24-03:00. `pref` runs from −2 (A much better) to +2 (B much better).

- **Longer fill training, same plans and seeds:** `fill_rep_60k/last` beat `fill_rep_long/loss_0.40` (`pf_oldplan_fill040`) in 2 of 2 pairs, once on each side (pref −1 on side A, +2 on side B). Deciding reasons: form 2, harmony 2. This goes against the sweep, which predicted no gain and found more silence for every new fill checkpoint (section "Predicted grade vs loss for the longer plan-fill runs").
- **Longer plan training (fill `fill_rep_long/loss_0.40`):** `plan_rep_24k/last` vs `plan_rep/best` 1-1, each pair won by side B.
- **Single-stage configs, one pair each:** `t0.9 p1` beat `t1 p1` (+2) and `t1 p0.95` (+2); `t1 p0.95` beat `t1 p1` (+2). Element marks: `t0.9 p1` mean level 0.5 to 1.0 on all six elements, `t1 p1` −1.0 to 0.0.
- **Order bias:** side B won 6 of 7 (sign test p 0.125); T1-T3 together: A 10, B 12, same 2; all four sessions: A 11, B 18 (p 0.26). All three single-stage pairs were won by side B, and `t0.9 p1` was on side B in both of its pairs: `listening.py pairs` puts the first named system of a matchup on side A in its first pair, so with one pair per matchup the command fixed the sides.

Comments, checked against the MIDI:

- **"A too fast, too many notes, so more wrong notes" (pair 02, `single_t1_p1/04` vs `single_t1_p095/01`):** tempo 120 vs 60, 12.2 vs 9.7 notes per second, semitone-clash share 0.030 vs 0.019, key fit 0.985 vs 0.992 (`grade_features.csv`).
- **"Weird chord in the latter 8 bars" (pair 07, `single_t09_p1/10`, the winner):** bars 25-32 have the lowest key fit of its 20-piece set (0.938, set mean 0.992) and a clash share of 0.038 (set mean 0.019).

Decisions (rules in `docs/status.md`):

- **Plan-fill fill checkpoint:** `fill_rep_60k/last` (most wins). **Plan checkpoint:** `plan_rep/best` stays (tie).
- **TASK A config:** `t0.9 p1` has the most wins, but every pair was decided by side B in a session that B won 6 to 1, so it is not a clear result. TASK A stays `t1 p1`. (A side-swapped re-test was planned for T5 and dropped on 2026-10-18 in favour of the cut test.)

### Grade predictor on T4 (second held-out session)

`grade_model.py check` before refitting: `v1` 5 of 7, `v2` 4 of 7, `v3` 6 of 7. `v3` missed pair 03 (`pf_oldplan_fill040/07` vs `plan_rep_24k/last/03`) and got pair 05 right by a score margin of 0.01, so its edge over `v1` is one near-tie pair; "side B wins" would also get 6 of 7. After refitting on 29 decisive pairs, leave-one-out at L2 10 (the penalty in use): `v1` 0.66, `v2` 0.62, `v3` 0.79; "more notes wins" 0.52. By the rule in `docs/grade_model.md`, `predict` now uses `v3`. `grade_pred_*` files written before 2026-10-18 without a variant suffix are `v1`.

## Sudden loud notes (2026-10-18; `velocity_spikes.csv`, `scripts/velocity_spikes.py`)

Share of notes at least 20 above their bar's median velocity (all tracks merged, the `velocity_spike_share` rule), first 32 bars, after `one_keyboard`. Real: every tenth training song (90). "map" is the listening build's real-velocity map, "lift" adds the `min_gap` melody lift.

| | spike share | share of spikes on melody notes | spikes against the note's own track | melody-piano median gap (sd) |
|---|---|---|---|---|
| real openings | 0.077 | 0.59 | 0.037 | 14.0 (11.7) |
| `single_t1_p1` written / map | 0.064 / 0.046 | (one track) | 0.064 / 0.046 | |
| `pf_oldplan_fill040` written / map / map + lift | 0.060 / 0.085 / 0.095 | 0.79 / 0.67 / 0.71 | 0.021 / 0.041 / 0.041 | 10.0 / 9.0 / 14.5 |
| `fill_rep_60k/last` written / map / map + lift | 0.086 / 0.072 / 0.074 | 0.67 / 0.57 / 0.59 | 0.044 / 0.044 / 0.044 | 16.0 / 9.5 / 11.0 |

- **Cause:** most spikes are melody notes above a bar median set by the more numerous piano notes, in real songs as in plan-fill. Against their own track, plan-fill notes spike at the real level (0.041-0.044 vs 0.037). The 6% real level quoted before (`docs/listening_issues.md` section 11) was whole songs; real openings are at 7.7%.
- **Plan-fill as played** is at 7.4% (`fill_rep_60k/last`) to 9.5% (`pf_oldplan_fill040`) against 7.7% real; the lift adds 0.2 and 1.0 points.
- **Submitted WAVs are quiet:** `single_t1_p1` median −33.8 LUFS (−38.3 to −30.1), TASK B `t1_k0_p0.95` −34.2 to −31.2, while every rated clip played at −18 LUFS.

Decisions: keep the `min_gap` lift for plan-fill audio (the rated clips have it). Single-stage velocities stay as written (spike share under the real level, quantiles within 2 of real), and the final TASK A and TASK B WAVs get the listening build's loudness step (gain to −18 LUFS, peak limiter) so the submission sounds like what was rated.
