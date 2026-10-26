# Findings for the report

Measured results only: each heading names the CSV next to this file (or the output folder) and the script that writes it, numbers are copied from those files, and "as of" is the date of that data (the last commit of its CSV, or the date in the heading). Decisions and plans are in `docs/status.md`. Results that rest on an assumption found wrong later are listed under "Archived" at the end with the reason; do not reuse them. Cleaned 2026-10-18; the text before the cleanup is in commit `2d826a7` (`git show 2d826a7:hw2/results/findings.md`).

## Data: POP909 (896 songs in `data/POP909/training`)

### Notes heard longest give the chord (`chord_from_notes.csv`) · as of 2026-10-16

For each maj/min segment of `chord_midi.txt`, add up how long each pitch class sounds (dur) or how many times it is pressed (count), pick the major or minor triad with the largest total, compare with the label. Weighted by segment time. The true chord boundaries are given, so this is an upper bound for the rule.

| Notes from | Triad match, dur | Triad match, count | Longest pitch class = root, dur | Longest = root, count |
|---|---|---|---|---|
| all tracks | 94.2% | 90.6% | 70.5% | 55.2% |
| PIANO + BRIDGE | 95.4% | 92.6% | 76.4% | 61.7% |
| PIANO | 97.4% | 94.9% | 84.6% | 64.9% |
| BRIDGE | 33.8% | 31.0% | 32.4% | 28.6% |
| MELODY | 28.4% | 26.4% | 30.4% | 28.0% |

- Duration weighting beats press counting for every source; the gap is largest for "longest = root" (PIANO 84.6% vs 64.9%).
- PIANO carries the harmony; MELODY and BRIDGE alone say little about the chord with this rule.

### Notes and mean duration per bar, by track (`track_stats.csv`) · as of 2026-10-16

Bars are downbeat to downbeat of `beat_midi.txt` (72748 bars). Mean duration = total sounding time / presses, in beats.

| Track | Notes per bar p10 / p50 / p90 | Mean duration (beats) p10 / p50 / p90 | Empty bars |
|---|---|---|---|
| PIANO | 7 / 13 / 20 | 0.48 / 0.80 / 1.50 | 1.6% |
| BRIDGE | 0 / 2 / 8 | 0.33 / 0.84 / 2.29 | 35.1% |
| MELODY | 0 / 4 / 8 | 0.29 / 0.55 / 1.50 | 24.5% |

### Token budget vs the 2048 context (`token_budget.csv`) · as of 2026-10-16

Sections of the `bonus_chd` tokens: plan = `Bar Chd <Pos Chord pairs>`, melody = `Mel` + melody notes, piano = `Acc` + piano notes (BRIDGE not included). Windows are 32 bars, stride 8.

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

### Chord label rule in use (`src/hw2/pop909.py`, `chord_token`) · as of 2026-10-16

Root as pitch class (enharmonics merged); quality collapsed to maj / min / dim / aug / sus; inversions and extensions dropped; N or unknown root = `Chord_N`. Share of song time: maj 54.3%, min 36.8%, sus 6.6%, N 1.5%, dim 0.7%, aug 0.1%. Detail dropped: inversions 7.4%, 7ths or extensions 7.8% of song time. So every model sees triads only; a melody note on a labelled 7th counts as off-chord in every harmony measure below, for real and generated pieces alike.

### Data split

The spec says to use the whole training set (896 songs). `train.py` holds out 45 (`--n_val 45`, seed 0) for validation loss, checkpoint selection and the real-music reference; models train on 851. No other data. State this in the report.

### Distributions of the plan features (`figures/data_distributions.png`, `progressions.csv`) · as of 2026-10-16

From `scripts/plot_data_stats.py`. Key per song by Krumhansl-Kessler profile match (no key file in this copy of POP909). Chords are degrees in the relative-major frame (a minor key's tonic is vi).

- **Song length:** 77 bars median (p10 57, p90 112.5, range 12 to 193).
- **Tempo:** 74 BPM median (p10 60, p90 116); 4 songs under 50 and 8 over 140 BPM.
- **Meter:** 94.6% of bars have 4 beats, 3.4% have 6, 1.3% have 2; by most common bar, 868 songs are 4-beat and 28 are 6-beat.
- **Key:** 63.3% major, 36.7% minor; most common C (70 songs), G (65), D (54).
- **Chords, share of song time:** vi 17.7%, I 17.2%, V 14.8%, IV 13.4%, ii 8.5%, iii 7.4%; everything else under 3% each. Six diatonic chords cover about 79%.
- **Chord types per song:** 11 median (p10 7, p90 18).
- **Chord changes per bar:** 0: 8.7%, 1: 47.4%, 2: 38.3%, 3: 5.1%, 4+: 0.5%.
- **Top progressions:** 2-chord V->I 6.2%, IV->V 5.5%, V->vi 3.7%, vi->IV 3.6%; 4-chord vi-IV-V-I 1.0%, vi-ii-V-I 0.9%, V-vi-IV-V 0.7%.
- **Register:** PIANO mean pitch per bar peaks below middle C (about 56), MELODY and BRIDGE overlap around 72.
- **PIANO notes per bar** has a spike at 16 (steady 16th-note patterns over 4 beats).

### Key changes (`key_changes.csv`, `figures/key_changes.png`) · as of 2026-10-16

From `scripts/key_changes.py`. Local key per bar by Viterbi over the 12 major scales (a minor key counts as its relative major). Bar cost = 1 - share of sounding time on the scale; a change costs a switch penalty.

| Penalty | Songs with a change | Changes | Interval +1 | Interval +2 | Median position in song | Median bars after |
|---|---|---|---|---|---|---|
| 2 | 132 (14.7%) | 160 | 78 | 37 | 0.75 | 17 |
| 4 (default) | 107 (11.9%) | 111 | 74 | 24 | 0.77 | 17 |
| 8 | 69 (7.7%) | 70 | 62 | 4 | 0.77 | 16 |

- About 8-15% of songs change key, almost always once, mostly up 1 or 2 semitones, about three quarters of the way in, for the last ~17 bars (the pop "key up for the last chorus").
- At penalty 4, 80% of changes come after 70% of the song; 103 songs change once, 4 twice, none more.
- Share of chord time whose triad fits the scale: 88.1% with one key per song, 90.9% with the local key.
- Spot checks match the chord labels: song 010 G to Ab at bar 70, 012 Eb to E at bar 53, 025 A to C at bar 78.

### Melody repeats (`hw2.plan.repeat_label`, `data/tokens/{plan,fill}_rep`) · as of 2026-10-16

Melody bars of POP909 (54968), each against the bars 2, 4, 8 and 16 back:

- Exact repeat (80% or more of the notes match in start and pitch): 24.5%. Velocity is identical in 59% of exact repeats.
- Varied repeat: same rhythm with some pitches changed 8.4%, same pitches with timing changed 4.8%.
- Same rhythm, new pitches: 8.1%. New material: 50.9%.
- Tempo inside a song is near constant: beat-to-beat variation median 0 (p90 1.7%); last vs first quarter differs by more than 5% in 0.9% of songs.

Repeat label per bar (plan and fill prefix, right after `Bar`): `Rep_exact_L`, `Rep_var_L`, `Rep_rhythm_L` (L = 2, 4, 8, 16; strongest kind wins, then the shorter lag) or `Rep_none`.

### Theme return and repetition period, real songs (`return.csv`, `return_lags.csv`, `figures/return_lags.png`, `scripts/return_stats.py`) · as of 2026-10-16

Intro = bars before the first melody note, outro = bars after the last. Whole POP909 songs: intro median 6-7 bars (p25 4, p75 9, p90 11-12), outro median 4, body median 66 bars.

- Theme (first 8 body bars) vs later 8-bar windows, pitch-class cosine per bar, return = 0.85 or more: 80-84% of whole songs bring the theme back (about 2 times per song), mostly mid-body; in the first 32 bars, 39-54% of real songs already have a return.
- At theme returns (1437 cases): melody notes repeat exactly at median 72%; piano at median 35% while its pitch-class cosine stays at 0.90. The melody comes back, the accompaniment keeps the harmony and varies the notes.
- Lag profile (bar i vs bar i+L): peaks at lags 4 and 8 (also 2, 16, 24), dips at odd lags. Exact note match at lag 4 / 8 vs lag 3 / 6: melody 0.13 / 0.15 vs 0.04 / 0.05; piano 0.16 / 0.16 vs 0.08 / 0.09.
- Melody entry is late: median bar 8, only 30% by bar 5.
- Bars 1-8 are thinner than the rest (bar 1: 10-12 notes and 66-75% sounding, a pickup bar).

## Rendering and loudness

### Silent starts in rendered audio (2026-10-16)

Many POP909 songs start with a bar that holds only a pickup at its end, and both models learned it: the first note of a generated piece came 1.5-5.7 s after the start. `hw2.render.midi_to_wav` cuts the WAV to 0.2 s before the first sound above -50 dBFS; the MIDI keeps all 32 bars. 239 existing WAVs were trimmed (325.6 s in total).

### Stem loudness: melody vs piano (`stem_gap.csv`, `scripts/stem_gap.py`) · as of 2026-10-17

Melody stem minus piano stem, integrated LUFS (ffmpeg `ebur128`, gated), real songs seconds 8-72.

| Source | pieces | mean dB | median dB | sd |
|---|---|---|---|---|
| real POP909 | 299 | -2.73 | -2.80 | 2.42 |
| `oldplan_fill040` (real-velocity map) | 20 | -2.80 | -4.00 | 2.73 |
| `newplan_fill040` | 20 | -2.31 | -2.00 | 2.15 |
| `oldplan_fill060` | 20 | -3.32 | -3.40 | 2.13 |
| same three with the `min_gap` lift | 20 each | -1.98 / -1.65 / -2.30 | -3.00 / -1.90 / -1.95 | 2.11 / 1.62 / 1.66 |

The generated sets match the real melody-piano balance; the lift moves plan-fill about 0.8 dB above it.

### Sudden loud notes (`velocity_spikes.csv`, `scripts/velocity_spikes.py`) · as of 2026-10-18

Share of notes at least 20 above their bar's median velocity (all tracks merged), first 32 bars, after `one_keyboard`. Real: every tenth training song (90). "map" is the listening build's real-velocity map, "lift" adds the `min_gap` melody lift.

| | spike share | share of spikes on melody notes | spikes against the note's own track | melody-piano median gap (sd) |
|---|---|---|---|---|
| real openings | 0.077 | 0.59 | 0.037 | 14.0 (11.7) |
| `single_t1_p1` written / map | 0.064 / 0.046 | (one track) | 0.064 / 0.046 | |
| `pf_oldplan_fill040` written / map / map + lift | 0.060 / 0.085 / 0.095 | 0.79 / 0.67 / 0.71 | 0.021 / 0.041 / 0.041 | 10.0 / 9.0 / 14.5 |
| `fill_rep_60k/last` written / map / map + lift | 0.086 / 0.072 / 0.074 | 0.67 / 0.57 / 0.59 | 0.044 / 0.044 / 0.044 | 16.0 / 9.5 / 11.0 |

- Most spikes are melody notes above a bar median set by the more numerous piano notes, in real songs as in plan-fill; against their own track, plan-fill notes spike at the real level (0.041-0.044 vs 0.037).
- Submitted WAVs are quiet: `single_t1_p1` median −33.8 LUFS (−38.3 to −30.1), TASK B `t1_k0_p0.95` −34.2 to −31.2; every rated clip played at −18 LUFS.

## Single-stage model (`gpt2_ppb16`, TASK A and TASK B)

### Loss review (`loss_review.csv`, `scripts/loss_review.py`; 45 held-out songs, teacher forced) · as of 2026-10-17

| Checkpoint | All | Pitch | Pos | Dur | Vel |
|---|---|---|---|---|---|
| `step_6000` | 1.44 | 1.32 | 0.57 | 2.17 | 1.70 |
| `step_10000` | 1.47 | 1.16 | 0.59 | 2.43 | 1.71 |
| `loss_0.40` | 2.70 | 1.33 | 1.00 | 4.99 | 3.55 |

The held-out loss at `loss_0.40` is almost twice that of `step_6000`; most of the rise is in duration and velocity, pitch stays at 1.33. Listening picked `loss_0.40` anyway (T1 below).

### Silence and repetition (`silence.csv`, `return_lags.csv`) · as of 2026-10-16

- Silent share: `step_6000` 6.0% (1.40 empty bars per piece), `loss_0.40` 1.0% (0.05); real song starts 1.4-1.6% (0.04-0.06).
- No lag peaks: exact match decays with lag (`loss_0.40`: 0.09 at lag 1 to 0.07 at 24); real songs peak at 4 and 8.
- The best-matching pair of 8-bar windows in a 32-bar piece scores 0.64-0.70 for the main model and the first plan-fill sets (real crops 0.79-0.81): no near-repeated section.

### TASK A sets (`outputs/cand_a/single_*`, `cand_a_metrics.csv`) · as of 2026-10-17

| Set | H1 | H4 | GS | silent share | empty bars |
|---|---|---|---|---|---|
| real held-out | | | | 0.014 | 0.044 |
| `single_t1_p1` | 2.18 | 2.62 | 0.970 | 0.009 | 0 |
| `single_t1_p095` | 2.12 | 2.59 | 0.971 | 0.009 | 0 |
| `single_t09_p1` | 2.12 | 2.58 | 0.968 | 0.009 | 0 |

`bar_copy` (a bar copied from a training song) is 0 for every TASK A set; 8-note copies 1.2-2.8%. H1/H4/GS stay at the real level for these sets and the plan-fill sets below (H4 2.57-2.65, GS 0.958-0.971; real 32-bar crops 2.64 / 0.960), so they separate nothing.

### Cut: 40 bars generated, the first 8 dropped (2026-10-18; `outputs/cand_a/single_*_skip8`, `cand_a_*.csv`, `grade_pred_cand_a_v3_sets.csv`)

`generate_uncond.py --skip_bars 8` (the tempo is kept). With the same seeds, 19 of 20 pieces per config are the uncut piece's bars 9-40 note for note (bars 9-31 compared), so a cut and an uncut piece of the same seed share 24 bars. Predicted win vs a real opening (`v3`), uncut → cut: `t1 p1` 0.428 → 0.423, `t1 p0.95` 0.491 → 0.484, `t0.9 p1` 0.461 → 0.514 (per-piece sd 0.06-0.15). Not graded by ear yet.

## Plan-fill system (TASK A extra; `docs/plan_fill.md`)

### Plan adherence (`adherence.csv`, `cand_a_adherence.csv`, `scripts/adherence.py`) · as of 2026-10-18

The written notes of each piece, labelled again by the training rules and compared with the plan, bar by bar. Chord: the triad the piano holds longest in each half bar vs the planned chord; real held-out songs vs their labels give 0.85.

- The piano follows the planned chords at 0.80-0.85 (real 0.85) and the density bins within one step 94-98% of the time, for every fill model.
- Without help the fill model ignores repeat labels: bars the plan marks as a repeat get the asked kind 4.3% of the time, any repeat at the asked lag 13.5% (`plan_fill_rep/free`); melody bars with an exact repeat 1.6-3.7% vs 21.5% real.

### Teacher-forced repeat probe (`repeat_probe.csv`, `scripts/repeat_probe.py`) · as of 2026-10-16

Loss on the melody tokens of the 45 held-out songs (bars with at least 16 bars of context), by the bar's true repeat kind:

| Model, labels | Exact | Varied | Rhythm | None | Pitch tokens: exact / none |
|---|---|---|---|---|---|
| `fill_rep`, true labels | 0.585 | 1.145 | 1.183 | 1.425 | 0.223 / 1.461 |
| `fill_rep`, all `Rep_none` | 0.715 | 1.156 | 1.183 | 1.428 | 0.482 / 1.465 |
| `fill_rep`, labels removed | 0.641 | 1.146 | 1.188 | 1.433 | 0.361 / 1.486 |
| `fill_v1` (no labels) | 0.554 | 1.126 | 1.178 | 1.421 | 0.274 / 1.485 |

With the real earlier bar in context, the fill models predict exact repeats well (pitch loss 0.22-0.48 vs about 1.47 for new material); the label helps a little for exact repeats and not for varied or rhythm repeats. The same models write few repeats when sampling (previous section).

### Fill loss reads the plan (`loss_review.csv`) · as of 2026-10-17

`fill_rep/step_6000`: melody loss 1.16 with the true labels, 1.28 with the labels removed, 1.72 with each song's labels shuffled between its bars (piano 1.39 / 1.45 / 1.67). Held-out loss is lowest near `step_6000` and rises by `step_10000`, as for the main model.

### Repeat bias at sampling time (`outputs/plan_fill_rep_fix/*`; `adherence_fix.csv`, `return_stats_fix.csv`) · as of 2026-10-17

`--repeat_bias` β adds a logit bonus toward the source bar's notes on a melody bar labelled `Rep_<kind>_<L>`. The first version stacked up to 32 melody notes on one onset (fixed 2026-10-17); numbers below are from the fixed guide, same plans and seeds as `free`.

| Run | Max melody notes on one onset | Asked kind written at the lag | Any repeat at the asked lag | Melody bars exact / var / rhythm | MDen within 1 | Melody theme returns |
|---|---|---|---|---|---|---|
| real held-out | | 1 | 1 | 21.5% / 16.8% / 14.0% | 1 | |
| `free` (β 0) | | 0.05 | 0.14 | 1.6% / 6.6% / 14.6% | 0.98 | |
| `free_bias8` | 2 | 0.39 | 0.60 | 13.3% / 8.9% / 11.7% | 0.96 | 0.08 |
| `free_bias16` | 2 | 0.47 | 0.82 | 18.5% / 5.0% / 14.2% | 0.88 | 0.23 |
| `free_pbias4_bias8` | 5 | 0.51 | 0.76 | 14.5% / 8.0% / 16.3% | 0.98 | 0.20 |
| `form_IABAE_var_bias8` | 2 | 0.22 | 0.73 | 27.8% / 12.7% / 14.7% | 0.96 | 0.63 |

- Larger β writes more repeats and costs density adherence (MDen within one bin 0.96 at β 8, 0.88 at β 16).
- The plans ask for lag-4 repeats less often than real songs have them: repeat labels on melody bars, plans of `free` vs real first 32 bars: lag 2 13.5% vs 14.3%, lag 4 9.4% vs 15.7%, lag 8 10.8% vs 11.0%, lag 16 3.7% vs 2.9%, none 62.7% vs 56.2%.

### Plan side: `--plan_repeat_bias` · as of 2026-10-16

Share of labelled melody bars whose chords / melody bins equal the source bar's (real: held-out first 32 bars; plans: 20 each):

| | Real | Plan β 0 | Plan β 4 | Plan β 8 |
|---|---|---|---|---|
| exact | 0.68 / 0.80 | 0.20 / 0.14 | 0.70 / 0.75 | 0.88 / 0.98 |
| var | 0.36 / 0.27 | 0.19 / 0.14 | 0.37 / 0.50 | 0.79 / 0.87 |
| rhythm | 0.15 / 0.40 | 0.18 / 0.19 | 0.10 / 0.85 | 0.18 / 0.93 |

Plan β 4 brings exact and varied repeats to the real coupling. The plan's chord pairs repeat at lag 4 in 11-15% of bars vs 22% in POP909.

### Form rule (`--form`, `outputs/plan_fill_v2/form_*`) · as of 2026-10-17

`I4A8B8A8E4` with the exact melody copy (`--return_as copy`): the theme comes back in 100% of pieces by melody (by construction), 79% by all notes, 58% by piano (real first 32 bars: 39-54%); bar pairs 16 apart: melody exact match 0.62, piano cosine 0.79 with exact match 0.20 (real returns: piano cosine 0.90, exact 35%). At lags 4 and 8 the melody exact match stays at 0.06 (real 0.13 / 0.15): the rule gives one long return, not the short repeats inside sections. With `--return_as var --repeat_bias 8` the melody of A returns in 63% of pieces (table above).

### Silence (`silence.csv`, `cand_a_silence.csv`, `sweep_silence.csv`) · as of 2026-10-17

- **Superseded first plans (`plan_fill/melody_first`):** 10 of 11 empty bars were planned empty; the plan model wrote fully empty bars in 1.7% of bars vs 0.17% in POP909. `--min_piano_bin 1 --tempo_range 60,125` (sampling limits) removed them (`plan_fill_v2/gaps`: 0 empty bars).
- **Current sets:** silent share `pf_oldplan_fill040` 0.016, every `fill_rep_60k` checkpoint on the same plans 0.020-0.028; real openings 0.014. With real plans the fill model leaves 0.024 (`real_plan`), so the extra silence comes from the fill model.

### TASK A candidate sets (`outputs/cand_a/pf_*`, `cand_a_*.csv`) · as of 2026-10-17

20 pieces of 32 bars, `--form I4A8B8A8E4 --return_as var --repeat_bias 8`. `newplan` = `plan_rep_long/loss_0.60`, `oldplan` = `plan_rep/best`, fill = `fill_rep_long/loss_0.40` or `loss_0.60`.

| Set | H1 | H4 | GS | silent share | empty bars | chord match | asked repeat written |
|---|---|---|---|---|---|---|---|
| real held-out | | | | 0.014 | 0.044 | 0.852 | 1.00 |
| `newplan_fill040` | 2.04 | 2.57 | 0.962 | 0.033 | 0.100 | 0.803 | 0.220 |
| `oldplan_fill040` | 2.11 | 2.65 | 0.958 | 0.016 | 0 | 0.820 | 0.205 |
| `oldplan_fill060` | 2.09 | 2.64 | 0.962 | 0.030 | 0 | 0.805 | 0.138 |

Plan-fill pieces are shorter in tokens (2160-2520) than single-stage (3180-3430).

### Longer runs: predicted grade vs loss (`outputs/sweep/*`, `grade_pred_<run>*.csv`, `figures/grade_vs_loss_*.png`) · as of 2026-10-17

20 pieces per checkpoint. Fill sweep (`fill_rep_60k`) on the 20 plans of `pf_oldplan_fill040` with the same seeds; plan sweep (`plan_rep_24k`) with fill `fill_rep_long/loss_0.40`. Predicted win vs a real opening: `v1` 0.40-0.48 (fill), 0.40-0.49 (plan); `v3` 0.37-0.46 and 0.38-0.48; per-set standard error about 0.02, no trend with training loss. The listening test went the other way for the fill model (T4: the last checkpoint won 2 of 2).

### Fill model on real plans (`outputs/real_plan/fill_rep_long_loss_0.40`, `real_plan_*.csv`) · as of 2026-10-17

The fill checkpoint of `pf_oldplan_fill040` fills the real plans (first 32 bars) of the 45 held-out songs (`--real_plans`, `--return_as var --repeat_bias 8`).

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
| predicted win vs real (`v1` / `v3`) | 0.51 / 0.52 (`heldout_open32`, 12) | 0.45 / 0.43 | 0.43 / 0.41 |

- The chord match is the same with real and generated plans (0.81-0.82, real 0.85).
- With real plans the fill writes the asked repeat twice as often (0.44 vs 0.21); generated plans ask for a repeat in 46% of bars (real 30% in the first 32 bars).

### Harmony: melody against the chords and the piano (2026-10-18; `harmony.csv`, `scripts/harmony_stats.py`)

In the C-shifted song, first 32 bars, time-weighted per piece. Real: the 851 training songs. "Worst 10% line" = the value that 10% of real songs fall past.

| | real | `pf_oldplan_fill040` | `fill_rep_60k/last` (same plans) | real plan + `fill_rep_long/loss_0.40` |
|---|---|---|---|---|
| pieces | 851 | 20 | 20 | 45 |
| melody time in the local key | 0.995 | 0.998 | 0.998 | 0.997 |
| melody time on chord tones, pooled / per-piece median / worst 10% | 0.708 / 0.712 / 0.574 | 0.674 / 0.660 / 0.556 | 0.622 / 0.646 / 0.531 | 0.657 / 0.657 / 0.572 |
| pieces past the real worst 10% line (chord tones) | 0.10 | 0.20 | 0.35 | 0.11 |
| strong-beat melody onsets on chord tones, median | 0.765 | 0.709 | 0.686 | 0.733 |
| melody time against a sounding piano note: minor 2nd / major 7th / minor 9th | 0.018 / 0.016 / 0.011 | 0.025 / 0.019 / 0.017 | 0.020 / 0.029 / 0.018 | 0.021 / 0.022 / 0.014 |
| pieces past the real worst 10% line (minor 2nd + minor 9th) | 0.10 | 0.10 | 0.15 | 0.16 |
| piano time on chord tones, bars 1-8, median | 0.901 | 0.787 | 0.877 | 0.816 |
| melody step, semitones: within a chord / across a chord change | 2.19 / 2.72 | 2.13 / 2.81 | 2.08 / 2.39 | 2.19 / 2.62 |
| melody leaps (a minor 6th or more): within / across | 0.029 / 0.048 | 0.024 / 0.034 | 0.019 / 0.025 | 0.023 / 0.041 |

- No key or transposition fault: generated melodies are as in key as real ones (99.7-99.8%).
- The gap is melody time on chord tones: 20-35% of generated pieces fall past the line that 10% of real songs fall past. Harsh rubs (minor 2nd, minor 9th) are near the real level; the longer fill has more major-7th time (2.9% vs 1.6%).
- In bars 1-8 the longer fill's piano is at the real chord-tone level; the earlier fill was lower (0.79).
- Melody steps (consecutive onsets at most two beats apart, top note per onset): the current fill models move as smoothly as real melodies, within a chord and across a chord change, with fewer leaps. This is the baseline for the chord-relative model, whose risk is larger steps at chord changes.

### Chord-relative fill pitches (2026-10-18; `data/tokens/fill_rel`)

`Rel_k` (pitch measured from the chord root) = pitch − signed(root of the chord at the note's start), in the C-shifted song; chord tones become the same numbers under every chord. Decoding every `Rel_` note of the 896 songs with its bar's chord gives back all 1,508,522 notes (0 bar blocks differ). The default tokenizer output is unchanged (30 of 30 songs checked). Model `fill_rel_60k`: same settings and length as `fill_rep_60k` (60k steps, done 2026-10-18 22:23; final train loss 0.121 vs 0.115, val 3.79 vs 3.81).

### Chord-relative fill, trained (2026-10-20; `outputs/cand_a/pf_fillrel60k`; `harmony.csv`, `fill_rel_*.csv`, `grade_pred_fill_rel.csv`)

Same 20 plans and seeds as `fill_rep_60k/last` (`outputs/sweep/fill_rep_60k/last`), so piece i pairs by file name. Real = the 851 training songs (silence: song starts).

| | real | `fill_rep_60k/last` | `fill_rel_60k/last` |
|---|---|---|---|
| melody time on chord tones, pooled / per-piece median / worst 10% | 0.708 / 0.712 / 0.574 | 0.622 / 0.646 / 0.531 | 0.705 / 0.707 / 0.586 |
| pieces past the real worst 10% line (chord tones) | 0.10 | 0.35 | 0.10 |
| strong-beat melody onsets on chord tones, median | 0.765 | 0.686 | 0.800 |
| melody time in the local key | 0.995 | 0.998 | 0.986 |
| melody time against a sounding piano note: minor 2nd / major 7th / minor 9th | 0.018 / 0.016 / 0.011 | 0.020 / 0.029 / 0.018 | 0.034 / 0.020 / 0.016 |
| minor 2nd + minor 9th, per-piece median / worst 10% | 0.023 / 0.065 | 0.030 / 0.069 | 0.037 / 0.098 |
| pieces past the real worst 10% line (minor 2nd + minor 9th) | 0.10 | 0.15 | 0.25 |
| melody time on chord tones, bars 1-8, per-piece median | 0.721 | 0.702 | 0.625 |
| piano time on chord tones, median | 0.922 | 0.870 | 0.930 |
| melody step, semitones: within a chord / across a chord change | 2.19 / 2.72 | 2.08 / 2.39 | 2.06 / 2.80 |
| melody leaps (a minor 6th or more): within / across | 0.029 / 0.048 | 0.019 / 0.025 | 0.019 / 0.040 |
| plan chord match (`fill_rel_adherence.csv`; real held-out 0.852) | | 0.790 | 0.890 |
| silent share / `gaps_1beat` (`fill_rel_silence.csv`) | 0.016 / 0.48 | 0.026 / 0.95 | 0.019 / 0.70 |
| H1 / H4 / GS (`fill_rel_metrics.csv`) | | 2.09 / 2.61 / 0.973 | 2.09 / 2.66 / 0.968 |
| predicted win rate vs real, v3, mean (sd) | | 0.420 (0.114) | 0.402 (0.107) |

- Melody on chord tones is at the real level (pooled, median, worst 10%); the share of pieces past the real worst-10% line drops from 0.35 to 0.10, the real share.
- No jump at chord changes: the step across a chord change is 2.80 semitones (real 2.72), leaps 4.0% (real 4.8%). The early-checkpoint jumpiness (5.2 semitones, 17% leaps on 3 pieces) is gone.
- Cost: more minor-2nd time against the piano (0.034 vs 0.018 real), 25% of pieces past the real harsh line, and 1.4% of melody time out of key (real 0.5%). Bars 1-8 melody is less on chord tones than the whole piece. Split of the minor-2nd time (`m2_same_chord_time`, `m2_change_chord_time`): 0.029 between notes started under the same chord label vs 0.016 real, 0.005 across a chord change vs 0.002, so held notes over a chord change are the smaller part.
- Silence moves toward real (1.9% vs 2.6%; real 1.6%). Predicted grade: no difference (0.40 vs 0.42, sd 0.11).

### Melody-lead tests (2026-10-20; `outputs/cand_a/pf_fill60k_nochord`; `harmony.csv`, `harmony_piano_chords.csv`)

`harmony_stats.py --chords_from_piano` labels each half bar with the major or minor triad the piano sounds longest, for real and generated pieces alike; on 200 training songs it matches 86.8% of the major/minor labels. Same 20 plans and seeds as above; `pf_fill60k_nochord` = `fill_rep_60k/last` with `--drop chords` (no training).

| chords from the piano | real | `fill_rep_60k/last` | `fill_rel_60k/last` | `--drop chords` |
|---|---|---|---|---|
| melody time on chord tones, pooled / median | 0.715 / 0.719 | 0.611 / 0.622 | 0.678 / 0.662 | 0.525 / 0.527 |
| pieces past the real worst 10% line (chord tones) | 0.10 | 0.35 | 0.20 | 0.85 |
| minor 2nd + minor 9th, median; pieces past the real worst 10% line | 0.023; 0.10 | 0.030; 0.15 | 0.037; 0.25 | 0.058; 0.45 |

- Without chord labels the current fill model's melody and piano do not agree (85% of pieces past the line). The model saw chords in 85% of its training crops (`--drop_p 0.15`), so this tests the dropout, not a model trained without chords (`fill_nochord_60k`, in training).
- With chords read from the piano the chord-relative set is still closer to real than the current one (0.20 vs 0.35 of pieces past the line).

### Melody model plus BONUS piano (2026-10-20; `outputs/cand_a/pf_melody20k`; `harmony*.csv`, `melody_lead_*.csv`, `grade_pred_melody_lead.csv`)

`melody_20k/best` (step 5000, lowest val 0.90; val rose to 1.85 by step 13000) writes the whole melody without chord labels, then `abl_melody/step_7000` (BONUS TASK model, melody input only) writes the piano. Same 20 plans and seed as `fill_rep_60k/last`.

| | real | `fill_rep_60k/last` | melody + BONUS piano |
|---|---|---|---|
| chords from the piano: melody time on chord tones, median; pieces past the real worst 10% line | 0.719; 0.10 | 0.622; 0.35 | 0.480; 0.80 |
| chords from the piano: piano time on chord tones, median | 0.926 | 0.931 | 0.988 |
| minor 2nd / major 7th / minor 9th time | 0.018 / 0.016 / 0.011 | 0.020 / 0.029 / 0.018 | 0.034 / 0.032 / 0.022 |
| minor 2nd + minor 9th: median; pieces past the real worst 10% line | 0.023; 0.10 | 0.030; 0.15 | 0.035; 0.20 |
| plan chord match (`melody_lead_adherence.csv`) | 0.852 | 0.790 | 0.115 |
| melody returns of the plan's repeats (`melody_returns_ge1`) | 0.475 | 0.053 | 0.737 |
| silent share / `gaps_1beat` | 0.016 / 0.48 | 0.026 / 0.95 | 0.040 / 0.20 |
| predicted win rate vs real, v3, mean (sd) | | 0.420 (0.114) | 0.325 (0.146) |

- The piano does not follow the melody: with chords read from the piano, 80% of pieces have the melody off those chords past the real line. The piano itself is plain triads (0.988 on its own triads, real 0.926).
- The plan's chords are ignored (match 0.115), as expected: neither model gets them.
- The melody repeats the plan's sections (0.74 of pieces, real 0.48, current fill 0.05), so the melody model does follow the plan's repeat structure.
- Not a listening candidate. The minor-2nd penalty (`--rub_penalty`) was not tested: harsh rubs are at 20% of pieces, but the main gap is melody vs piano chords, which a rub penalty does not fix.

### Fill trained without chord labels (2026-10-20; `outputs/cand_a/pf_fillnochord60k`; same files)

`fill_nochord_60k/last` (60k steps, `--always_drop chords`; train loss 0.109, val 4.00; lowest val 1.38 at step 5500): melody and piano from one fill model, no chord labels in training or generation. `last`, as for `fill_rep_60k`. Same plans and seed.

| | real | `fill_rep_60k/last` | no-chord fill |
|---|---|---|---|
| chords from the piano: melody time on chord tones, median; pieces past the real worst 10% line | 0.719; 0.10 | 0.622; 0.35 | 0.547; 0.80 |
| chords from the piano: piano time on chord tones, median | 0.926 | 0.931 | 0.944 |
| minor 2nd / major 7th / minor 9th time | 0.018 / 0.016 / 0.011 | 0.020 / 0.029 / 0.018 | 0.034 / 0.025 / 0.017 |
| minor 2nd + minor 9th: median; pieces past the real worst 10% line | 0.023; 0.10 | 0.030; 0.15 | 0.042; 0.25 |
| melody time in the local key | 0.995 | 0.998 | 0.981 |
| plan chord match | 0.852 | 0.790 | 0.167 |
| silent share / `gaps_1beat` | 0.016 / 0.48 | 0.026 / 0.95 | 0.025 / 0.65 |
| predicted win rate vs real, v3, mean (sd) | | 0.420 (0.114) | 0.375 (0.130) |

- Better than `--drop chords` without training (median 0.547 vs 0.527, harsh-rub share 0.25 vs 0.45), but 80% of pieces are still past the real line: without labels the fill model does not keep melody and piano on one chord.
- All three melody-lead options fail the harmony check (`--drop chords`, melody + BONUS piano, no-chord fill). The chord labels are what keeps the two parts together; the current fill and the chord-relative fill stay the candidates.

### BONUS piano against the melody (2026-10-20; `bonus_harmony.csv`, `scripts/bonus_harmony.py`, `scripts/check_piano_hookup.py`)

Two checks on the melody + BONUS piano result above: is the BONUS model weak, or is the new hookup (`generate_plan_fill.py --piano_ckpt`) broken?

- **Hookup:** `check_piano_hookup.py` runs the same model, melody and seed through `generate_bonus.py`'s path and through the hookup, with the melody also moved to F: prompts 001-003, every bar identical (16 of 16, 251 / 251 / 475 piano notes). The hookup is not the cause.
- **Model, on the 10 BONUS prompts (real melodies, prompt beat grid), chords read from the piano:**

| piano under the real melody | pieces | melody on the piano's chords, median | below the real worst-10% line | piano on its own triads, median |
|---|---|---|---|---|
| real piano | 10 | 0.719 | 0.40 | 0.813 |
| `abl_melody` free (the hookup's setting) | 30 (3 seeds) | 0.507 | 0.67 | 0.828 |
| `abl_melody` grammar | 30 | 0.459 | 0.83 | 0.810 |
| `abl_chords` free (gets the chord labels) | 30 | 0.486 | 0.80 | 0.847 |

- The BONUS piano fits the melody much less than the real piano (0.46-0.51 vs 0.72), about the level of the melody-first set (0.48). It matches the low TA chord accuracy (`bonus_ablation.csv`: 0.16-0.23, ceiling 0.76).
- The line comes from 32-bar training openings; on these 16-bar prompts the real piano is below it in 4 of 10, so compare the columns, not the line.
- The variant given the chord labels is no better (0.486), because it does not follow its labels (`bonus_label_follow.csv`, `scripts/bonus_label_follow.py`; piano note time on the tones of the label sounding at each slot): real piano 0.971 (200 training songs) and 0.988 (the 10 prompts), `abl_chords` 0.527.
- **Cause, a training bug:** `CropDataset` (`src/hw2/data.py`) transposes `Pitch_` tokens by up to ±5 semitones but leaves `Chord_<root>` tokens alone, so in most crops the labels name the wrong root and the model learns to ignore them. Only `abl_chords` has chord tokens and `max_shift` 5; `abl_melody` and `abl_lookahead` (no chord tokens), the TASK A model (no chord tokens) and the plan-fill models (`max_shift` 0) are not affected. The `chords` row of the BONUS ablation (`bonus_ablation.csv`) is therefore invalid until retrained with the roots shifted too.

### Semitone clashes: which note is off the chord (2026-10-25; `clash_sides.csv`, `scripts/clash_sides.py`)

A clash = melody time against a sounding piano note a minor 2nd or minor 9th away (major 7ths not counted). With triad labels two chord tones are never a semitone apart, so every clash has the melody, the piano or both off the label. Shares of melody time, first 32 bars; real = 300 training songs.

| | real | `fill_rep_60k/last` | `fill_rel_60k/last` |
|---|---|---|---|
| clash time | 0.0266 | 0.0387 | 0.0505 |
| melody off the chord | 0.0209 | 0.0268 (1.3x) | 0.0335 (1.6x) |
| piano off the chord | 0.0046 | 0.0104 (2.3x) | 0.0138 (3.0x) |
| both off | 0.0011 | 0.0015 | 0.0033 |

- In real songs a clash is mostly a melody note passing over the chord; the excess in plan-fill is mostly on the piano side, which is written after the melody in each bar and could avoid it. This is why `--rub_penalty` acts on the piano only (`docs/plan_fill.md`).
- T5's two "bad harmony" chord-relative pieces split: piece 18, 70% of its clash time with the piano off the chord (23% both); piece 16, 65% with the melody off (`listening_issues.md` section 13).

### T6 sets: semitone penalty and cut (2026-10-25; `t6_clash_sides.csv`, `t6_harmony.csv`, `t6_silence.csv`, `t6_metrics.csv`)

30 new plans with an 8-bar warm-up (seed 200, `--skip_bars 8`, form `W8I4A8B8A8E4`), the same plans for every set below; the uncut sets use 20 other new plans (seed 100). Real: clash time 0.0266, piano side 0.0046 (300 training songs); melody chord-tone median 0.712; silent share 0.016.

| set | clash time | piano off / melody off | melody on chord tones, median; past the real line | piano on chord tones | silent |
|---|---|---|---|---|---|
| chord-relative, cut, penalty 0 | 0.0251 | 0.0056 / 0.0186 | 0.746; 0.10 | 0.950 | 0.026 |
| **chord-relative, cut, penalty 2 (T6 challenger)** | 0.0240 | 0.0022 / 0.0211 | 0.767; 0.03 | 0.958 | 0.028 |
| chord-relative, cut, penalty 4 | 0.0236 | 0.0022 / 0.0209 | 0.760; 0.07 | 0.957 | 0.026 |
| chord-relative, cut, penalty 8 | 0.0191 | 0.0026 / 0.0159 | 0.748; 0.03 | 0.954 | 0.027 |
| current fill, cut, penalty 2 | 0.0284 | 0.0051 / 0.0221 | 0.659; 0.07 | 0.906 | 0.035 |
| chord-relative, uncut (seed 100) | 0.0372 | 0.0063 / 0.0291 | 0.710; 0.20 | 0.932 | 0.025 |
| current fill, uncut (seed 100) | 0.0449 | 0.0083 / 0.0341 | 0.602; 0.25 | 0.854 | 0.018 |

- Penalty strength by the rule in `docs/status.md` (fixed before generating): 2 is the smallest value with piano-side clash time within 1.25x real, chord-tone time within 0.01 of penalty 0 and silence within 0.005. It cuts piano-side clash time from 0.0056 to 0.0022 (below real); melody-side clashes and chord tones do not move beyond the spread between strengths.
- The cut sets clash less and sit more on chord tones than the uncut ones, but on different plans, and plan-to-plan spread is large (the chord-relative fill: 0.0505 on the T5 plans, 0.0372 on the seed-100 plans), so the cut's share is not measured.
- The chord-relative cut sets have lower pitch-class entropy (H1 1.95 vs 2.08-2.14): more time on chord tones, less variety. No copied bars or n-grams in any set.

## Listening tests (`results/listening/*.csv`, `docs/listening_rubric.md`) · as of 2026-10-25

Blind A/B pairs on the phone page; `pref` from −2 (A much better) to +2 (B much better). The four pilot sessions (`candidates`, `ckpt`, `ckpt2`, `planfill`) used a flawed rendering and are a pilot only.

- **T1 (`t1_checkpoint`, 2026-10-17):** `loss_0.40` vs `step_10000` 3 wins, 1 loss, 1 same. Real openings won both anchor pairs.
- **T2 (`t2_openings`, 2026-10-17):** plan-fill (`pf_oldplan_fill040`) vs single-stage (`single_t1_p1`) 3 wins, 2 losses, 1 same, below the 5-of-6 rule; real openings won both anchor pairs.
- **T3 (`t3_continuations`, 2026-10-17):** TASK B, 3 prompts x 3 configs: `t1 p0.95` won 6 of 6 of its pairs, `t1 p1` beat `t0.9 p1` 2 of 3. Checked against the MIDI: the "weird key change in the last 8 bars" piece (prompt 3, `t0.9 p1`) moves to C# in bars 29-32 (fit of the prompt's A key 0.48 in that window; every other piece stays at 0.8 or more); the "sounds broken" piece (prompt 2, `t0.9 p1`) has 2.1% of onsets with two sounding notes a semitone apart vs 0.4% for the `t1 p0.95` piece.
- **T4 (`t4_openings`, 2026-10-18):** `fill_rep_60k/last` beat `fill_rep_long/loss_0.40` 2 of 2 on the same plans (once on each side); `plan_rep_24k/last` vs `plan_rep/best` 1-1; single-stage `t0.9 p1` beat `t1 p1` and `t1 p0.95`, `t1 p0.95` beat `t1 p1` (one pair each, all won by side B). Checked against the MIDI: the "too fast, too many notes" piece has tempo 120 vs 60 and 12.2 vs 9.7 notes per second; the "weird chord in the latter 8 bars" piece has the lowest key fit of its set in bars 25-32 (0.938, set mean 0.992).
- **T5 (`t5_sections`, rated 2026-10-23 to 25, 12 pairs, sides balanced; rules fixed before rating in `docs/status.md`):** single-stage `t1 p1` cut (`--skip_bars 8`) beat uncut 4 of 4 on the same seeds (sign test p 0.125); chord-relative plan-fill vs current 1-1; chord-relative plan-fill vs single-stage 4 wins, 1 loss, 1 same (2-1 against the cut, 2-0-1 against the uncut), below the 5-of-6 rule. Per section (`t5_sections_sections.csv`): the uncut single-stage lost bars 1-8 in 4 of 5 rated pairs and the cut won them in 4 of 5; chord-relative plan-fill lost bars 1-8 in 5 of 8 (flaws: wrong notes 5, empty 4, messy 4) and won bars 9-32 in 14 of 24. Checked against the MIDI (`listening_issues.md` section 13): every bar-level comment matches (high notes, pitch 94 vs median 70, in bars 12-13; little sound in bar 21; no melody in bar 11; the second half of bar 16 silent; more gaps in the current-fill piece 17 than the chord-relative one); "what happened in transpose?" is a move from D to G at bar 17 with the lowest key fit in bars 25-32 (0.922); the two "bad harmony" chord-relative pieces have the 4th and 5th most minor-2nd + minor-9th time of their set (0.083, 0.074; real worst-10% line 0.065).
- **T6 (`t6_select`, rated by 2026-10-26 18:20, 22 pairs, fresh pieces; challenger named and rules fixed before building, `docs/status.md`):** challenger (chord-relative fill + cut + semitone penalty 2) vs cut single-stage 6 wins, 4 losses (sign test p 0.75): below the 8-of-10 rule, so TASK A stays the cut single-stage. Report only: challenger vs penalty 0 on the same plans 1 win, 2 losses, 1 same (the penalty is not supported by listening; with one seed the two pieces stay identical until the first penalised pitch, and one pair was heard as "exactly the same"); challenger vs current fill with the same penalty and plans 3-1; real openings beat the cut 2-0 and the challenger 2-0. Dropped on checking the samples (user: drop wrong samples): pair 12, whose real clip is 192 beats (a 6-beat meter shown as 48 bars of 4/4, first note in bar 2) against 128 for the generated one; pair 09, the same piece on both sides (penalty 2 left 11 of the 30 pieces unchanged). After the drops: challenger vs cut still 6-4; penalty 1-2; real vs cut 1-0, real vs challenger 2-0. Generated pieces sit on the 16th-note grid (0.0-0.4% of onsets off it vs 2.8% real), so the "misaligned melody and piano" comments are not off-grid notes; not measured further. Many pieces end with a 33rd bar holding only the last notes' release. Comments: "long empty" bars and "very long notes in early bars" on plan-fill, "misaligned melody and piano", "long semitone clash", "random jump to a high note" on single-stage.
- **Order bias:** side B won 6 of 7 in T4 (sign test p 0.125); T5 A 6, B 5; T6 A 11, B 10; all six sessions A 28, B 33. With one pair per matchup, `listening.py pairs` put the first named system on side A, so T4's config pairs were confounded with the side.

## Listening-grade predictor (`docs/grade_model.md`; `grade_model_cv.csv`, `grade_model_check.csv`) · as of 2026-10-25

Logistic model on feature differences, fit on the decisive A/B pairs and the pilot sessions.

| Session scored before refitting | `v1` | `v2` | `v3` |
|---|---|---|---|
| T3 (9 pairs) | 5 | 3 | (built after T3) |
| T4 (7 pairs) | 5 | 4 | 6 |
| T5 (11 decisive pairs) | 6 | 6 | 5 |
| T6 (21 decisive pairs) | 12 | 11 | 14 |

- Leave-one-out on all 29 decisive pairs at L2 10 (T1-T4): `v1` 0.66, `v2` 0.62, `v3` 0.79; "more notes wins" 0.52. On T4, "side B wins" would also get 6 of 7.
- After T5 (40 decisive pairs, refit 2026-10-25), at L2 10: `v1` 0.55, `v2` 0.55, `v3` 0.70; "more notes wins" 0.60. T5 scored before the refit: 5-6 of 11, a coin flip, so the predictor ranks sets only roughly and is not evidence on its own.
- Strongest single features (`grade_single_features.csv`, 40 pairs, ties left out): the winner has the lower `semitone_clash_share` in 69% of pairs and the higher `half_density_change` in 65% (on 29 pairs: `half_density_change` 72%, `key_fit` and `opening_key_fit_min4` 62%).
- Tags (`grade_tags.csv`, AUC of the feature for clips with vs without the tag): `silent_gap` vs `longest_onset_gap_beats` 0.87; `wrong_notes` vs `key_fit` 0.28 (lower fit with the tag); `abrupt_start` vs `opening_high_share` 0.58.
- On T3, 5 of the 9 `v1` predictions sat between 0.46 and 0.54: the pairs share the 8-bar prompt, and whole-piece features dilute the 24-bar difference.

## Archived (do not reuse) · as of 2026-10-18

Removed from this file on 2026-10-18; the numbers are in commit `2d826a7` if the report needs to show the history.

- **Repeat-guide results from before the fix (2026-10-16/17):** the first `RepeatGuide` stacked up to 32 melody notes on one onset at β 8 or more, so every number from those runs (`free_bias4/8/16`, the first `free_pbias4_bias8` and `form_IABAE_var_bias8` tables, their cost and lag bullets) is unusable. The fixed re-run (`plan_fill_rep_fix`) replaces them.
- **Plan-fill candidate sets of 2026-10-16 night** (`task_a_plan_form_var_bias8`, `task_a_plan_form_var_pbias4_bias8`, `task_b_plan_bias8`, `task_b_plan_pbias4_bias8`): made with that guide. Replaced by `outputs/cand_a/pf_*` (2026-10-17).
- **Melody-vs-chord numbers quoted in chat on 2026-10-18 before `harmony_stats.py`:** they counted notes instead of time, pooled all pieces, and put every interval of 1 or 11 semitones mod 12 under "semitone rub", which lumps major 7ths and minor 9ths with minor seconds. `harmony.csv` (2026-10-18) replaces them. They were never saved to a file. The predictor's `semitone_clash_share` was not affected: it uses the real pitch distance.
- **Melody time on chord tones as a target:** the labels are triads, so "chord tone" means root, 3rd and 5th only, and 7ths, 9ths and 6ths count as off-chord. The shortfall against real songs is measured with the same labels on both sides, but it is not a share to push to 71% by force; a sampling bias toward triad tones (`--chord_bias`) was stopped for this reason before any set used it.
- **Harmony of the plan-fill cut set (`pf_fill60k_skip8`):** its `plans.json` lost a key change made inside the dropped warm-up in piece 04, so per-piece numbers for that set misread the piece. Not reported.
- **Velocity spikes "6% real":** measured on whole songs; real openings (the right reference for 32-bar pieces) are at 7.7%.
- **Predictor single features and tags fit on T1-T2 only (13 pairs)**, e.g. `half_pc_sim` winning 11 of 13: replaced by the refit on 29 pairs.
- **"Sparse intros, bars 1-4: 14.4 notes per bar vs 20.8"** (old `docs/status.md`): not traceable to a tracked CSV; the traceable figure is the bar-1 one under "Theme return and repetition period".

