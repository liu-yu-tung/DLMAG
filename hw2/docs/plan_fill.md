# Plan-fill pipeline

A second `TASK A` system in two stages (`src/hw2/plan.py`, `src/hw2/plan_generate.py`). Rule labels per bar (local key, two chords, note count and mean duration bins per stream, optional melody repeat label) come from each song's MIDI, `beat_midi.txt` and `chord_midi.txt`. A plan model (6 layers, 11.3M parameters) writes the labels for the whole piece; a fill model (main-model size) writes the MELODY and PIANO notes bar by bar with that bar's labels in its prefix.

For `TASK B` (`--prompt_dir`), each prompt folder is labelled by the same rules, the plan model continues the prompt's plan, and the fill model writes the new bars after the prompt notes; `--config` (repeatable) fills the same plans once per sampling setting.

Commands are in the README (Inference, Training). Numbers and design notes are in `results/findings.md`.

## Labels per bar (`src/hw2/plan.py`)

- **Bars and positions:** the TA tokenizer's grid (bar 0 may be a pickup); BRIDGE is merged into PIANO.
- **Key:** Viterbi over the 12 major scales (a minor key counts as its relative major) on each bar's duration-weighted pitch classes, switch penalty 4 (`hw2.pop909.key_path`). The song is transposed so its first key is C; `KeyShift_k` marks a bar whose key differs from the bar before, k = key relative to the first key.
- **Chords:** the `chord_midi.txt` chord at the bar start and at the half-bar beat, reduced to maj/min/dim/aug/sus by `hw2.pop909.chord_token`. Plan tokens hold the degree above the local key; fill tokens hold the pitch class in the transposed song.
- **Density and duration:** per stream (melody, piano), onsets in the bar and their mean duration in beats as bins 0-7 (0 = empty, 1-7 = quantile bins of the nonzero training values, edges in `bins.json`): `PDen`, `PDur`, `MDen`, `MDur`.
- **Repeat label (`--repeat` token sets):** how the bar's melody relates to the bar 2, 4, 8 or 16 bars back: `Rep_exact_L`, `Rep_var_L` (pitches or timing changed), `Rep_rhythm_L` (same rhythm, new pitches) or `Rep_none`.

## Token layout

- **Plan (whole song):** `<BOS> Tempo Meter Key`, then per bar `[Sec] [KeyShift_k] Bar [Rep] Chord Chord PDen PDur MDen MDur` (`Sec` every 8 bars), then `<EOS>`.
- **Fill block (one bar):** the same bar prefix with fill chords, then `Mel` (melody block) and `Acc` (piano block), each followed by `Pos Pitch Vel Dur` notes, in either order (random per training crop, fixed at generation).

## Sampling rules

All off by default (`generate_plan_fill.py --help`):

- **Plan limits:** `--min_piano_bin` (no bar without piano), `--tempo_range`, `--melody_by_bar`, `--cadence` (the plan ends on V-I).
- **Form:** `--form I4A8B8A8E4` (sections; a repeated letter repeats that section) with `--return_as copy|exact|var|rhythm`.
- **Repeat bias (repeat-label models):** `--repeat_bias` pulls a labelled melody bar toward the bar it repeats (the fill model copies well when teacher-forced but loses the copy when sampling); `--plan_repeat_bias` makes the plan give such a bar its source bar's chords and melody bins, as POP909 repeats do.
- **Comparisons:** `--plans_from` fills the plans of an earlier run; `--drop` leaves label groups out of the fill prefix.
