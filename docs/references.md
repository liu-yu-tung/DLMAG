# Tools, packages and papers

What HW2 uses, what it dropped, and what the course slides name that we do not use. Slide pages refer to `docs/slides_notes.md`.

## Data and course files

- **POP909:** pop songs, each a MIDI with MELODY, BRIDGE and PIANO tracks plus beat and chord label files. The course folder `data/POP909/training` has 896 songs; 45 of them are held out, so the models train on 851.
- **TA `tokenization.py`:** turns POP909 into REMI-style tokens on a 16-per-beat grid. Our single-stage tokens come from it unchanged.
- **TA `eval_basic.py`:** H1, H4 (pitch-class entropy over 1 and 4 bars) and GS (rhythm similarity between bars), through MusDr.
- **TA `eval_bonus.py`:** BONUS TASK scores, including chord accuracy against `chord_midi.txt`.
- **MusDr:** the metric code from the Jazz Transformer paper, cloned into `scripts/ta/MusDr`. It also has CPI and SI, which we do not run yet.

## Packages

- **PyTorch:** training and inference, with bf16 autocast, `torch.compile` and fused AdamW.
- **Hugging Face `transformers`:** `GPT2LMHeadModel` for every model; `safetensors` stores the weights.
- **miditoolkit:** tokens to MIDI (`hw2.remi`), in ticks.
- **pretty_midi:** reads POP909 and generated MIDI in seconds; used by rendering and analysis.
- **numpy, scipy:** arrays; scipy reads WAVs (`scipy.io.wavfile`).
- **pandas, matplotlib, tqdm:** analysis tables, figures, progress bars (uv `analysis` group, not in `requirements.txt`).
- **uv:** Python version and venv.

## External programs

- **FluidSynth + FluidR3 GM soundfont:** MIDI to WAV, called as the `fluidsynth` command.
- **ffmpeg (`ebur128` filter):** loudness in LUFS for listening builds.
- **Listening page:** our own HTML template (`scripts/listening_page.html`), no outside scripts; hosted on the self-hosted site.

## Papers our methods come from

- **REMI / Pop Music Transformer (Huang and Yang, ACM MM 2020):** the token format (bar, position, duration, velocity, chord). Lecture 05 p46-49, p62-63.
- **GPT-2 (Radford et al., 2019):** the decoder-only Transformer we train from scratch. Lecture 06b.
- **Nucleus sampling (Holtzman et al., 2019):** top-p with temperature. Lecture 05 p74-79.
- **Jazz Transformer / MusDr (Wu and Yang, ISMIR 2020):** H, GS metrics; the loss-level vs Real table format. Lecture 05 p85-91.
- **Compose & Embellish (Wu and Yang, ICASSP 2023):** melody first, then piano; bar-interleaved melody and piano; the five-aspect listening rubric; sampling settings matched to validation perplexity. Our plan-fill and BONUS TASK token layout follow this idea. Lecture 05 p78-83, Lecture 06 p48-54.
- **Whole-song hierarchical generation (Wang et al., ICLR 2024):** form strings such as `(i4)(A8B8)(o2)`; same idea as our `--form I4A8B8A8E4`. Lecture 06 p71-75.
- **MuseMorphose:** per-bar attribute bins, like our density bins, and the warning that a condition can be ignored. Lecture 06 p41-44.
- **Theme Transformer:** prompt influence fades over time; theme regions bring material back. Background for TASK B prompt fade and our repeat guide. Lecture 06 p34-40.

## Ours (no paper)

- **Plan-fill:** plan model writes per-bar labels, fill model writes notes bar by bar (`docs/plan_fill.md`).
- **Repeat guide (`--repeat_bias`, `--plan_repeat_bias`):** a logit bonus toward the notes of the bar a repeat label points to.
- **Local key path:** Viterbi over 12 major scales per bar (`hw2.pop909.key_path`).
- **One-keyboard rendering:** a key struck again while sounding ends the earlier note; same key twice at once sounds once.
- **Silence, return and adherence stats:** `silence_stats.py`, `return_stats.py`, `adherence.py`.

## Dropped or replaced

- **midi2audio:** dropped 2026-10-16; the `fluidsynth` command is called directly.
- **`plan_v1`, `fill_v1`:** replaced by `plan_rep`, `fill_rep` (adds melody repeat labels). Kept for comparison.
- **Piano-first fill order:** melody-first is the default; piano-first was only tested with the v1 models.
- **Exact melody returns (`--return_as copy`):** reference only; the user does not want note-for-note returns.
- **BONUS chords variant (`abl_chords`):** an ablation only, since it reads the labels chord accuracy is scored against.
- **`step_6000` as the main checkpoint:** replaced by `loss_0.40` (T1 listening, 2026-10-17); `outputs/task_b` (old, `step_6000`) is replaced by `outputs/task_b_final` (`loss_0.40`).
- **`--chord_bias` (2026-10-18):** a chord-tone bonus on strong beats; removed before use, since the triad labels would also push out 7ths and 9ths. Replaced by chord-relative fill pitches (`Rel_k`, `docs/plan_fill.md`).
- **Mid-song real crops as anchors:** real song openings added (`outputs/ref/heldout_open32`), since intros are sparse.
- **Category averages in listening:** replaced by A/B preference against same-session real anchors (user decision).
- **`scripts/trim_wavs.py`:** one-off, removed after use.

## Named in the slides, not used

- **MidiTok, MusPy, pypianoroll, mido, music21:** the TA tokenizer and pretty_midi / miditoolkit cover what we need. MusPy's empty beat and empty measure rates match our silence stats.
- **Compound Word (CP) tokens, per-type sampling temperature:** not used; per-type temperature is a cheap test (Lecture 06 p21).
- **Transformer-XL, x-transformers, encoder/decoder conditioning:** not used.
- **CPI and SI (MusDr):** not run yet; candidates for the structure gap where H1/H4/GS separate nothing.
- **html-midi-player:** not used; our page draws its own piano roll.
