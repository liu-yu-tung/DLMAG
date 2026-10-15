# HW2: symbolic music generation on POP909

A GPT-2 decoder (12 layers, 8 heads, 512 wide, 2048-token context, 39.2M parameters) trained from scratch on REMI-style tokens from the TA tokenizer (16 positions per beat). A second model of the same size, trained on bar-interleaved MELODY/PIANO tokens, does the bonus task.

## Setup

Python 3.12, Linux, NVIDIA GPU with CUDA (tested on an RTX 4060 Ti 16 GB, torch 2.14.1+cu130).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with uv: `uv venv --python 3.12 && uv pip install -r requirements.txt`.

`requirements.txt` holds only what inference needs and installs this folder as the `hw2` package.

WAV output (`--wav`) runs the `fluidsynth` command with the FluidR3 General MIDI soundfont:

- **Arch:** `pacman -S fluidsynth soundfont-fluid`
- **Debian/Ubuntu:** `apt install fluidsynth fluid-soundfont-gm`
- **Other soundfont:** set `SOUNDFONT=/path/to/file.sf2`

## Checkpoints and token files

Download from the cloud drive and place them as:

```
checkpoints/gpt2_ppb16/<checkpoint>/    TASK A and TASK B model
checkpoints/abl_melody/step_7000/       BONUS TASK model
checkpoints/plan_rep/best/              plan-fill: plan model
checkpoints/fill_rep/best/              plan-fill: fill model
data/tokens/ppb16/tok2idx.json          vocabulary
data/tokens/ppb16/prompts_basic_b.json  TASK B prompts, tokenized
data/tokens/bonus_ppb16/tok2idx.json    BONUS TASK vocabulary
data/tokens/fill_rep/tok2idx.json       plan-fill vocabulary
data/tokens/fill_rep/bins.json          plan-fill bin edges (labels TASK B prompts)
```

## Inference

Every script takes `--output_dir`; it is created if missing.

Task a, 20 pieces of 32 bars:

```
python scripts/generate_uncond.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir <dir> --temperature <t> --top_p <p> --wav
```

Task b, 24 bars after each 8-bar prompt, one folder per configuration (`temperature,top_k,top_p`):

```
python scripts/generate_cont.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir <dir> --config 1.0,0,0.95 --wav
```

To continue other prompts, tokenize their folders (same layout as POP909: `<name>/<name>.mid`, `beat_midi.txt`, `chord_midi.txt`) and pass `--prompts`:

```
python scripts/ta/tokenization.py --dataset_path <prompt_folder> --tok2idx_output /tmp/tok2idx.json --sequences_output <prompts.json>
```

Bonus, a PIANO track for each 16-bar MELODY prompt (folders with `melody.mid` and `beat_midi.txt`):

```
python scripts/generate_bonus.py --ckpt checkpoints/abl_melody/step_7000 --tok2idx data/tokens/bonus_ppb16/tok2idx.json --prompt_dir <bonus_prompt_folder> --output_dir <dir> --top_p 0.95 --wav
```

## Training and experiments

These need the full uv environment (`uv sync`), POP909 in `data/POP909/`, and MusDr cloned into `scripts/ta/` (see `scripts/ta/README.md`).

The `Makefile` has the same steps as named targets (`make help` lists them). Settings at its top (checkpoints, sampling, output folders, the GPU start temperature) can be overridden per run, e.g. `make task-a TEMP=1.2 TOP_P=0.95`. Training targets first wait until the GPU is at or below `GPU_START_C` (75 C) and log GPU temperature and clocks to `runs/<run>_gpu.csv`.

```
make setup
make tokens check stats
make train-main train-bonus train-plan-rep train-fill-rep
make task-a task-b bonus eval-task-a eval-task-b
make task-b-plan eval-task-b-plan PLAN_CKPT=checkpoints/plan_rep/best FILL_CKPT=checkpoints/fill_rep/best PLAN_RULES="--min_piano_bin 1 --cadence"
make analyze
```

Not in the Makefile: the sampling grid and checkpoint comparison (`scripts/eval_generated.py <folders> --summary_csv results/<name>.csv`), the prompt fade (`scripts/exp4_prompt_fade.py`), the bonus ablation (`scripts/bonus_ablation.py`, resumes finished runs) and its baselines (`scripts/bonus_baselines.py`).

## BONUS TASK tokens

`src/hw2/bonus.py`: a decoder-only model writes PIANO for a given MELODY on bar-interleaved tokens.

- **Sequence:** `<BOS>`, then per bar `Bar Mel <melody notes> Acc <piano notes>` (`Mel` = melody block, `Acc` = piano block), each note `Pos Pitch Vel Dur` on the TA grid and bins. No tempo tokens: timing comes from `beat_midi.txt`.
- **Inference:** the `Bar Mel <melody> Acc` part of each bar is forced; piano notes are sampled until the model writes `Bar`.
- **Ablation variants** (stored in `variant.json` next to the vocabulary): `lookahead` puts bar b+1's melody in bar b's melody block, so the piano sees the melody one bar ahead; `chords` starts each bar with `Chd` and (`Pos`, `Chord_<root>_<quality>`) pairs from `chord_midi.txt`. `chords` reads the labels that the TA chord accuracy is scored against, so it is reported as an ablation, not the headline.

## Plan-fill pipeline

A second `TASK A` system in two stages (`src/hw2/plan.py`, `src/hw2/plan_generate.py`). Rule labels per bar (local key, two chords, note count and mean duration bins per stream, optional melody repeat label) come from each song's MIDI, `beat_midi.txt` and `chord_midi.txt`. A plan model (6 layers, 11.3M parameters) writes the labels for the whole piece; a fill model (main-model size) writes the MELODY and PIANO notes bar by bar with that bar's labels in its prefix. For `TASK B` (`--prompt_dir`), each prompt folder is labelled by the same rules, the plan model continues the prompt's plan, and the fill model writes the new bars after the prompt notes; `--config` (repeatable) fills the same plans once per sampling setting. The `--tok2idx` default comes from `runs/`, which is not shipped, so pass it. Numbers and design notes are in `results/findings.md`.

```
uv run python scripts/tokenize_plan.py --repeat
make train-plan-rep train-fill-rep
uv run python scripts/generate_plan_fill.py --plan_ckpt checkpoints/plan_rep/best --fill_ckpt checkpoints/fill_rep/best --tok2idx data/tokens/fill_rep/tok2idx.json --output_dir <dir> --min_piano_bin 1 --tempo_range 60,125 --cadence --wav
uv run python scripts/generate_plan_fill.py --plan_ckpt checkpoints/plan_rep/best --fill_ckpt checkpoints/fill_rep/best --tok2idx data/tokens/fill_rep/tok2idx.json --output_dir <dir> --prompt_dir data/POP909/prompts/basic-b --melody_by_bar 9 --config 1.0,0,0.95 --min_piano_bin 1 --cadence --wav
```

Labels per bar (`src/hw2/plan.py`):

- **Bars and positions:** the TA tokenizer's grid (bar 0 may be a pickup); BRIDGE is merged into PIANO.
- **Key:** Viterbi over the 12 major scales (a minor key counts as its relative major) on each bar's duration-weighted pitch classes, switch penalty 4. The song is transposed so its first key is C; `KeyShift_k` marks a bar whose key differs from the bar before, k = key relative to the first key.
- **Chords:** the `chord_midi.txt` chord at the bar start and at the half-bar beat, reduced to maj/min/dim/aug/sus by `hw2.bonus.chord_token`. Plan tokens hold the degree above the local key; fill tokens hold the pitch class in the transposed song.
- **Density and duration:** per stream (melody, piano), onsets in the bar and their mean duration in beats as bins 0-7 (0 = empty, 1-7 = quantile bins of the nonzero training values, edges in `bins.json`): `PDen`, `PDur`, `MDen`, `MDur`.
- **Repeat label (`--repeat` token sets):** how the bar's melody relates to the bar 2, 4, 8 or 16 bars back: `Rep_exact_L`, `Rep_var_L` (pitches or timing changed), `Rep_rhythm_L` (same rhythm, new pitches) or `Rep_none`.

Token layout:

- **Plan (whole song):** `<BOS> Tempo Meter Key`, then per bar `[Sec] [KeyShift_k] Bar [Rep] Chord Chord PDen PDur MDen MDur` (`Sec` every 8 bars), then `<EOS>`.
- **Fill block (one bar):** the same bar prefix with fill chords, then `Mel` (melody block) and `Acc` (piano block), each followed by `Pos Pitch Vel Dur` notes, in either order (random per training crop, fixed at generation).

Sampling rules, all off by default (`--help`): `--min_piano_bin` (no bar without piano), `--tempo_range`, `--melody_by_bar`, `--cadence` (the plan ends on V-I), `--form I4A8B8A8E4` (sections; a repeated letter repeats that section) with `--return_as copy|exact|var|rhythm`. With the repeat-label models, `--repeat_bias` pulls a labelled melody bar toward the bar it repeats (the fill model copies well when teacher-forced but loses the copy when sampling), and `--plan_repeat_bias` makes the plan give such a bar its source bar's chords and melody bins, as POP909 repeats do. `--plans_from` fills the plans of an earlier run for paired comparisons; `--drop` leaves label groups out of the fill prefix.

## Analysis

Each script reads POP909 for the real reference plus any generated folders, prints a summary and writes `results/<name>.csv`. Per-piece and per-bar tables (`*_pieces.csv`, `*_bars.csv`, `silence_closure.csv`) are regenerated, not kept in git.

- **`scripts/data_stats.py`, `plot_data_stats.py`, `key_changes.py`:** dataset statistics and figures (`make stats`).
- **`scripts/silence_stats.py`:** silence, empty bars, melody gaps, contrast between 8-bar windows, start/end similarity.
- **`scripts/return_stats.py`:** intro and outro length, theme returns, repetition by lag (`results/figures/return_lags.png`).
- **`scripts/adherence.py`:** plan-fill outputs only; how often the written notes match the planned chords, bins and repeat labels.
- **`scripts/repeat_probe.py`:** teacher-forced melody loss of the fill models on held-out songs by true repeat kind, with repeat labels kept, set to `Rep_none` or removed (GPU, about 3 min).

## Layout

- **`src/hw2/remi.py`:** tokens to MIDI. Bar length is the piece meter (4, or 6 when most bars use beat 5 or later), stretched for bars whose notes run past it.
- **`src/hw2/generate.py`:** batched sampling with temperature, top-k, top-p; a sliding window restarts at a `Bar` token when a piece outgrows the context.
- **`src/hw2/bonus.py`:** bar-interleaved MELODY/PIANO tokens; generated notes are timed by the prompt's beat file.
- **`src/hw2/metrics.py`:** H1, H4, GS through the TA's `eval_basic.py` functions, with a fixed seed.
- **`src/hw2/novelty.py`:** share of generated bars and 32-token n-grams copied from the training songs.
- **`scripts/ta/`:** TA scripts, with two fixes listed in its README.
- **`src/hw2/plan.py`:** rule labels, plan tokens and fill blocks of the plan-fill pipeline.
- **`src/hw2/plan_generate.py`:** grammar-masked sampling of plans and fills, the sampling rules, MIDI output.
- **`src/hw2/data.py`:** training crops; the fill dataset drops each label group at random (15%), so the fill model also runs without it.
- **`scripts/listening.py`:** blind listening sessions; ratings go in `listening.csv`.
- **`scripts/listening_page.html`, `docs/listening_rubric.md`:** template and rubric of the phone rating page (`listening.py web`).
- **`scripts/health_check.sh`:** GPU, running jobs, stalled logs (`make health`).
- **`results/`:** metric tables behind the report; `results/findings.md` holds the findings and decisions.
