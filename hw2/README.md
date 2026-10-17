# HW2: symbolic music generation on POP909

GPT-2 decoders trained from scratch on POP909 (the course's 896 training songs; 45 of them are held out for validation and as the real reference, so the models train on 851; see `results/findings.md`):

- **TASK A, single-stage:** one model writes REMI-style tokens from the TA tokenizer (16 positions per beat), all tracks merged.
- **TASK A, plan-fill:** a plan model writes per-bar labels (key, chords, note density, melody repeats), then a fill model writes the MELODY and PIANO notes bar by bar (`docs/plan_fill.md`).
- **TASK B:** continues the 3 prompts with the TASK A checkpoint.
- **BONUS TASK:** a model on bar-interleaved MELODY/PIANO tokens writes PIANO for a given MELODY (`docs/bonus.md`).

## Pipeline

Each stage reads what the stage before wrote. The `Makefile` runs most stages (`make help`).

| Stage | Scripts | Writes |
|---|---|---|
| Data | POP909 in `data/POP909/` (`training/`, `prompts/basic-b/`, `prompts/bonus/`) | |
| Tokens (`make tokens`) | `scripts/ta/tokenization.py`, `tokenize_bonus.py`, `tokenize_plan.py [--repeat]` | `data/tokens/<set>/` |
| Checks, data stats (`make check stats`) | `check_roundtrip.py`, `baselines.py`, `data_stats.py`, `plot_data_stats.py`, `key_changes.py` | `results/*.csv`, `results/figures/` |
| Train (`make train-*`) | `train.py` | `checkpoints/<run>/`, `runs/<run>/` (args, loss log, val songs) |
| Generate | `generate_uncond.py` (TASK A), `generate_cont.py` (TASK B), `generate_plan_fill.py` (TASK A, or TASK B with `--prompt_dir`), `generate_bonus.py` | `outputs/<dir>/`: `NN.mid`, `NN.wav`, `sequences.json`, `gen_info.json` (plan-fill also `plans.json`) |
| Evaluate (`make eval-*`) | `eval_generated.py` (H1, H4, GS, sampled-token NLL, copy rate), `prompt_fade.py` (TASK B), `scripts/ta/eval_bonus.py`, `bonus_ablation.py`, `bonus_baselines.py` | `results/task_*.csv`, `results/bonus*.csv`, `<bonus dir>/metrics.csv` |
| Analyze (`make analyze`) | `silence_stats.py`, `return_stats.py`, `adherence.py`, `repeat_probe.py`; not in the Makefile: `loss_review.py` (held-out loss per token type), `stem_gap.py` (melody vs piano stem loudness) | `results/*.csv` |
| Listen | `render_ref.py` (real anchors), `listening.py` (`prepare`/`pairs`, `web`, `import`, `reveal`) | `outputs/listening/`, `results/listening/` |
| Grade predictor (`make grade-*`) | `grade_model.py` (`fit`, `check`, `predict`; `docs/grade_model.md`) | `results/grade_*.csv`, `results/figures/grade_vs_loss_*.png`, `outputs/sweep/<run>/` |

Current state and plan: `docs/status.md` (file map for contributors: `CLAUDE.md`). Findings and decisions: `results/findings.md`. Listening comments and open issues: `docs/listening_issues.md`.

Models (`runs/<run>/args.json` has every setting):

| Run | Tokens | Use | Size |
|---|---|---|---|
| `gpt2_ppb16` | `ppb16` | TASK A single-stage, TASK B | 12 layers, 8 heads, 512 wide, 2048 context, 39.2M parameters |
| `abl_melody` | `bonus_ppb16` | BONUS TASK (`abl_lookahead`, `abl_chords`: ablations) | same as `gpt2_ppb16` |
| `plan_rep`, `fill_rep` (longer runs: `plan_rep_long`, `fill_rep_long`) | `plan_rep`, `fill_rep` | plan-fill with melody repeat labels | plan: 6 layers, 6 heads, 384 wide, 1024 context, 11.3M; fill: same as `gpt2_ppb16` |
| `plan_v1`, `fill_v1` | `plan`, `fill` | first plan-fill version, no repeat labels | as above |

## Setup

Python 3.12, Linux, NVIDIA GPU with CUDA (tested on an RTX 4060 Ti 16 GB, torch 2.14.1+cu130).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with uv: `uv venv --python 3.12 && uv pip install -r requirements.txt`.

`requirements.txt` holds only what inference needs and installs this folder as the `hw2` package (its `pyproject.toml` dependencies are the same inference set; analysis libraries are in the uv `analysis` group).

WAV output (`--wav`) runs the `fluidsynth` command with the FluidR3 General MIDI soundfont. Each piece is played as one keyboard (`hw2.render.one_keyboard`: a key struck again while it sounds ends the earlier note, and the same key struck twice at once sounds once); the MIDI files are not changed.

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

TASK A, 20 pieces of 32 bars, single-stage:

```
python scripts/generate_uncond.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir <dir> --temperature <t> --top_p <p> --wav
```

TASK A with plan-fill (the `--tok2idx` default comes from `runs/`, which is not shipped, so pass it):

```
python scripts/generate_plan_fill.py --plan_ckpt checkpoints/plan_rep/best --fill_ckpt checkpoints/fill_rep/best --tok2idx data/tokens/fill_rep/tok2idx.json --output_dir <dir> --min_piano_bin 1 --tempo_range 60,125 --cadence --wav
```

TASK B, 24 bars after each 8-bar prompt, one folder per configuration (`temperature,top_k,top_p`):

```
python scripts/generate_cont.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir <dir> --config 1.0,0,0.95 --wav
python scripts/generate_plan_fill.py --plan_ckpt checkpoints/plan_rep/best --fill_ckpt checkpoints/fill_rep/best --tok2idx data/tokens/fill_rep/tok2idx.json --output_dir <dir> --prompt_dir data/POP909/prompts/basic-b --melody_by_bar 9 --config 1.0,0,0.95 --min_piano_bin 1 --cadence --wav
```

To continue other prompts with `generate_cont.py`, tokenize their folders (same layout as POP909: `<name>/<name>.mid`, `beat_midi.txt`, `chord_midi.txt`) and pass `--prompts`:

```
python scripts/ta/tokenization.py --dataset_path <prompt_folder> --tok2idx_output /tmp/tok2idx.json --sequences_output <prompts.json>
```

BONUS TASK, a PIANO track for each 16-bar MELODY prompt (folders with `melody.mid` and `beat_midi.txt`):

```
python scripts/generate_bonus.py --ckpt checkpoints/abl_melody/step_7000 --tok2idx data/tokens/bonus_ppb16/tok2idx.json --prompt_dir <bonus_prompt_folder> --output_dir <dir> --top_p 0.95 --wav
```

## Training and experiments

These need the full uv environment (`uv sync`), POP909 in `data/POP909/`, and MusDr cloned into `scripts/ta/` (see `scripts/ta/README.md`).

Settings at the top of the `Makefile` (checkpoints, sampling, output folders, the GPU start temperature) can be overridden per run, e.g. `make task-a TEMP=1.2 TOP_P=0.95`. Training targets first wait until the GPU is at or below `GPU_START_C` (75 C) and log GPU temperature and clocks to `runs/<run>_gpu.csv`.

```
make setup
make tokens check stats
make train-main train-bonus train-plan-rep train-fill-rep
make task-a task-b bonus eval-task-a eval-task-b
make task-b-plan eval-task-b-plan PLAN_CKPT=checkpoints/plan_rep/best FILL_CKPT=checkpoints/fill_rep/best PLAN_RULES="--min_piano_bin 1 --cadence"
make analyze
```

Not in the Makefile: the TASK A sampling grid and checkpoint comparison (`scripts/eval_generated.py <folders> --summary_csv results/task_a_sampling.csv` or `task_a_checkpoints.csv`), the bonus ablation (`scripts/bonus_ablation.py`, resumes finished runs) and its baselines (`scripts/bonus_baselines.py`).

The analysis scripts read POP909 for the real reference plus any generated folders, print a summary and write `results/<name>.csv`. Per-piece and per-bar tables (`*_pieces.csv`, `*_bars.csv`, `silence_closure.csv`) are regenerated, not kept in git.

- **`silence_stats.py`:** silence, empty bars, melody gaps, contrast between 8-bar windows, start/end similarity.
- **`return_stats.py`:** intro and outro length, theme returns, repetition by lag (`results/figures/return_lags.png`).
- **`adherence.py`:** plan-fill outputs only; how often the written notes match the planned chords, bins and repeat labels.
- **`grade_model.py`:** listening-grade predictor fit on the graded sessions; `predict --plot` gives predicted grade vs checkpoint loss (`docs/grade_model.md`).
- **`repeat_probe.py`:** teacher-forced melody loss of the fill models on held-out songs by true repeat kind, with repeat labels kept, set to `Rep_none` or removed (GPU, about 3 min).

## Code layout

`src/hw2/` (installed as the `hw2` package):

- **`remi.py`:** token files, bars, tokens to MIDI. Bar length is the piece meter (4, or 6 when most bars use beat 5 or later), stretched for bars whose notes run past it.
- **`pop909.py`:** POP909 readers on the TA grid (beats, notes per bar, chord labels) and the local-key Viterbi, shared by the bonus and plan-fill tokenizers and the data analyses.
- **`data.py`:** training crops with transposition; the fill dataset drops each label group at random (15%), so the fill model also runs without it.
- **`generate.py`:** batched sampling with temperature, top-k, top-p; a sliding window restarts at a `Bar` token when a piece outgrows the context.
- **`bonus.py`:** BONUS TASK tokens and their ablation variants (`docs/bonus.md`).
- **`plan.py`, `plan_generate.py`:** plan-fill labels and tokens; grammar-masked sampling of plans and fills, the sampling rules, MIDI output (`docs/plan_fill.md`).
- **`metrics.py`:** H1, H4, GS through the TA's `eval_basic.py` functions, with a fixed seed.
- **`novelty.py`:** share of generated bars, 32-token n-grams and 8-note runs copied from the training songs.
- **`render.py`:** MIDI to WAV; for listening pages also velocity matching to POP909 and loudness.
- **`grade_features.py`:** piano-roll features of one MIDI (tracks merged) for the listening-grade predictor.

Other folders:

- **`scripts/`:** the scripts named in the pipeline table; `--help` lists each one's options.
- **`scripts/ta/`:** TA scripts, with two fixes listed in its README.
- **`scripts/listening_page.html`, `docs/listening_rubric.md`:** template and rubric of the phone rating page. `listening.py web --target site` (default) builds `outputs/listening/site/<page>/` (`--page`, default `audio`) with audio files in `clips/<session>/` next to the page and a JSON ratings API on the same origin; `--target claude` embeds the audio for a claude.ai page. `listening.py import` takes a JSON export or a folder of rating files; ratings and revealed results go in `results/listening/`.
- **`scripts/health_check.sh`:** GPU, running jobs, stalled logs (`make health`).
- **`results/`:** metric tables behind the report; `results/findings.md` holds the findings and decisions.
- **`docs/`:** `status.md` (current state, open decisions, plan), plan-fill and bonus token details, listening rubric and issues.
