# HW2 entry map

Symbolic music generation on POP909. Deadline Mon 2026-10-26 23:59; the report (PPT-style PDF) is the whole grade. Start with `docs/status.md`.

## Read first

- **Current state, decisions, plan:** `docs/status.md`
- **Pipeline, commands, setup, code layout:** `README.md`
- **Evidence behind every number:** `results/findings.md` and the CSV it names in `results/`
- **Listening comments and open issues:** `docs/listening_issues.md`; rubric: `docs/listening_rubric.md`

## Where things are

- **Code:** `src/hw2/` (package), `scripts/` (one script per stage; `scripts/ta/` is the TA's code with two fixes, see its README), `Makefile` (`make help`)
- **Design notes:** `docs/plan_fill.md` (two-stage system), `docs/grade_model.md` (listening-grade predictor), `docs/bonus.md` (BONUS TASK), `docs/references.md` (tools, papers, dropped ideas), `docs/slides_notes.md` (course slides)
- **Tracked outputs:** `outputs/<set>/gen_info.json` and `metrics.csv` only (MIDI, WAV, `sequences.json` and logs are ignored); `outputs/cand_a/` holds the TASK A candidate sets, `outputs/sweep/<run>/<checkpoint>` the per-checkpoint grade sweeps
- **Not in git:** `checkpoints/`, `data/`, `runs/`, `outputs/**/*.mid|wav`, `outputs/listening/` (listening sessions with their hidden `key.json`), course zips and the assignment PDF

## Results index

| Files | Written by |
|---|---|
| `task_a_sampling.csv`, `task_a_checkpoints.csv`, `cand_a_metrics.csv` | `eval_generated.py` |
| `task_a_plan.csv`, `task_b_plan*.csv`, `task_b_fade.csv`, `plan_fill*.csv` | plan-fill and TASK B evaluations |
| `silence*.csv`, `cand_a_silence*.csv`, `sweep_silence*.csv` | `silence_stats.py` (per-piece, per-bar and closure tables are regenerated, not tracked) |
| `return*.csv`, `cand_a_return*.csv`, `figures/*return*` | `return_stats.py` |
| `adherence*.csv`, `cand_a_adherence.csv` | `adherence.py` |
| `loss_review.csv`, `repeat_probe.csv` | `loss_review.py`, `repeat_probe.py` |
| `stem_gap.csv` | `stem_gap.py` |
| `grade_model_*.json`, `grade_features.csv`, `grade_model_cv.csv`, `grade_model_check.csv`, `grade_single_features.csv`, `grade_tags.csv`, `grade_pred_*.csv`, `figures/grade_vs_loss_*.png` | `grade_model.py` (`docs/grade_model.md`) |
| `real_plan_*.csv` | fill model on real held-out plans (`generate_plan_fill.py --real_plans`, then the analysis scripts) |
| `bonus*.csv`, `bonus_ablation*` | BONUS TASK evaluation and ablation |
| `baselines.csv`, `track_stats.csv`, `key_changes.csv`, `progressions.csv`, `chord_from_notes.csv`, `token_budget.csv`, `figures/` | data checks and data stats |
| `listening/*.csv` | `listening.py reveal` and `import` (`candidates`, `ckpt`, `ckpt2`, `planfill`, `ratings_pilot` are the pilot sessions) |

## Rules for this folder

- Only output records are tracked; never commit MIDI, audio, checkpoints or `notes-from-*.md` files.
- New runs and outputs get new names; old checkpoints and folders stay so results stay traceable.
- Listening pages and notices carry no model or checkpoint names; sessions use neutral ids.
- `gen_info.json` and `args.json` record the code version; a final output must not be `+dirty`.
