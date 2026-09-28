# Conventions

Rules for anyone (human or agent) working in `hw1/`.

## Layout

```
hw1/
  src/hw1/            importable package: reusable code only
    data.py           manifest + audio loading, label lists
    metrics.py        top-k, S score, confusion matrix, prediction JSON
    features/         one module per encoder (mert.py, clap.py, ...)
  scripts/            thin CLIs that import from src/hw1 (extract_*, train_*, predict.py)
  tests/              pytest, fast, no GPU needed
  docs/
    PLAN.md           the approved plan
    DECISIONS.md      decision log (append-only)
    worklog/<topic>.md  one report per work item
  results/            small JSON metrics, confusion-matrix PNGs (tracked)
  features/           cached embeddings (.npz, ignored)
  checkpoints/        trained probes/models (ignored; uploaded to the cloud drive)
  data/               datasets (ignored)
```

## Code

- Python 3.12, type hints, small functions, no notebooks in the main path.
- Label strings and order come only from `hw1.data.LABELS`.
- Train on `train` only. `validation` is for model selection. `test` is only for the final JSON.
- Deps change only through `uv add` / `uv remove`, done by the main session.
- Run everything with `uv run` from `hw1/`.

## Tests

- `uv run pytest` must pass before a commit.
- New reusable functions get a test in `tests/`.

## Agent hand-off

- Each work item writes `docs/worklog/<topic>.md` with the sections **Changes**, **Decisions**, **Tests**, **Open issues**.
- Numbers go to `results/<topic>.json`.
- The reply to the main session is the report path plus a one-line status. Don't paste logs or file contents.
- Workers don't commit. The main session reviews, runs the tests, and commits.
- Use at most 2 parallel workers, only for heavy, independent items. Small items stay in the main session.

## Git

- One logical change per commit, prefixed with its scope: `hw1: add MERT-v2 feature extraction`.
- Subject in the imperative, 72 characters or fewer. The body says what changed and why, and points to the worklog or decision entry.
- Push after each verified commit.
