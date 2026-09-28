# data.py / metrics.py

## Changes

- `src/hw1/data.py`
  - `DATA_ROOT` — `parents[2] / "data"`
  - `LABELS: dict[str, list[str]]` — `"A"`: 1960s..2010s, `"B"`: US/UK/Brazil/Spain/Germany/Italy
  - `dataset_dir(key: str, root: Path | None = None) -> Path`
  - `load_manifest(key_or_dir: str | Path) -> pd.DataFrame` — adds `path` (absolute) and `y` (int, -1 for test/unlabeled)
  - `load_split(key_or_dir: str | Path, split: str) -> pd.DataFrame` — filtered, index reset
  - `load_audio(path: str | Path) -> np.ndarray` — float32 1-D via soundfile, asserts 24000 Hz
- `src/hw1/metrics.py`
  - `topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float`
  - `score_s(probs: np.ndarray, y: np.ndarray) -> float` — top1 + 0.5*top3
  - `confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> np.ndarray`
  - `plot_confusion(cm, labels: list[str], path: str | Path, normalize: bool = False, title: str = "") -> None` — Agg backend, annotated PNG, title states "counts" or "row-normalized"
  - `evaluate(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict` — `top1`, `top3`, `S`, `cm` (nested list)
  - `top3_labels(probs: np.ndarray, labels: list[str]) -> list[list[str]]`
  - `write_predictions(path, preds_by_dataset: dict[str, dict[str, list[str]]]) -> None` — JSON, indent=2
  - `validate_predictions(obj: dict, manifests: dict[str, pd.DataFrame]) -> list[str]` — returns list of error strings (empty = valid); checks every test `sample_id` present exactly once, 3 distinct valid labels
- `tests/test_data_metrics.py` — 13 pytest cases covering both modules
- `src/hw1/__init__.py` — untouched

## Decisions

- `load_manifest` accepts `"A"`/`"B"` directly, or any dataset folder path; when given a path it infers the key from the folder name suffix (`..._A`/`..._B`) and falls back to the `sample_id` prefix (`A_`/`B_`) from the first manifest row, per the spec's note that the TA may point at a different folder.
- `label` column is coerced to string and NaN filled to `""` (pandas reads empty CSV field as NaN by default), so test rows compare cleanly to `""` and get `y = -1`.
- `validate_predictions` takes `manifests` as `{"dataset_A": df, "dataset_B": df}` (matching the `preds_by_dataset` key naming) and derives the dataset key by splitting on `"_"`; it returns a list of human-readable error strings rather than raising, so a caller can print/log all problems in one pass instead of stopping at the first one.
- `plot_confusion` row-normalizes by dividing by row sums (rows with sum 0 left as 0 to avoid div-by-zero) and always appends `(counts)`/`(row-normalized)` to the title.
- Test file placed at `tests/test_data_metrics.py` (not under `src/`), using `tmp_path` fixtures instead of writing to `/tmp` directly, so pytest handles cleanup.

## Tests

Command: `cd /home/liuyu/Developments/DLMAG/hw1 && uv run pytest tests/ -q`

Result: `13 passed in 1.44s`. pytest was already present as a dev dependency by the time this ran (added by the main session), so no fallback verification was needed.

Manual checks folded into the test run (via test assertions, not a throwaway script):
- Dataset A manifest: 1291 rows incl. header (1290 samples); Dataset B: 1003 rows incl. header (1002 samples). Split/label counts implicitly checked (test rows have `label == ""` and `y == -1`; train/validation rows have `label` in the expected 6-class set).
- `load_audio` on one file from each dataset returns shape `(720000,)`, dtype `float32`.
- `evaluate()` on random probs for a batch produces the 4 expected keys and a 6x6 `cm`.
- `plot_confusion` writes a non-empty PNG to a pytest tmp dir (auto-cleaned).
- `write_predictions` / `validate_predictions` round-trip: valid full test-set predictions produce zero errors; a broken entry (duplicate label instead of 3 distinct) is caught.

## Open issues

- None. Did not touch `pyproject.toml`, `uv.lock`, or make any commits, per instructions.
