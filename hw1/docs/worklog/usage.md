# Worker usage

One row per finished worker. Numbers come from the harness completion notice, not estimates.

| Date | Worker | Model | Task | Tokens | Tool calls | Duration | Worth it? |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-28 | data-metrics | Sonnet | `data.py`, `metrics.py`, 13 tests (about 300 lines) | 61,584 | 13 | 139 s | No. Small and well specified; the main session could have written it for less, since it needs the same context anyway. |
| 2026-09-28 | clap (paused) | Sonnet | CLAP embeddings + zero-shot; code written, not yet run (no weights) | 88,750 | 50 | 708 s | Not yet. Most of the tokens went to waiting on a stalled download; it should have been paused sooner. |
| 2026-09-28 | mert (paused) | Opus | MERT-v2 loader, extraction CLI, smoke + sanity scripts; checked on 5.17 with random weights | 78,674 | 28 | 863 s | Partly. Its main value was clearing the risk that transformers 5.17 breaks MERT-v2; the rest is plain code. |

## Rule of thumb

Use a worker only when all of these hold:
- the task runs long (GPU extraction, model downloads, training) or needs a lot of exploration that would flood the main context;
- it is independent of other running work;
- the hand-off fits in one file pointer.

Otherwise do it in the main session.
