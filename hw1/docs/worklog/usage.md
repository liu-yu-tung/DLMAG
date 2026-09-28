# Worker usage

One row per finished worker. Numbers come from the harness completion notice, not estimates.

| Date | Worker | Model | Task | Tokens | Tool calls | Duration | Worth it? |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-28 | data-metrics | Sonnet | `data.py`, `metrics.py`, 13 tests (about 300 lines) | 61,584 | 13 | 139 s | No. Small and well specified; the main session could have written it for less, since it needs the same context anyway. |

## Rule of thumb

Use a worker only when all of these hold:
- the task runs long (GPU extraction, model downloads, training) or needs a lot of exploration that would flood the main context;
- it is independent of other running work;
- the hand-off fits in one file pointer.

Otherwise do it in the main session.
