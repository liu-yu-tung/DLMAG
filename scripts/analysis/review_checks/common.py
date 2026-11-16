import json
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[3]
RESULTS = HW3 / "results"


def per_prompt(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / name / "per_prompt.csv", dtype={"id": int}).set_index("id").sort_index()
    mqa_path = RESULTS / name / "mqa_per_item.csv"
    if mqa_path.exists():
        mqa = pd.read_csv(mqa_path, dtype={"id": int}).set_index("id").sort_index()
        frame = frame.join(mqa)
    quick = RESULTS / name / "quick.jsonl"
    if quick.exists():
        rows = [json.loads(line) for line in quick.read_text().splitlines() if line.strip()]
        q = pd.DataFrame(rows).assign(id=lambda t: t["id"].astype(int)).set_index("id").sort_index()
        frame = frame.join(q[[c for c in ("peak", "clip_fraction", "rms", "tokens") if c in q]])
    return frame


def boot_ci(values: np.ndarray, resamples: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    idx = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(values: np.ndarray, flips: int = 100000, seed: int = 0) -> float:
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(flips, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float(((null >= abs(values.mean()) - 1e-15).sum() + 1) / (flips + 1))


def holm(pvalues: dict) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = running
    return out


def fmt(ci, digits=3):
    return f"{ci[0]:+.{digits}f} [{ci[1]:+.{digits}f}, {ci[2]:+.{digits}f}]"
