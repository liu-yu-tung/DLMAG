import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS

WINDOWS = {"steps 0-20": slice(0, 21), "steps 71-99": slice(71, 100)}


def load(run: str) -> dict[str, np.ndarray]:
    records = [json.loads(line) for line in (RESULTS / run / "trajectory.jsonl").read_text().splitlines() if line.strip()]
    keys = ("difference_norm", "difference_norm_valid", "parallel_fraction", "cond_norm")
    return {key: np.array([r[key] for r in records], dtype=float) for key in keys}


for run in ("dev__cfg7", "dev__rescale12"):
    t = load(run)
    ratio = t["difference_norm"] / t["cond_norm"]
    valid_share = t["difference_norm_valid"] / t["difference_norm"]
    print(f"== {run}: {len(ratio)} clips x {ratio.shape[1]} steps (NaN = guidance off at that step: {int(np.isnan(ratio).sum())} cells)")
    for name, window in WINDOWS.items():
        print(f"{name}: norm ratio ||D_c - D_u|| / ||D_c|| median over all clip-steps {np.nanmedian(ratio[:, window]):.3f}, "
              f"median of per-step medians {np.nanmedian(np.nanmedian(ratio[:, window], axis=0)):.3f}; "
              f"parallel fraction median {np.nanmedian(t['parallel_fraction'][:, window]):.3f}")
    print(f"valid share (norm of the difference on the 216 valid frames / norm over all frames): min over clips and steps {np.nanmin(valid_share):.4f}, "
          f"lowest per-step median {np.nanmin(np.nanmedian(valid_share, axis=0)):.4f}, highest {np.nanmax(valid_share):.4f}")

t = load("dev__cfg7")
ratio = t["difference_norm"] / t["cond_norm"]
print("\n== dev__cfg7 late-window sensitivity of the norm ratio: first step of the window to step 99")
for start in (65, 70, 71, 75):
    print(f"steps {start}-99: median {np.median(ratio[:, start:]):.4f}, mean {ratio[:, start:].mean():.4f}")
