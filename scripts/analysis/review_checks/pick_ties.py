import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, per_prompt

SCORE_DRIFT = 0.022
CANDIDATES = ["rescale12", "rescale12_seed1", "rescale12_seed2"]

for set_name in ("dev", "test"):
    print(f"== {set_name}: {set_name}__rescale12_best3_c8998 selection.csv")
    selection = pd.read_csv(RESULTS / f"{set_name}__rescale12_best3_c8998" / "selection.csv", dtype={"id": int}).set_index("id")
    scores = np.sort(selection[CANDIDATES].to_numpy(), axis=1)
    gap = scores[:, -1] - scores[:, -2]
    n = len(gap)
    print(f"{n} picks; top-2 score gap under {SCORE_DRIFT}: {int((gap < SCORE_DRIFT).sum())}, under 0.01: {int((gap < 0.01).sum())}, "
          f"minimum gap {gap.min():.4f}, median gap {np.median(gap):.4f}")

    print(f"-- {set_name}: per-item correlation of the two CLAP checkpoints (90.14 clap_cosine vs 89.98 clap8998_cosine), 100 items per run")
    correlations = []
    for prefix in ("rescale12", "cfg7"):
        for suffix in ("", "_seed1", "_seed2"):
            name = f"{set_name}__{prefix}{suffix}"
            frame = per_prompt(name)
            if "clap8998_cosine" not in frame:
                print(f"{name}: no clap8998_cosine column, skipped")
                continue
            r = stats.pearsonr(frame["clap_cosine"], frame["clap8998_cosine"])[0]
            correlations.append(r)
            print(f"{name}: Pearson {r:+.2f}, Spearman {stats.spearmanr(frame['clap_cosine'], frame['clap8998_cosine'])[0]:+.2f}")
    if correlations:
        print(f"range {min(correlations):+.2f} to {max(correlations):+.2f}")

    frames = {c: per_prompt(f"{set_name}__{c}") for c in CANDIDATES}
    if all("clap8998_cosine" in f for f in frames.values()):
        by_8998 = pd.DataFrame({c: frames[c]["clap8998_cosine"] for c in CANDIDATES}).idxmax(axis=1)
        by_9014 = pd.DataFrame({c: frames[c]["clap_cosine"] for c in CANDIDATES}).idxmax(axis=1)
        agree = float((by_8998 == by_9014).mean())
        print(f"-- {set_name}: the two checkpoints pick the same rescale12 seed on {agree:.0%} of items (chance 1/3 = 33%)")
        print(f"   picks by 89.98: {by_8998.value_counts().to_dict()}; by 90.14: {by_9014.value_counts().to_dict()}")
        check = pd.read_csv(RESULTS / f"{set_name}__rescale12_best3_c8998" / "selection.csv", dtype={"id": int}).set_index("id")["chosen_run"]
        print(f"   selection.csv matches the argmax of clap8998_cosine on {float((check == by_8998).mean()):.0%} of items")
    print()
