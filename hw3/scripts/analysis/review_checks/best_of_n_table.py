import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import HW3, RESULTS, boot_ci, per_prompt

OUT = HW3 / "results" / "analysis" / "best_of_n_vs_seed_mean.csv"
SEEDS = ("", "_seed1", "_seed2")
RANKERS = {"same-checkpoint (90.14)": "clap_cosine", "cross-checkpoint (89.98)": "clap8998_cosine"}
METRICS = {"dclap9014": "clap_cosine", "dclap8998": "clap8998_cosine", "dkl_passt": "kl_passt_softmax", "dkl_panns": "kl_panns_softmax"}
rows = []
for set_name, setting in (("dev", "cfg7"), ("dev", "rescale12"), ("test", "rescale12"), ("test", "cfg7")):
    frames = [per_prompt(f"{set_name}__{setting}{s}") for s in SEEDS]
    if not all("clap8998_cosine" in f for f in frames):
        continue
    stack = {m: np.stack([f[c].to_numpy() for f in frames], 1) for m, c in METRICS.items()}
    items = np.arange(len(frames[0]))
    for ranker, column in RANKERS.items():
        pick = np.stack([f[column].to_numpy() for f in frames], 1).argmax(1)
        submitted = RESULTS / f"{set_name}__{setting}_best3{'_c8998' if column == 'clap8998_cosine' else ''}" / "per_prompt.csv"
        if submitted.exists():
            recorded = pd.read_csv(submitted).set_index("id").sort_index()
            assert np.allclose(recorded["clap_cosine"].to_numpy(), stack["dclap9014"][items, pick], atol=1e-6), submitted
        row = {"set": set_name, "setting": setting, "ranker": ranker, "baseline": "mean of seeds 0-2 per prompt (random pick)",
               "checked_against_record": submitted.exists(), "picks_seed0": int((pick == 0).sum()), "picks_seed1": int((pick == 1).sum()),
               "picks_seed2": int((pick == 2).sum())}
        for name, values in stack.items():
            mean, low, high = boot_ci(values[items, pick] - values.mean(1))
            row.update({name: round(mean, 4), f"{name}_low": round(low, 4), f"{name}_high": round(high, 4)})
        rows.append(row)
table = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT, index=False)
print(table.to_string(index=False))
print(f"-> {OUT}")
