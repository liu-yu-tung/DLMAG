import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import HW3, boot_ci

OUT = HW3 / "results" / "analysis" / "clap_margin.csv"
NAMES = [str(i) for i in range(100)]
TEXT = {"dev": HW3 / "outputs/ref/dev_real_cache/clap_laion_text.pth", "test": HW3 / "outputs/ref/test_retrieved_cache/clap_laion_text.pth"}


def unit(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def text(set_name):
    cache = torch.load(TEXT[set_name], weights_only=True, map_location="cpu")
    return unit(np.stack([cache[n].numpy() for n in NAMES]).astype(float))


def parts(set_name, run):
    cache = torch.load(HW3 / "outputs/pred_cache" / f"{set_name}__{run}" / "clap_laion_audio.pth", weights_only=True, map_location="cpu")
    similarity = unit(np.stack([cache[n].numpy() for n in NAMES]).astype(float)) @ text(set_name).T
    own = np.diag(similarity)
    return own, (similarity.sum(1) - own) / 99


COMPARISONS = {
    "dev": [("rescale12", "cfg7"), ("rescale12_seed1", "cfg7_seed1"), ("rescale12_seed2", "cfg7_seed2"), ("pooled:rescale12", "cfg7"),
            ("cfg12", "cfg7"), ("cfg3", "cfg7"), ("apg12", "cfg7"), ("tags", "cfg7"), ("interval_late", "cfg7"),
            ("rescale12_best3_c8998", "rescale12"), ("rescale12_best3", "rescale12"), ("cfg7_best3_c8998", "cfg7"), ("cfg7_best3", "cfg7"),
            ("rescale12_best3_c8998", "cfg7"), ("real", "cfg7")],
    "test": [("rescale12", "cfg7"), ("rescale12_seed1", "cfg7_seed1"), ("rescale12_seed2", "cfg7_seed2"), ("pooled:rescale12", "cfg7"),
             ("rescale12_best3_c8998", "rescale12"), ("rescale12_best3_c8998", "cfg7")],
}
SEEDS = ("", "_seed1", "_seed2")
rows = []
for set_name, pairs in COMPARISONS.items():
    for run, baseline in pairs:
        if run.startswith("pooled:"):
            setting = run.split(":", 1)[1]
            got = [(parts(set_name, setting + s), parts(set_name, baseline + s)) for s in SEEDS]
            own = np.mean([a[0] - b[0] for a, b in got], 0)
            other = np.mean([a[1] - b[1] for a, b in got], 0)
            label, base_label = f"{setting} (3 seed pairs)", f"{baseline} (same seeds)"
        else:
            a, b = parts(set_name, run), parts(set_name, baseline)
            own, other = a[0] - b[0], a[1] - b[1]
            label, base_label = run, baseline
        row = {"set": set_name, "run": label, "baseline": base_label}
        for name, values in (("own", own), ("other", other), ("margin", own - other)):
            mean, low, high = boot_ci(values)
            row.update({f"d{name}": round(mean, 4), f"d{name}_low": round(low, 4), f"d{name}_high": round(high, 4)})
        rows.append(row)
table = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT, index=False)
print(table.to_string(index=False))
print(f"-> {OUT}")
