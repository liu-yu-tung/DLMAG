import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import per_prompt

SEEDS = ["", "_seed1", "_seed2"]
rescale = {s: per_prompt(f"dev__rescale12{s}") for s in SEEDS}
cfg = {s: per_prompt(f"dev__cfg7{s}") for s in SEEDS}

for metric in ("clap_cosine", "kl_passt_softmax"):
    same, other = [], []
    same_corr, other_corr = [], []
    for si in SEEDS:
        for sj in SEEDS:
            d = (rescale[si][metric] - cfg[sj][metric]).to_numpy()
            r = np.corrcoef(rescale[si][metric], cfg[sj][metric])[0, 1]
            (same if si == sj else other).append(d.std(ddof=1))
            (same_corr if si == sj else other_corr).append(r)
    print(f"{metric}: SD of per-prompt difference rescale12 vs cfg7")
    print(f"  same seed ({len(same)} pairs): mean SD {np.mean(same):.3f}, per pair {[round(float(x), 3) for x in same]}")
    print(f"  different seed ({len(other)} pairs): mean SD {np.mean(other):.3f}")
    print(f"  ratio same/different {np.mean(same) / np.mean(other):.2f}x; equivalent prompt factor (ratio squared inverse) {(np.mean(other) / np.mean(same)) ** 2:.2f}x")
    print(f"  per-prompt Pearson correlation rescale12 vs cfg7: same seed {np.mean(same_corr):.2f}, different seed {np.mean(other_corr):.2f}")
