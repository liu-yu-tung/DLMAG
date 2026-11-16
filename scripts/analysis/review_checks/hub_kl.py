import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, per_prompt

HUB_MIN = 3
retrieval = json.loads((RESULTS / "retrieval.json").read_text())
counts = Counter(retrieval.values())
k_occurrence = {int(i): counts[clip] for i, clip in retrieval.items()}
print(f"{len(retrieval)} test captions, {len(counts)} distinct reference clips, max k-occurrence {max(counts.values())}")
print(f"k-occurrence histogram (k: number of clips): {dict(sorted(Counter(counts.values()).items()))}")

for name in ("test__cfg7", "test__cfg7_seed1", "test__cfg7_seed2"):
    frame = per_prompt(name)
    k = np.array([k_occurrence[i] for i in frame.index])
    hub = k >= HUB_MIN
    kl = frame["kl_passt_softmax"].to_numpy()
    rho, p = stats.spearmanr(k, kl)
    welch = stats.ttest_ind(kl[hub], kl[~hub], equal_var=False).pvalue
    print(f"{name}: captions with k >= {HUB_MIN}: {int(hub.sum())}, KL-PaSST mean {kl[hub].mean():.2f} vs rest {kl[~hub].mean():.2f} "
          f"(Welch p {welch:.4f}); Spearman of KL with k-occurrence {rho:+.2f} (p {p:.4f}), n {len(kl)}")
