"""Why validation S sits above 5-fold S: validation scores of the 5 fold models (trained on 80% of train) vs the
full-train model, and the bootstrap SE of validation S. Run from scripts/ with PYTHONPATH=.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from train_final import cached
from hw1 import final
from non_mert import per_clip

def fitpred(comp, X, y, a, Xv, k):
    m = final._fit_component(comp, X[a], y[a], k)
    return final._component_logprobs(comp, m, Xv)

for key, comps in (("A", ["mert_layeravg", "mert_ord10"]), ("B", ["mert_layeravg"])):
    f, y, sp, ids = cached(key)
    tr, va = sp == "train", sp == "validation"
    ytr, yva, k = y[tr], y[va], 6
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr, ytr))
    for c in comps:
        M = f.get("mert_layeravg", f.get("mert_ord10")); X, Xv = M[tr], M[va]
        full = fitpred(c, X, ytr, np.arange(len(ytr)), Xv, k)
        per = Parallel(n_jobs=5)(delayed(fitpred)(c, X, ytr, a, Xv, k) for a, _ in folds)
        s80 = [per_clip(p, yva).mean() for p in per]
        pc = per_clip(full, yva)
        rng = np.random.default_rng(0)
        se = np.std([pc[rng.integers(0, len(pc), len(pc))].mean() for _ in range(2000)])
        print(f"{key} {c}: val S from 100% train {pc.mean():.3f} (bootstrap SE {se:.3f}) | from each 80% fold model {np.round(s80,3).tolist()} mean {np.mean(s80):.3f}", flush=True)
