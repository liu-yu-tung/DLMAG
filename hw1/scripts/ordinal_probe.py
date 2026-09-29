"""A: MERT all-layer probe trained with ordinal soft labels (reward for neighbouring decades).

Soft-label cross-entropy via sample weights: each train clip appears once with its own decade (weight 1 - eps * n_nb)
and once per neighbouring decade (weight eps). Same 5 train folds as scripts/cv_fusion.py; eps = 0.1 fixed in
advance, 0.05 and 0.2 reported as sensitivity. Baseline (eps 0) is the cached oof_mert_mixture in features/cv_A.npz.
Writes results/ordinal_probe_A.json and features/cv_A_ordinal.npz.
"""

import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from cv_fusion import specs
from hw1.probe import make_model

HW1 = Path(__file__).resolve().parents[1]
K, EPS = 6, 1e-9


def expand(X, y, eps):
    Xs, ys, ws = [X], [y], [np.where((y == 0) | (y == K - 1), 1 - eps, 1 - 2 * eps)]
    for d in (-1, 1):
        m = (y + d >= 0) & (y + d < K)
        Xs.append(X[m]); ys.append(y[m] + d); ws.append(np.full(m.sum(), eps))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def layer(Xtr, ytr, Xte, eps):
    Xe, ye, we = expand(Xtr, ytr, eps)
    return np.log(make_model("logreg").fit(Xe, ye, logisticregression__sample_weight=we).predict_proba(Xte) + EPS)


def per_clip(lp, y):
    o = np.argsort(-lp, 1)[:, :3]
    return (o[:, 0] == y) + 0.5 * (o == y[:, None]).any(1)


def main() -> None:
    comps, y, split = specs("A")
    X = comps["mert_mixture"][1]
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    base = np.load(HW1 / "features" / "cv_A.npz")
    ytr, yva = y[tr], y[va]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, ytr))
    fit = lambda a, b, eps: np.mean(Parallel(n_jobs=4)(delayed(layer)(X[a, l], y[a], X[b, l], eps) for l in range(X.shape[1])), axis=0)
    b_oof, b_val = per_clip(base["oof_mert_mixture"], ytr), per_clip(base["val_mert_mixture"], yva)
    out, save = {"baseline_eps0": {"S_oof": round(float(b_oof.mean()), 4), "S_val": round(float(b_val.mean()), 4)}}, {}
    rng = np.random.default_rng(0)
    for eps in (0.1, 0.05, 0.2):
        oof = np.zeros((len(tr), K))
        for a, b in folds:
            oof[b] = fit(tr[a], tr[b], eps)
        val = fit(tr, va, eps)
        s_oof, s_val = per_clip(oof, ytr), per_clip(val, yva)
        d = s_oof - b_oof
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        err = np.abs(oof.argmax(1) - ytr)
        out[f"eps{eps}"] = {"S_oof": round(float(s_oof.mean()), 4), "S_val": round(float(s_val.mean()), 4),
                            "top1_oof": round(float((err == 0).mean()), 4),
                            "top3_oof": round(float((np.argsort(-oof, 1)[:, :3] == ytr[:, None]).any(1).mean()), 4),
                            "mean_abs_decades_oof": round(float(err.mean()), 3),
                            "diff_vs_eps0_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}
        save[f"oof_mert_ord{int(eps * 100):02d}"], save[f"val_mert_ord{int(eps * 100):02d}"] = oof, val
        print(eps, out[f"eps{eps}"], flush=True)
    print("baseline", out["baseline_eps0"])
    np.savez(HW1 / "features" / "cv_A_ordinal.npz", **save, y_train=ytr, y_val=yva)
    (HW1 / "results" / "ordinal_probe_A.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
