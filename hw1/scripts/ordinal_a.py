"""Task A ordinal analysis: how far off are the decade errors, and does ordinal post-processing help?

Out-of-fold predictions on train (5-fold, current A recipe) give the larger sample; validation (train-only fit) is
the check. Smoothing strength and rules are chosen on the out-of-fold set only. Writes results/ordinal_A.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from hw1 import final
from hw1.data import LABELS
from hw1.metrics import evaluate
from hw1.probe import load_features

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
K = 6


def load():
    mert = load_features(FEAT / "A_mertv2.npz")
    ch = load_features(FEAT / "A_handcrafted_chunks.npz")
    if not (ch["sample_id"] == mert["sample_id"]).all():
        raise ValueError("clip order differs")
    feats = {"mert_layeravg": final.mert_means(mert["feats"]), "hc_c10": ch[f"c{final.HC_CHUNK_S}"]}
    y = np.array([LABELS["A"].index(l) if l else -1 for l in mert["label"]])
    return feats, y, mert["split"]


def smooth(p: np.ndarray, e: float) -> np.ndarray:
    """Mix each class's probability with its neighbours: p'[k] = (1-2e) p[k] + e p[k-1] + e p[k+1] (edges reflect)."""
    q = (1 - 2 * e) * p
    q[:, 1:] += e * p[:, :-1]
    q[:, :-1] += e * p[:, 1:]
    q[:, 0] += e * p[:, 0]
    q[:, -1] += e * p[:, -1]
    return q


def window3(p: np.ndarray) -> np.ndarray:
    """Top-3 set as the best run of three adjacent decades; returns (N, 3) class ids sorted with top-1 first."""
    s = np.stack([p[:, i : i + 3].sum(1) for i in range(K - 2)], 1)
    start = s.argmax(1)
    out = np.zeros((len(p), 3), int)
    for n in range(len(p)):
        ids = np.arange(start[n], start[n] + 3)
        top1 = p[n].argmax()
        if top1 not in ids:
            top1 = ids[p[n, ids].argmax()]
        out[n] = [top1] + [i for i in ids[np.argsort(-p[n, ids])] if i != top1]
    return out


def score(p: np.ndarray, y: np.ndarray, top3: np.ndarray | None = None) -> dict:
    if top3 is None:
        top3 = np.argsort(-p, 1)[:, :3]
    t1 = float((top3[:, 0] == y).mean())
    t3 = float((top3 == y[:, None]).any(1).mean())
    return {"top1": round(t1, 4), "top3": round(t3, 4), "S": round(t1 + 0.5 * t3, 4)}


def offsets(p: np.ndarray, y: np.ndarray) -> dict:
    d = np.abs(p.argmax(1) - y)
    top3 = np.argsort(-p, 1)[:, :3]
    contig = np.mean([np.ptp(np.sort(t)) == 2 for t in top3])
    wrong = d > 0
    return {"abs_error_share": {str(k): round(float((d == k).mean()), 4) for k in range(K)},
            "mean_abs_decades": round(float(d.mean()), 3),
            "within_1_of_wrong": round(float((d[wrong] == 1).mean()), 4),
            "top3_is_three_adjacent_decades": round(float(contig), 4)}


def main() -> None:
    feats, y, split = load()
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    sel = lambda idx: {c: v[idx] for c, v in feats.items()}

    oof = np.zeros((len(tr), K))
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr])):
        ck = final.fit("A", sel(tr[a]), y[tr[a]])
        oof[b] = final.predict_proba(ck, sel(tr[b]))
        print("fold", f, flush=True)
    ck = final.fit("A", sel(tr), y[tr])
    pva = final.predict_proba(ck, sel(va))
    ytr, yva = y[tr], y[va]

    chance = float(np.abs(np.arange(K)[:, None] - np.arange(K)[None]).mean())
    out = {"chance_mean_abs_decades": round(chance, 3),
           "oof": {"base": score(oof, ytr), **offsets(oof, ytr)},
           "validation": {"base": score(pva, yva), **offsets(pva, yva)}}

    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    out["smoothing_grid_oof"] = {str(e): score(smooth(oof, e), ytr) for e in grid}
    best_e = max(grid, key=lambda e: out["smoothing_grid_oof"][str(e)]["S"])
    out["best_e_on_oof"] = best_e
    out["smoothing_validation"] = score(smooth(pva, best_e), yva)
    out["window3_oof"] = score(oof, ytr, window3(oof))
    out["window3_validation"] = score(pva, yva, window3(pva))
    out["expected_decade_round_oof"] = {
        "top1": round(float((np.rint(oof @ np.arange(K)) == ytr).mean()), 4)}
    (HW1 / "results" / "ordinal_A.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
