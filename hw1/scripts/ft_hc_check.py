"""Task A: does hand-crafted (97 features) add to ordinal probe + Qwen zero-shot + fine-tuned top 12?

Motivation (fixed before scoring): every A model is weakest on 2000s-2010s and hand-crafted is the one component the
per-class fusion weights trusted there. Same scoring as grouped_ft_score.py: calibrated equal weight, stratified and
grouped out-of-fold, validation from the train fits; FT is the 3-seed mean, plus the difference with each single
seed (stability). Per-decade S of both fusions. Writes results/ft_hc_check.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from combo_tree import load
from hw1.data import LABELS
from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
SEEDS = [0, 1, 2]
BASE = ["mert_ord10", "qwen_zs", "ft"]


def take(x, ids, want):
    p = {s: i for i, s in enumerate(ids)}
    return x[[p[s] for s in want]]


def main() -> None:
    rng = np.random.default_rng(0)
    comp, ytr, yva = load("A")
    ref = np.load(FEAT / "A_mertv2.npz")
    tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
    g = np.load(FEAT / "cvg_A.npz")
    assert (g["y_train"] == ytr).all()
    zs = [np.load(FEAT / f"ft_A_L12_s{s}.npz") for s in SEEDS]
    zg = [np.load(FEAT / f"ft_A_L12_g_s{s}.npz") for s in SEEDS]
    assert all(int(z[f"done_f{i}"]) for z in zg + zs for i in range(5))
    ft = {"stratified": [take(z["oof"], z["oof_id"], tr_ids) for z in zs],
          "grouped": [take(z["oof"], z["oof_id"], tr_ids) for z in zg],
          "val": [take(z["val"], z["val_id"], va_ids) for z in zs]}

    def prep(o, v=None):
        o = norm(o)
        t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
        return norm(o * t), None if v is None else norm(norm(v) * t)

    out = {}
    for scheme in ("stratified", "grouped"):
        src = {c: (comp[c][0] if scheme == "stratified" else g[f"grouped_{c}"], comp[c][1])
               for c in ("mert_ord10", "qwen_zs", "handcrafted")}
        parts = {c: prep(o, v if scheme == "stratified" else None) for c, (o, v) in src.items()}
        runs = [prep(o, v) for o, v in zip(ft[scheme], ft["val"])]
        mean = prep(np.mean(ft[scheme], 0), np.mean(ft["val"], 0))

        def S(cs, f, i=0, y=ytr):
            return per_clip(np.mean([(f if c == "ft" else parts[c])[i] for c in cs], 0), y)

        new = BASE + ["handcrafted"]
        d = S(new, mean) - S(BASE, mean)
        r = {"base": round(float(S(BASE, mean).mean()), 4), "S": round(float(S(new, mean).mean()), 4),
             "diff": round(float(d.mean()), 4), "ci95": ci(d, rng),
             "per_seed_diff": [round(float((S(new, f) - S(BASE, f)).mean()), 4) for f in runs],
             "per_decade": {lab: [round(float(S(BASE, mean)[ytr == k].mean()), 3), round(float(S(new, mean)[ytr == k].mean()), 3)]
                            for k, lab in enumerate(LABELS["A"])}}
        if scheme == "stratified":
            dv = S(new, mean, 1, yva) - S(BASE, mean, 1, yva)
            r["val"] = {"base": round(float(S(BASE, mean, 1, yva).mean()), 4), "S": round(float(S(new, mean, 1, yva).mean()), 4),
                        "ci95": ci(dv, rng),
                        "per_seed_diff": [round(float((S(new, f, 1, yva) - S(BASE, f, 1, yva)).mean()), 4) for f in runs],
                        "per_decade": {lab: [round(float(S(BASE, mean, 1, yva)[yva == k].mean()), 3),
                                             round(float(S(new, mean, 1, yva)[yva == k].mean()), 3)]
                                       for k, lab in enumerate(LABELS["A"])}}
        out[scheme] = r
        print(scheme, json.dumps(r), flush=True)
    (HW1 / "results" / "ft_hc_check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
