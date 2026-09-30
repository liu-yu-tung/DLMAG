"""A: do the production features (src/hw1/features/production.py) add to the recipe? Tests fixed before scoring.

Components (flat logistic probe, as in scripts/cv_fusion.py): M = master_* (13), H = high_* (14), MH = both.
Scored on the stratified folds, the grouped folds of scripts/grouped_cv.py, and validation (train fit; temperature
from the stratified out-of-fold set), calibrated equal weight:
  alone        hand-crafted, M, H, MH
  over R+hc    R = ordinal MERT + Qwen zero-shot, hc = current hand-crafted; add M, H, MH
  over R       add M, H, MH (in place of hand-crafted)
Writes results/cv_production.json and features/cv_A_production.npz.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from combo_tree import load
from cv_fusion import logp
from grouped_cv import calibrate
from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
R = ["mert_ord10", "qwen_zs"]


def main() -> None:
    rng = np.random.default_rng(0)
    comp, y, yv = load("A")
    ref = np.load(FEAT / "A_mertv2.npz")
    z = np.load(FEAT / "A_production.npz")
    assert (z["sample_id"] == ref["sample_id"]).all()
    tr, va = z["split"] == "train", z["split"] == "validation"
    names = z["names"]
    X = {"M": z["feats"][:, np.char.startswith(names, "master_")], "H": z["feats"][:, np.char.startswith(names, "high_")]}
    X["MH"] = z["feats"]
    g = np.load(FEAT / "cvg_A.npz")
    assert (g["y_train"] == y).all()
    folds = {"stratified": list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y)),
             "grouped": list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, g["grouped_groups"]))}
    oof = {"stratified": {c: comp[c][0] for c in R + ["handcrafted"]},
           "grouped": {c: g[f"grouped_{c}"] for c in R + ["handcrafted"]}}
    val = {c: comp[c][1] for c in R + ["handcrafted"]}
    save = {}
    for n, x in X.items():
        xtr = x[tr]
        for t, fs in folds.items():
            o = np.zeros((len(y), 6))
            for a, b in fs:
                o[b] = logp("flat", xtr[a], y[a], xtr[b])
            oof[t][n] = save[f"{t}_{n}"] = o
        val[n] = save[f"val_{n}"] = logp("flat", xtr, y, x[va])
    np.savez(FEAT / "cv_A_production.npz", **save, y_train=y, y_val=yv)

    def cal_val(c):
        o, v = norm(oof["stratified"][c]), norm(val[c])
        t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(y)), y].mean(), bounds=(0.05, 20), method="bounded").x
        return norm(v * t)
    cal = {t: {c: calibrate(o, y) for c, o in d.items()} for t, d in oof.items()}
    calv = {c: cal_val(c) for c in val}
    S = lambda cs, t: per_clip(np.mean([cal[t][c] for c in cs], 0), y)
    Sv = lambda cs: per_clip(np.mean([calv[c] for c in cs], 0), yv)

    def row(cs, base=None):
        r = {t: round(float(S(cs, t).mean()), 4) for t in folds} | {"val": round(float(Sv(cs).mean()), 4)}
        if base:
            for t in folds:
                d = S(cs, t) - S(base, t)
                r[f"{t}_diff"], r[f"{t}_ci95"] = round(float(d.mean()), 4), ci(d, rng)
            r["val_diff"] = round(float((Sv(cs) - Sv(base)).mean()), 4)
        return r
    report = {"alone": {c: row([c]) for c in ["handcrafted", "M", "H", "MH"]},
              "R": row(R), "R+hc": row(R + ["handcrafted"], R),
              "over_R+hc": {n: row(R + ["handcrafted", n], R + ["handcrafted"]) for n in ("M", "H", "MH")},
              "over_R": {n: row(R + [n], R) for n in ("M", "H", "MH")}}
    for k, v in report.items():
        print(k, json.dumps(v))
    (HW1 / "results" / "cv_production.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
