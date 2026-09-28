"""Mutual information, group redundancy, and feature-combination checks on hand-crafted features."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from hw1.data import LABELS
from hw1.features.handcrafted import feature_group
from hw1.probe import SEED, fit_eval, load_features, split_xy

HW1 = Path(__file__).resolve().parents[1]
TOP_K = [5, 10, 20, 40, 97]


def run(key: str) -> dict:
    feat = load_features(HW1 / "features" / f"{key}_handcrafted.npz")
    names = feat["names"].tolist()
    groups = np.array([feature_group(n) for n in names])
    labels = LABELS[key]
    Xtr, ytr, _ = split_xy(feat, key, "train")
    Xva, yva, _ = split_xy(feat, key, "validation")
    S = lambda cols: round(fit_eval("logreg", Xtr[:, cols], ytr, Xva[:, cols], yva, labels)[2]["S"], 4)

    mi = mutual_info_classif(Xtr, ytr, random_state=SEED)
    res: dict = {"mi_top": [(names[i], round(float(mi[i]), 4)) for i in np.argsort(-mi)[:10]]}
    res["mi_by_group"] = {
        g: {"sum": round(float(mi[groups == g].sum()), 3), "max": round(float(mi[groups == g].max()), 4)}
        for g in sorted(set(groups))
    }

    Z = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
    C = np.abs(np.corrcoef(Z, rowvar=False))
    e = groups == "energy"
    res["mean_abs_corr_with_energy"] = {
        g: round(float(C[np.ix_(e, groups == g)].mean()), 3) for g in sorted(set(groups)) if g != "energy"
    }

    all_cols = np.arange(len(names))
    res["energy_plus"] = {"energy only": S(np.where(e)[0])}
    for g in sorted(set(groups) - {"energy"}):
        res["energy_plus"][f"energy+{g}"] = S(np.where(e | (groups == g))[0])
    res["energy_plus"]["all"] = S(all_cols)

    order = np.argsort(-mi)
    res["mi_top_k"] = {str(k): S(order[:k]) for k in TOP_K}
    return res


def main() -> None:
    out = {k: run(k) for k in ("A", "B")}
    path = HW1 / "results" / "handcrafted_mi.json"
    path.write_text(json.dumps(out, indent=2))
    for k, r in out.items():
        print(f"== {k}")
        print("  MI by group:", r["mi_by_group"])
        print("  |corr| with energy:", r["mean_abs_corr_with_energy"])
        print("  energy + group S:", r["energy_plus"])
        print("  MI top-k S:", r["mi_top_k"])
        print("  top MI:", r["mi_top"][:6])
    print(path)


if __name__ == "__main__":
    main()
