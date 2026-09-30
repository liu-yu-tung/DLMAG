"""Class-specific fusion rules on the recipe pools with the fine-tuned MERT top 12 (3 seeds averaged) added.

Pools: A = ordinal probe + Qwen zero-shot + ft_L12, B = probe + language + ft_L12. Rules from
scripts/weighted_fusion.py, fitted on 4 folds of the out-of-fold set and scored on the 5th, then refitted on all
of train and applied to validation:
  eq2        calibrated equal weight on the 2-part recipe (current submission shape)
  eq         calibrated equal weight on the 3-part pool
  bias       eq + one additive bias per class
  perclass   w[model, class] weights + class bias, L2 pull toward equal weight (lam 10 / 1 / 0.1 / 0)
Writes results/weighted_fusion_ft.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from combo_tree import load
from hw1.data import LABELS
from non_mert import ci, norm, per_clip
from weighted_fusion import K, rule_eq, rule_perclass

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
POOL = {"A": ["mert_ord10", "qwen_zs"], "B": ["mert_mixture", "lang_mixture"]}


def ft_l12(key, tr_ids, va_ids):
    o, v = [], []
    for s in (0, 1, 2):
        z = np.load(FEAT / f"ft_{key}_L12_s{s}.npz")
        p, vp = {x: i for i, x in enumerate(z["oof_id"])}, {x: i for i, x in enumerate(z["val_id"])}
        o.append(z["oof"][[p[i] for i in tr_ids]])
        v.append(z["val"][[vp[i] for i in va_ids]])
    return np.mean(o, 0), np.mean(v, 0)


def rule_bias(Ls_tr, y, Ls_te):
    A, B = norm(rule_eq(Ls_tr, y, Ls_tr)), norm(rule_eq(Ls_tr, y, Ls_te))
    b = minimize(lambda b: -norm(A + b)[np.arange(len(y)), y].mean(), np.zeros(K), method="L-BFGS-B").x
    return B + b


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for key, rec in POOL.items():
        comp, y, yv = load(key)
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
        comp["ft_L12"] = ft_l12(key, tr_ids, va_ids)
        names = rec + ["ft_L12"]
        O, V = [comp[c][0] for c in names], [comp[c][1] for c in names]
        rules = {"eq": rule_eq, "eq2": lambda Lt, yy, Le: rule_eq(Lt[:2], yy, Le[:2]), "bias": rule_bias}
        for lam in (10, 1, 0.1, 0):
            rules[f"perclass_lam{lam}"] = lambda Lt, yy, Le, lam=lam: rule_perclass(Lt, yy, Le, lam=lam)[0]
        res, base = {"components": names}, None
        for name, fn in rules.items():
            cvp = np.zeros((len(y), K))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=2).split(y, y):
                cvp[b] = fn([L[a] for L in O], y[a], [L[b] for L in O])
            s = per_clip(cvp, y)
            base = s if name == "eq" else base
            res[name] = {"S_cv": round(float(s.mean()), 4), "S_val": round(float(per_clip(fn(O, y, V), yv).mean()), 4),
                         "diff_vs_eq_ci95": None if name == "eq" else ci(s - base, rng)}
        w = rule_perclass(O, y, O, lam=1)[1]
        res["perclass_lam1_weights_all_train"] = {c: dict(zip(LABELS[key], np.round(w[i], 2).tolist()))
                                                  for i, c in enumerate(names)}
        report[key] = res
        print(key, json.dumps(res, indent=1), flush=True)
    (HW1 / "results" / "weighted_fusion_ft.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
