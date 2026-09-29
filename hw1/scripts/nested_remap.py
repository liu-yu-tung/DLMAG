"""Nested check of the learned output remap (a logistic regression on a recipe's component log-probs).

Outer 5-fold on train. Inside each outer-train set, 5 inner folds give out-of-fold component log-probs; the remap is
fit on those (C picked by inner cross-validation from a small grid) and applied to the outer-test log-probs of
components fit on the whole outer-train set. Compared with plain equal-weight fusion on the same outer folds.
Writes results/nested_remap.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv_fusion import logp, specs

HW1 = Path(__file__).resolve().parents[1]
RECIPES = {"A": [["mert_mixture"], ["mert_mixture", "handcrafted"]],
           "B": [["mert_mixture"], ["mert_mixture", "lang_mixture"]]}


def per_clip(lp, y):
    o = np.argsort(-lp, 1)[:, :3]
    return (o[:, 0] == y) + 0.5 * (o == y[:, None]).any(1)


def level1(comps, names, idx_fit, idx_apply, y):
    return {n: logp(comps[n][0], comps[n][1][idx_fit], y[idx_fit], comps[n][1][idx_apply]) for n in names}


def main() -> None:
    out = {}
    for key in ("A", "B"):
        comps, y, split = specs(key)
        tr = np.where(split == "train")[0]
        need = sorted({n for r in RECIPES[key] for n in r})
        res = {"+".join(r): {"eq": np.zeros(len(tr)), "remap": np.zeros(len(tr)), "C": []} for r in RECIPES[key]}
        for o, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr])):
            ia, ib = tr[a], tr[b]
            inner = {n: np.zeros((len(ia), 6)) for n in need}
            for c, d in StratifiedKFold(5, shuffle=True, random_state=1).split(ia, y[ia]):
                for n, v in level1(comps, need, ia[c], ia[d], y).items():
                    inner[n][d] = v
            outer = level1(comps, need, ia, ib, y)
            for r in RECIPES[key]:
                name = "+".join(r)
                res[name]["eq"][b] = per_clip(np.mean([outer[n] for n in r], 0), y[ib])
                X = lambda d: np.concatenate([d[n] for n in r], 1)
                gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                  {"logisticregression__C": [0.001, 0.01, 0.1]}, cv=5, scoring="neg_log_loss")
                gs.fit(X(inner), y[ia])
                res[name]["C"].append(gs.best_params_["logisticregression__C"])
                res[name]["remap"][b] = per_clip(gs.predict_log_proba(X(outer)), y[ib])
            print(key, "outer fold", o, flush=True)
        rng = np.random.default_rng(0)
        out[key] = {}
        for name, r in res.items():
            d = r["remap"] - r["eq"]
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            out[key][name] = {"S_eq": round(float(r["eq"].mean()), 4), "S_remap": round(float(r["remap"].mean()), 4),
                              "diff_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)], "C": r["C"]}
            print(key, name, out[key][name], flush=True)
    (HW1 / "results" / "nested_remap.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
