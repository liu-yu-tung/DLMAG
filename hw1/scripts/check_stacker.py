"""Robustness check for cv_fusion: equal-weight vs temperature-calibrated fusion vs a logistic stacker at three C values, on the cached out-of-fold log-probs (features/cv_{A,B}.npz). Prints OOF/validation S."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

def S(lp, y):
    o = np.argsort(-lp, 1)[:, :3]
    return float(((o[:, 0] == y) + 0.5 * (o == y[:, None]).any(1)).mean())

def norm(z):
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))

for key in "AB":
    z = dict(np.load(f"features/cv_{key}.npz"))
    ytr, yva = z["y_train"], z["y_val"]
    names = [k[4:] for k in z if k.startswith("oof_")]
    print(f"== {key}")
    nll = lambda lp, y: -lp[np.arange(len(y)), y].mean()
    temps = {}
    for n in names:
        lp = z[f"oof_{n}"]
        temps[n] = minimize_scalar(lambda t: nll(norm(lp * t), ytr), bounds=(0.05, 20), method="bounded").x
    print(" fitted temperature on OOF:", {n: round(t, 2) for n, t in temps.items()})
    groups = {"all 11": names, "MERT 6 inputs": [n for n in names if n.startswith("mert")],
              "MERT mix only": ["mert_mixture"], "MERT mix + lang_mixture": ["mert_mixture", "lang_mixture"],
              "MERT mix + handcrafted": ["mert_mixture", "handcrafted"]}
    for g, comps in groups.items():
        X = lambda pre: np.concatenate([z[f"{pre}_{c}"] for c in comps], 1)
        eq_o = S(np.mean([z[f"oof_{c}"] for c in comps], 0), ytr); eq_v = S(np.mean([z[f"val_{c}"] for c in comps], 0), yva)
        tc_o = S(np.mean([norm(z[f"oof_{c}"] * temps[c]) for c in comps], 0), ytr)
        tc_v = S(np.mean([norm(z[f"val_{c}"] * temps[c]) for c in comps], 0), yva)
        row = [f"eq {eq_o:.3f}/{eq_v:.3f}", f"temp+eq {tc_o:.3f}/{tc_v:.3f}"]
        for C in (0.001, 0.01, 0.1):
            stk = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            po = cross_val_predict(stk, X("oof"), ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_log_proba")
            pv = stk.fit(X("oof"), ytr).predict_log_proba(X("val"))
            row.append(f"stk C={C} {S(po, ytr):.3f}/{S(pv, yva):.3f}")
        print(f" {g:26s}", " | ".join(row))
