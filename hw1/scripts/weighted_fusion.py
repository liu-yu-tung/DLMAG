"""Class-specific and conditional fusion on the cached out-of-fold log-probs (level-1 outputs from scripts/cv_fusion.py
and friends). Each rule is fitted on 4 folds of the out-of-fold set and scored on the 5th (5-fold CV over the
level-1 outputs), then refitted on all of it and applied to validation. Rules:
  eq         calibrated equal-weight log-prob mean (product of experts), the current method
  perclass   z_c = sum_m w[m,c] * logp_m(c) + b_c, weights shrunk toward 1 (L2), fitted by max likelihood
  confbayes  posterior ∝ prod_m P(model m predicts k_m | class c), P from the fold confusion matrices (+1 smoothing)
  gate_lang  (B) separate model weights for clips whose Whisper top-1 language is English vs not
Writes results/weighted_fusion.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
K = 6


def fit_temp(L, y):
    from scipy.optimize import minimize_scalar
    return minimize_scalar(lambda t: -norm(L * t)[np.arange(len(y)), y].mean(), bounds=(0.05, 20), method="bounded").x


def rule_eq(Ls_tr, y, Ls_te, g_tr=None, g_te=None):
    ts = [fit_temp(L, y) for L in Ls_tr]
    return np.mean([norm(L * t) for L, t in zip(Ls_te, ts)], 0)


def rule_perclass(Ls_tr, y, Ls_te, g_tr=None, g_te=None, lam=1.0):
    M = len(Ls_tr)
    A = np.stack([norm(L) for L in Ls_tr], 0)
    B = np.stack([norm(L) for L in Ls_te], 0)
    def nll(p):
        w, b = p[: M * K].reshape(M, K), p[M * K :]
        z = norm((A * w[:, None, :]).sum(0) + b)
        return -z[np.arange(len(y)), y].mean() + lam / len(y) * ((w - 1 / M) ** 2).sum() * 100
    p0 = np.concatenate([np.full(M * K, 1 / M), np.zeros(K)])
    p = minimize(nll, p0, method="L-BFGS-B").x
    w, b = p[: M * K].reshape(M, K), p[M * K :]
    return (B * w[:, None, :]).sum(0) + b, w


def rule_confbayes(Ls_tr, y, Ls_te, g_tr=None, g_te=None):
    out = np.zeros((len(Ls_te[0]), K))
    for L, Lt in zip(Ls_tr, Ls_te):
        C = np.ones((K, K))
        np.add.at(C, (y, L.argmax(1)), 1)
        C /= C.sum(1, keepdims=True)
        out += np.log(C[:, Lt.argmax(1)].T)
    return out + 1e-6 * rule_eq(Ls_tr, y, Ls_te)  # break ties with the soft fusion


def rule_gate(Ls_tr, y, Ls_te, g_tr, g_te):
    out = np.zeros((len(Ls_te[0]), K))
    for g in (True, False):
        m_tr, m_te = g_tr == g, g_te == g
        if m_te.any():
            out[m_te] = rule_perclass([L[m_tr] for L in Ls_tr], y[m_tr], [L[m_te] for L in Ls_te])[0] if False else \
                _scalar_weights([L[m_tr] for L in Ls_tr], y[m_tr], [L[m_te] for L in Ls_te])
    return out


def _scalar_weights(Ls_tr, y, Ls_te):
    M = len(Ls_tr)
    A = np.stack([norm(L) for L in Ls_tr], 0)
    def nll(w):
        z = norm((A * w[:, None, None]).sum(0))
        return -z[np.arange(len(y)), y].mean()
    w = minimize(nll, np.full(M, 1 / M), method="L-BFGS-B", bounds=[(0, 5)] * M).x
    return (np.stack([norm(L) for L in Ls_te], 0) * w[:, None, None]).sum(0)


def load(key, comps):
    z = dict(np.load(FEAT / f"cv_{key}.npz"))
    if "demucs_all" in comps:
        d = np.load(FEAT / f"cv_{key}_demucs.npz")
        z["oof_demucs_all"], z["val_demucs_all"] = d["oof_demucs_all"], d["val_demucs_all"]
    return [z[f"oof_{c}"] for c in comps], [z[f"val_{c}"] for c in comps], z["y_train"], z["y_val"]


def main() -> None:
    rng = np.random.default_rng(0)
    setups = {"A": ["mert_mixture", "demucs_all", "handcrafted"], "B": ["mert_mixture", "lang_mixture"]}
    report = {}
    for key, comps in setups.items():
        O, V, y, yv = load(key, comps)
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        lid = np.load(FEAT / f"{key}_langid.npz")
        is_en = np.array(lid["codes"])[lid["mixture"].argmax(1)] == "en"
        g_tr, g_va = is_en[ref["split"] == "train"], is_en[ref["split"] == "validation"]
        rules = {"eq": rule_eq, "perclass": lambda *a: rule_perclass(*a)[0], "confbayes": rule_confbayes}
        if key == "B":
            rules["gate_lang"] = rule_gate
        res, base = {"components": comps}, None
        for name, fn in rules.items():
            cvp = np.zeros((len(y), K))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=2).split(y, y):
                cvp[b] = fn([L[a] for L in O], y[a], [L[b] for L in O], g_tr[a], g_tr[b])
            s = per_clip(cvp, y)
            base = s if name == "eq" else base
            vs = per_clip(fn(O, y, V, g_tr, g_va), yv)
            res[name] = {"S_cv": round(float(s.mean()), 4), "S_val": round(float(vs.mean()), 4),
                         "diff_vs_eq_ci95": ci(s - base, rng) if name != "eq" else None}
        w = rule_perclass(O, y, O)[1]
        res["perclass_weights_all_train"] = {c: dict(zip(__import__("hw1.data", fromlist=["LABELS"]).LABELS[key],
                                                         np.round(w[i], 2).tolist())) for i, c in enumerate(comps)}
        report[key] = res
        print(key, json.dumps(res), flush=True)
    (HW1 / "results" / "weighted_fusion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
