"""Qwen2-Audio zero-shot scores in the cross-validated screen (same 5 train folds as scripts/cv_fusion.py).

The raw scores need no training, so they are already out-of-fold. Two label-bias corrections are fitted inside the
folds: (1) subtract each label's mean log-prob over the fold's training clips; (2) a logistic remap of the 6 scores
(C 0.1). Then calibrated equal-weight fusion with the current recipe, and the complementarity check on MERT's misses.
Writes results/cv_alm.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
CURRENT = {"A": ["mert_mixture"], "B": ["mert_mixture", "lang_mixture"]}


def main() -> None:
    keys = sys.argv[1:] or ["A", "B"]
    out_path = HW1 / "results" / "cv_alm.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    rng = np.random.default_rng(0)
    for key in keys:
        cv = dict(np.load(FEAT / f"cv_{key}.npz"))
        ytr, yva = cv["y_train"], cv["y_val"]
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids = ref["sample_id"][ref["split"] == "train"]
        va_ids = ref["sample_id"][ref["split"] == "validation"]
        tt = np.load(FEAT / f"{key}_alm_plain_train_test.npz")
        va = np.load(FEAT / f"{key}_alm_plain_validation.npz")
        pos = {s: i for i, s in enumerate(tt["sample_id"])}
        raw_tr = tt["logp"][[pos[s] for s in tr_ids]]
        vpos = {s: i for i, s in enumerate(va["sample_id"])}
        raw_va = va["logp"][[vpos[s] for s in va_ids]]

        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr, ytr))
        prior, remap = np.zeros_like(raw_tr), np.zeros_like(raw_tr)
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        for a, b in folds:
            prior[b] = raw_tr[b] - raw_tr[a].mean(0)
            remap[b] = mk().fit(raw_tr[a], ytr[a]).predict_log_proba(raw_tr[b])
        comps = {"alm_raw": (raw_tr, raw_va), "alm_prior": (prior, raw_va - raw_tr.mean(0)),
                 "alm_remap": (remap, mk().fit(raw_tr, ytr).predict_log_proba(raw_va))}

        def cal(o, v):
            o, v = norm(o), norm(v)
            t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
            return norm(o * t), norm(v * t)

        cur = [cal(cv[f"oof_{c}"], cv[f"val_{c}"]) for c in CURRENT[key]]
        base = per_clip(np.mean([c[0] for c in cur], 0), ytr)
        r = {"current": {"components": "+".join(CURRENT[key]), "S_oof": round(float(base.mean()), 4)}}
        wrong = np.argmax(cv["oof_mert_mixture"], 1) != ytr
        for name, (o, v) in comps.items():
            co, cvv = cal(o, v)
            f_o = per_clip(np.mean([c[0] for c in cur] + [co], 0), ytr)
            f_v = per_clip(np.mean([c[1] for c in cur] + [cvv], 0), yva)
            rank = (np.argsort(-o[wrong], 1) == ytr[wrong, None]).argmax(1) + 1
            r[name] = {"alone_S_oof": round(float(per_clip(o, ytr).mean()), 4), "alone_S_val": round(float(per_clip(v, yva).mean()), 4),
                       "current_plus_S_oof": round(float(f_o.mean()), 4), "current_plus_S_val": round(float(f_v.mean()), 4),
                       "diff_ci95": ci(f_o - base, rng),
                       "on_mert_misses": {"n": int(wrong.sum()), "mean_rank_true": round(float(rank.mean()), 3),
                                          "top1": round(float((rank == 1).mean()), 4)}}
        report[key] = r
        print(key, json.dumps(r), flush=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
