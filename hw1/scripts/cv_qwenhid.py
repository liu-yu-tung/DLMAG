"""Cross-validated screen of Qwen2-Audio internal features (scripts/extract_qwen_hidden.py), same 5 train folds.

Per family (enc, llm_audio, llm_last) and per layer in a fixed grid, the standard logistic probe gives out-of-fold and
validation log-probs. The component is the log-prob average over the grid layers (fixed in advance, as for MERT); the
per-layer scores are reported for the curve only. Then calibrated equal-weight fusion against the current recipe:
  A: current = ordinal MERT + zero-shot Qwen (label-mean removed); also tried: ordinal MERT + probe (probe replaces
     the zero-shot scores), and current + probe.
  B: current = MERT + language; tried: current + probe.
Writes results/cv_qwenhid.json and features/cv_{key}_qwenhid.npz.
"""

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from hw1.probe import make_model
from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
LAYERS = {"enc": [4, 8, 12, 16, 20, 24, 28, 32], "llm_audio": [4, 8, 12, 16, 20, 24, 28, 32],
          "llm_last": [4, 8, 12, 16, 20, 24, 28, 32]}


def probe(Xtr, ytr, Xva, folds):
    oof = np.zeros((len(ytr), len(np.unique(ytr))))
    for a, b in folds:
        oof[b] = np.log(make_model("logreg").fit(Xtr[a], ytr[a]).predict_proba(Xtr[b]) + 1e-9)
    return oof, np.log(make_model("logreg").fit(Xtr, ytr).predict_proba(Xva) + 1e-9)


def main() -> None:
    keys = sys.argv[1:] or ["A", "B"]
    out_path = HW1 / "results" / "cv_qwenhid.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    rng = np.random.default_rng(0)
    for key in keys:
        cv = dict(np.load(FEAT / f"cv_{key}.npz"))
        ytr, yva = cv["y_train"], cv["y_val"]
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
        z = np.load(FEAT / f"{key}_qwenhid.npz")
        pos = {s: i for i, s in enumerate(z["sample_id"])}
        itr, iva = [pos[s] for s in tr_ids], [pos[s] for s in va_ids]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr, ytr))

        comps, r, save = {}, {"per_layer": {}}, {}
        for fam, layers in LAYERS.items():
            X = z[fam].astype(np.float32)
            res = Parallel(n_jobs=4)(delayed(probe)(X[itr, l], ytr, X[iva, l], folds) for l in layers)
            r["per_layer"][fam] = {l: [round(float(per_clip(o, ytr).mean()), 4), round(float(per_clip(v, yva).mean()), 4)]
                                   for l, (o, v) in zip(layers, res)}
            o, v = np.mean([x[0] for x in res], 0), np.mean([x[1] for x in res], 0)
            comps[f"qh_{fam}"] = (o, v)
            save[f"oof_qh_{fam}"], save[f"val_qh_{fam}"] = o, v
            r[f"qh_{fam}"] = {"alone_S_oof": round(float(per_clip(o, ytr).mean()), 4),
                              "alone_S_val": round(float(per_clip(v, yva).mean()), 4)}
            print(key, fam, r["per_layer"][fam], r[f"qh_{fam}"], flush=True)

        if key == "A":
            ordz = np.load(FEAT / "cv_A_ordinal.npz")
            assert (ordz["y_train"] == ytr).all() and (ordz["y_val"] == yva).all()
            comps["mert_ord10"] = (ordz["oof_mert_ord10"], ordz["val_mert_ord10"])
            tt, va = np.load(FEAT / "A_alm_plain_train_test.npz"), np.load(FEAT / "A_alm_plain_validation.npz")
            p, vp = {s: i for i, s in enumerate(tt["sample_id"])}, {s: i for i, s in enumerate(va["sample_id"])}
            raw_tr, raw_va = tt["logp"][[p[s] for s in tr_ids]], va["logp"][[vp[s] for s in va_ids]]
            prior = np.zeros_like(raw_tr)
            for a, b in folds:
                prior[b] = raw_tr[b] - raw_tr[a].mean(0)
            comps["alm_qwen"] = (prior, raw_va - raw_tr.mean(0))
            current = ["mert_ord10", "alm_qwen"]
            tries = {f"{c}+{f}": c.split("+") + [f] for f in [k for k in comps if k.startswith("qh_")]
                     for c in ["mert_ord10", "mert_ord10+alm_qwen"]}
        else:
            for c in ("mert_mixture", "lang_mixture"):
                comps[c] = (cv[f"oof_{c}"], cv[f"val_{c}"])
            current = ["mert_mixture", "lang_mixture"]
            tries = {f"current+{f}": current + [f] for f in [k for k in comps if k.startswith("qh_")]}

        def cal(o, v):
            o, v = norm(o), norm(v)
            t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
            return norm(o * t), norm(v * t)

        C = {k: cal(*v) for k, v in comps.items()}
        fuse = lambda names, i, y: per_clip(np.mean([C[n][i] for n in names], 0), y)
        base_o, base_v = fuse(current, 0, ytr), fuse(current, 1, yva)
        r["current"] = {"components": "+".join(current), "S_oof": round(float(base_o.mean()), 4),
                        "S_val": round(float(base_v.mean()), 4)}
        for name, names in tries.items():
            fo, fv = fuse(names, 0, ytr), fuse(names, 1, yva)
            r[name] = {"S_oof": round(float(fo.mean()), 4), "S_val": round(float(fv.mean()), 4),
                       "diff_vs_current_ci95": ci(fo - base_o, rng), "val_diff_ci95": ci(fv - base_v, rng)}
            print(key, name, r[name], flush=True)
        report[key] = r
        np.savez(FEAT / f"cv_{key}_qwenhid.npz", **save, y_train=ytr, y_val=yva)
        out_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
