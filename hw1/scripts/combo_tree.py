"""Combination tree over all cached components, for the report and the progress page.

Every component has 5-fold out-of-fold log-probs on train (same folds) and validation log-probs from a train fit.
Fusion is the calibrated equal-weight mean (temperature per component fitted on the out-of-fold set, as in final.py).
Routes, all scored the same way:
  singles  every component and its variants alone
  level 1  the MERT representative + each other component
  level 2+ greedy: the best combination so far (by out-of-fold S) + each remaining component, until all are in
  no MERT  the same greedy route started from the best non-MERT component
Greedy picks use the out-of-fold set, so later levels are slightly optimistic. Each fused row has the paired
bootstrap CI of its difference to the combination it extends. Rows whose components all have grouped out-of-fold
log-probs (features/cvg_{key}.npz, scripts/grouped_cv.py) also get the grouped 5-fold S ("grp") and its difference.
Writes results/combo_tree.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"

LANES = {
    "A": [("mert", "MERT-v2", "enc", "mert_ord10", [("mert_mixture", "plain labels"), ("mert_ord10", "ordinal labels, eps 0.1"),
                                                    ("mert_accompaniment", "on the accompaniment stem"), ("mert_vocals", "on the vocal stem")]),
          ("qwen_zs", "Qwen2-Audio zero-shot", "alm", "qwen_zs", [("qwen_zs", "label-mean removed")]),
          ("qwen_hid", "Qwen2-Audio internal", "alm", "qh_llm_last", [("qh_llm_last", "LLM last state"), ("qh_llm_audio", "LLM over audio"),
                                                                       ("qh_enc", "audio encoder")]),
          ("demucs", "Demucs U-Net encoder", "enc", "demucs_all", [("demucs_all", "11 layers averaged")]),
          ("muq", "MuQ-large", "enc", "muq", [("muq", "12 layers averaged")]),
          ("cnn", "Short-Chunk CNN", "cnn", "cnn", [("cnn", "from scratch"), ("cnn_nogain", "no gain augmentation"),
                                                     ("cnn_ord10", "ordinal targets")]),
          ("hc", "Hand-crafted", "hc", "handcrafted", [("handcrafted", "97 features")]),
          ("mix", "Mix balance", "hc", "mixbalance", [("mixbalance", "29 stem features")]),
          ("lang", "Whisper language ID", "alm", "lang_mixture", [("lang_mixture", "from the mixture"), ("lang_vocals", "from the vocal stem")]),
          ("clap", "CLAP", "enc", "clap", [("clap", "audio embedding")])],
}
LANES["B"] = [(i, n, f, r, [v for v in vs if v[0] not in ("mert_ord10", "cnn_nogain", "cnn_ord10")])
              for i, n, f, r, vs in LANES["A"]]
LANES["B"][0] = ("mert", "MERT-v2", "enc", "mert_mixture", LANES["B"][0][4])


def load(key: str):
    cv = dict(np.load(FEAT / f"cv_{key}.npz"))
    ytr, yva = cv["y_train"], cv["y_val"]
    comp = {k[4:]: (cv[k], cv["val_" + k[4:]]) for k in cv if k.startswith("oof_")}
    for f in (f"cv_{key}_demucs", f"cv_{key}_muq", f"cv_{key}_qwenhid", f"cv_{key}_ordinal"):
        if (FEAT / f"{f}.npz").exists():
            z = np.load(FEAT / f"{f}.npz")
            comp.update({k[4:]: (z[k], z["val_" + k[4:]]) for k in z.files if k.startswith("oof_")})
    ref = np.load(FEAT / f"{key}_mertv2.npz")
    tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr, ytr))
    tt, va = np.load(FEAT / f"{key}_alm_plain_train_test.npz"), np.load(FEAT / f"{key}_alm_plain_validation.npz")
    p, vp = {s: i for i, s in enumerate(tt["sample_id"])}, {s: i for i, s in enumerate(va["sample_id"])}
    raw_tr, raw_va = tt["logp"][[p[s] for s in tr_ids]], va["logp"][[vp[s] for s in va_ids]]
    prior = np.zeros_like(raw_tr)
    for a, b in folds:
        prior[b] = raw_tr[b] - raw_tr[a].mean(0)
    comp["qwen_zs"] = (prior, raw_va - raw_tr.mean(0))
    vpos = {s: i for i, s in enumerate(va_ids)}
    for tag in ("", "_ord10", "_nogain"):
        files = [FEAT / f"{key}_cnn_s0_f{f}{tag}_logp.npz" for f in range(5)]
        if all(f.exists() for f in files):
            o, v = np.zeros((len(ytr), 6)), np.zeros((len(yva), 6))
            for f in files:
                d = np.load(f)
                o[d["oof_pos"]] = d["oof"]
                v[[vpos[s] for s in d["val_id"]]] += d["val"] / len(files)
            comp[f"cnn{tag}"] = (o, v)
    return comp, ytr, yva


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for key in ("A", "B"):
        comp, ytr, yva = load(key)

        def temp(o):
            return minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
        cal = {}
        for c, (o, v) in comp.items():
            o, v = norm(o), norm(v)
            t = temp(o)
            cal[c] = (norm(o * t), norm(v * t))
        g = np.load(FEAT / f"cvg_{key}.npz")
        assert (g["y_train"] == ytr).all()
        gcal = {k[8:]: norm(norm(g[k]) * temp(norm(g[k]))) for k in g.files if k.startswith("grouped_") and k != "grouped_groups"}
        rng_g = np.random.default_rng(1)

        def score(cs):
            po = per_clip(np.mean([cal[c][0] for c in cs], 0), ytr)
            pv = per_clip(np.mean([cal[c][1] for c in cs], 0), yva)
            return po, pv

        def row(cs, parent=None):
            po, pv = score(cs)
            r = {"comps": list(cs), "oof": round(float(po.mean()), 4), "val": round(float(pv.mean()), 4)}
            if parent:
                bo, _ = score(parent)
                r["diff"] = round(float(po.mean() - bo.mean()), 4)
                r["ci95"] = ci(po - bo, rng)
            if all(c in gcal for c in cs):
                pg = per_clip(np.mean([gcal[c] for c in cs], 0), ytr)
                r["grp"] = round(float(pg.mean()), 4)
                if parent and all(c in gcal for c in parent):
                    bg = per_clip(np.mean([gcal[c] for c in parent], 0), ytr)
                    r["grp_diff"], r["grp_ci95"] = round(float(pg.mean() - bg.mean()), 4), ci(pg - bg, rng_g)
            return r

        lanes = LANES[key]
        reps = [l[3] for l in lanes]
        singles = {v: row([v]) for l in lanes for v, _ in l[4]}

        def greedy(start, pool, full_levels):
            levels, cur = [], [start]
            while True:
                rest = [c for c in pool if c not in cur]
                if not rest:
                    break
                rows = sorted([row(cur + [c], cur) for c in rest], key=lambda r: -r["oof"])
                levels.append({"base": list(cur), "rows": rows if len(levels) < full_levels else rows[:1]})
                cur = rows[0]["comps"]
            return levels

        mert = reps[0]
        non = [c for c in reps if c != mert]
        best_non = max(non, key=lambda c: singles[c]["oof"])
        out[key] = {"lanes": [{"id": i, "name": n, "family": f, "rep": r, "variants": [{"id": v, "name": vn} for v, vn in vs]}
                              for i, n, f, r, vs in lanes],
                    "singles": singles,
                    "with_mert": greedy(mert, reps, 3),
                    "no_mert": greedy(best_non, non, 0),
                    "all": row(reps)}
        best = max((r for lv in out[key]["with_mert"] for r in lv["rows"]), key=lambda r: r["oof"])
        print(key, "best single", max(singles.items(), key=lambda kv: kv[1]["oof"]), "| best combo", best, "| all", out[key]["all"])
    (HW1 / "results" / "combo_tree.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
