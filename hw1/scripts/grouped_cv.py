"""Proxy check for artist leakage in the 5 stratified train folds.

The manifests have no artist column, so the train folds may put one artist on both sides. Proxy: Ward clusters of
train clips on the MERT layer-mean (centered, L2-normalized) act as "sound-alike" groups; StratifiedGroupKFold(5)
keeps each group on one side. The cluster count is chosen by a rule fixed before any score is seen: the largest
count whose held-out 1-NN label match (held-out clip vs its fold's train part) is at or below the val -> train
level, which is artist-disjoint by the HW statement. A harsher setting (a quarter of that count) is also scored.
Grouping on MERT features also removes genuine same-era/same-market neighbours, so the grouped scores are a
pessimistic bound, hardest on MERT-based components.

Components recomputed with the same code as scripts/cv_fusion.py, scripts/ordinal_probe.py, scripts/cv_qwenhid.py
and combo_tree.load (Qwen zero-shot prior). Not recomputed: MERT stems, Demucs, MuQ, the CNN, fine-tuned MERT (GPU).
Writes features/cvg_{A,B}.npz and results/grouped_cv.json.
"""

import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from cv_fusion import logp, specs
from hw1.probe import make_model
from non_mert import ci, norm, per_clip
from ordinal_probe import layer as ord_layer

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
SIZES = (2, 3, 4, 6, 8, 12, 16)
QH_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
TRIES = {"A": {"mert_mixture": [], "mert_ord10": [], "ord10+qwen_zs": ["mert_ord10"],
               "mert_mixture+handcrafted": ["mert_mixture"], "ord10+qwen_zs+handcrafted": ["mert_ord10", "qwen_zs"],
               "ord10+qwen_zs+qh_llm_last": ["mert_ord10", "qwen_zs"], "ord10+qwen_zs+lang_mixture": ["mert_ord10", "qwen_zs"]},
         "B": {"mert_mixture": [], "mert_mixture+lang_mixture": ["mert_mixture"],
               "mert_mixture+lang_mixture+qh_llm_last": ["mert_mixture", "lang_mixture"],
               "mert_mixture+lang_mixture+handcrafted": ["mert_mixture", "lang_mixture"],
               "mert_mixture+qh_llm_last": ["mert_mixture"]}}
ALIAS = {"ord10": "mert_ord10"}


def unit(X):
    x = X.mean(1)
    x = x - x.mean(0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_match(x, y, folds):
    hit = np.zeros(len(y), bool)
    for a, b in folds:
        hit[b] = y[a][(x[b] @ x[a].T).argmax(1)] == y[b]
    return float(hit.mean())


def components(key, folds):
    comps, y, split = specs(key)
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    ytr = y[tr]
    out = {}
    for name in ("mert_mixture", "handcrafted", "clap", "mixbalance", "lang_vocals", "lang_mixture"):
        kind, X = comps[name]
        oof = np.zeros((len(tr), 6))
        for a, b in folds:
            oof[b] = logp(kind, X[tr[a]], ytr[a], X[tr[b]])
        out[name] = oof
        print(key, name, flush=True)
    if key == "A":
        X = comps["mert_mixture"][1]
        oof = np.zeros((len(tr), 6))
        for a, b in folds:
            oof[b] = np.mean(Parallel(n_jobs=6)(delayed(ord_layer)(X[tr[a], l], ytr[a], X[tr[b], l], 0.1)
                                                for l in range(X.shape[1])), axis=0)
        out["mert_ord10"] = oof
        print(key, "mert_ord10", flush=True)
    ref = np.load(FEAT / f"{key}_mertv2.npz")
    tr_ids = ref["sample_id"][ref["split"] == "train"]
    tt = np.load(FEAT / f"{key}_alm_plain_train_test.npz")
    p = {s: i for i, s in enumerate(tt["sample_id"])}
    raw = tt["logp"][[p[s] for s in tr_ids]]
    prior = np.zeros_like(raw)
    for a, b in folds:
        prior[b] = raw[b] - raw[a].mean(0)
    out["qwen_zs"] = prior
    z = np.load(FEAT / f"{key}_qwenhid.npz")
    pos = {s: i for i, s in enumerate(z["sample_id"])}
    Xq = z["llm_last"].astype(np.float32)[[pos[s] for s in tr_ids]]

    def one(l):
        o = np.zeros((len(tr), 6))
        for a, b in folds:
            o[b] = np.log(make_model("logreg").fit(Xq[a, l], ytr[a]).predict_proba(Xq[b, l]) + 1e-9)
        return o
    out["qh_llm_last"] = np.mean(Parallel(n_jobs=4)(delayed(one)(l) for l in QH_LAYERS), 0)
    print(key, "qh_llm_last", flush=True)
    return out


def calibrate(oof, y):
    o = norm(oof)
    t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(y)), y].mean(), bounds=(0.05, 20), method="bounded").x
    return norm(o * t)


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for key in ("A", "B"):
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        comps, y, split = specs(key)
        tr, va = split == "train", split == "validation"
        x = unit(comps["mert_mixture"][1][tr | va])
        xtr, xva = x[tr[tr | va]], x[va[tr | va]]
        ytr, yva = y[tr], y[va]
        val_nn = float((ytr[(xva @ xtr.T).argmax(1)] == yva).mean())
        strat = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ytr, ytr))
        r = {"val_to_train_nn_match": round(val_nn, 4), "stratified_nn_match": round(nn_match(xtr, ytr, strat), 4),
             "grid": {}}
        grid = {}
        for s in SIZES:
            k = len(ytr) // s
            g = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(xtr)
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(xtr, ytr, g))
            grid[s] = (k, g, folds)
            r["grid"][s] = {"n_clusters": k, "largest_cluster": int(np.bincount(g).max()),
                            "nn_match": round(nn_match(xtr, ytr, folds), 4),
                            "fold_sizes": [len(b) for _, b in folds]}
        print(key, json.dumps(r), flush=True)
        chosen = next((s for s in SIZES if r["grid"][s]["nn_match"] <= val_nn), SIZES[-1])
        harsh = min((s for s in SIZES if s >= 4 * chosen), default=SIZES[-1])
        r["chosen_mean_group_size"], r["harsh_mean_group_size"] = chosen, harsh

        cv = dict(np.load(FEAT / f"cv_{key}.npz"))
        base = {c: cv[f"oof_{c}"] for c in ("mert_mixture", "handcrafted", "clap", "mixbalance", "lang_vocals", "lang_mixture")}
        if key == "A":
            base["mert_ord10"] = np.load(FEAT / "cv_A_ordinal.npz")["oof_mert_ord10"]
        base["qh_llm_last"] = np.load(FEAT / f"cv_{key}_qwenhid.npz")["oof_qh_llm_last"]
        tr_ids = ref["sample_id"][tr]
        tt = np.load(FEAT / f"{key}_alm_plain_train_test.npz")
        p = {s: i for i, s in enumerate(tt["sample_id"])}
        raw = tt["logp"][[p[s] for s in tr_ids]]
        base["qwen_zs"] = np.zeros_like(raw)
        for a, b in strat:
            base["qwen_zs"][b] = raw[b] - raw[a].mean(0)
        schemes = {"stratified": base}
        save = {}
        for tag, s in (("grouped", chosen), ("harsh", harsh)):
            schemes[tag] = components(key, grid[s][2])
            save.update({f"{tag}_{c}": v for c, v in schemes[tag].items()})
            save[f"{tag}_groups"] = grid[s][1]
        np.savez(FEAT / f"cvg_{key}.npz", **save, y_train=ytr)

        cal = {t: {c: calibrate(o, ytr) for c, o in comp.items()} for t, comp in schemes.items()}
        r["singles"] = {c: {t: round(float(per_clip(cal[t][c], ytr).mean()), 4) for t in schemes} for c in base}
        r["fusions"] = {}
        for name, parent in TRIES[key].items():
            cs = [ALIAS.get(c, c) for c in name.split("+")]
            row = {}
            for t in schemes:
                s_ = per_clip(np.mean([cal[t][c] for c in cs], 0), ytr)
                row[t] = {"S": round(float(s_.mean()), 4)}
                if parent:
                    d = s_ - per_clip(np.mean([cal[t][c] for c in parent], 0), ytr)
                    row[t].update({"diff": round(float(d.mean()), 4), "ci95": ci(d, rng)})
            r["fusions"][name] = row
            print(key, name, row, flush=True)
        report[key] = r
    (HW1 / "results" / "grouped_cv.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
