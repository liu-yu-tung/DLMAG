"""Score fine-tuned MERT top 12 on the stratified folds and on the grouped folds of scripts/grouped_cv.py, over seeds.

Seeds used: those with all 5 grouped folds (ft_{key}_L12_g_s{seed}) and a finished stratified run (ft_{key}_L12_s{seed}),
so both fold schemes see the same seeds. Options, each against its own base, fused by the equal-weight average with the
recipe's other part (Qwen2-Audio zero-shot on A, Whisper language on B): fine-tuned alone vs the probe, fine-tuned in
place of the probe, fine-tuned added to the recipe. Each option gets the difference per seed (stability) and for the
seed mean (log-prob mean, as it would be deployed) with its paired bootstrap CI, on both fold schemes, plus validation S
from the stratified full fits. Fusion is calibrated (temperature fitted on the out-of-fold set of the same scheme;
validation uses the stratified temperature); B is also scored as submitted (no calibration, final.py).
Writes results/grouped_ft.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from combo_tree import load
from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
RECIPE = {"A": ["mert_ord10", "qwen_zs"], "B": ["mert_mixture", "lang_mixture"]}
FUSION = {"A": ["calibrated"], "B": ["calibrated", "submitted"]}


def done(name: str) -> bool:
    f = FEAT / f"{name}.npz"
    return f.exists() and all(f"done_f{i}" in (z := np.load(f)).files and int(z[f"done_f{i}"]) for i in range(5))


def take(x, ids, want):
    p = {s: i for i, s in enumerate(ids)}
    return x[[p[s] for s in want]]


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for key, rec in RECIPE.items():
        comp, ytr, yva = load(key)
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
        g = np.load(FEAT / f"cvg_{key}.npz")
        assert (g["y_train"] == ytr).all()
        seeds = [s for s in range(10) if done(f"ft_{key}_L12_g_s{s}") and done(f"ft_{key}_L12_s{s}")]
        if not seeds:
            continue
        ft = {"stratified": [], "grouped": [], "val": []}
        for s in seeds:
            zs, zg = np.load(FEAT / f"ft_{key}_L12_s{s}.npz"), np.load(FEAT / f"ft_{key}_L12_g_s{s}.npz")
            ft["stratified"].append(take(zs["oof"], zs["oof_id"], tr_ids))
            ft["grouped"].append(take(zg["oof"], zg["oof_id"], tr_ids))
            ft["val"].append(take(zs["val"], zs["val_id"], va_ids))
        oof = {"stratified": {c: comp[c][0] for c in rec}, "grouped": {c: g[f"grouped_{c}"] for c in rec}}
        val = {c: comp[c][1] for c in rec}
        probe, rest = rec[0], rec[1:]
        options = {"alone": (["ft"], [probe]), "swapped_in": (["ft", *rest], rec), "added": ([*rec, "ft"], rec)}
        report[key] = {"seeds": seeds}
        clips = {}
        for fusion in FUSION[key]:

            def prep(o, v=None):
                o = norm(o)
                t = 1.0
                if fusion == "calibrated":
                    t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20),
                                        method="bounded").x
                return norm(o * t), None if v is None else norm(norm(v) * t)

            r = {name: {} for name in options}
            for scheme in ("stratified", "grouped"):
                parts = {c: prep(oof[scheme][c], val[c] if scheme == "stratified" else None) for c in rec}
                runs = [prep(o, v) for o, v in zip(ft[scheme], ft["val"])]
                mean = prep(np.mean(ft[scheme], 0), np.mean(ft["val"], 0))

                def S(cs, f, i=0, y=ytr):
                    return per_clip(np.mean([(f if c == "ft" else parts[c])[i] for c in cs], 0), y)

                clips[fusion, scheme] = {"recipe": (S(rec, mean), [S(rec, mean)] * len(runs)),
                                         **{n: (S(cs, mean), [S(cs, f) for f in runs]) for n, (cs, _) in options.items()}}
                if scheme == "stratified":
                    clips[fusion, "val"] = {"recipe": (S(rec, mean, 1, yva), [S(rec, mean, 1, yva)] * len(runs)),
                                            **{n: (S(cs, mean, 1, yva), [S(cs, f, 1, yva) for f in runs])
                                               for n, (cs, _) in options.items()}}
                for name, (cs, base) in options.items():
                    d = S(cs, mean) - S(base, mean)
                    r[name][scheme] = {"S": round(float(S(cs, mean).mean()), 4), "base": round(float(S(base, mean).mean()), 4),
                                       "diff": round(float(d.mean()), 4), "ci95": ci(d, rng),
                                       "per_seed_diff": [round(float((S(cs, f) - S(base, f)).mean()), 4) for f in runs]}
                    if scheme == "stratified":
                        r[name]["val"] = {"S": round(float(S(cs, mean, 1, yva).mean()), 4),
                                          "base": round(float(S(base, mean, 1, yva).mean()), 4),
                                          "per_seed": [round(float(S(cs, f, 1, yva).mean()), 4) for f in runs]}
            report[key][fusion] = r
            for name, x in r.items():
                print(f"{key} {fusion:10s} {name:10s} seeds {seeds}"
                      f" | strat {x['stratified']['base']:.3f}->{x['stratified']['S']:.3f} {x['stratified']['diff']:+.3f}"
                      f" {x['stratified']['ci95']} per seed {x['stratified']['per_seed_diff']}"
                      f" | grouped {x['grouped']['base']:.3f}->{x['grouped']['S']:.3f} {x['grouped']['diff']:+.3f}"
                      f" {x['grouped']['ci95']} per seed {x['grouped']['per_seed_diff']}"
                      f" | val {x['val']['base']:.3f}->{x['val']['S']:.3f} per seed {x['val']['per_seed']}", flush=True)
        if key == "B":
            vs = {}
            for name in ("recipe", *options):
                vs[name] = {}
                for scheme in ("stratified", "grouped", "val"):
                    (m, per), (b, _) = clips["calibrated", scheme][name], clips["submitted", scheme]["recipe"]
                    vs[name][scheme] = {"diff": round(float((m - b).mean()), 4), "ci95": ci(m - b, rng),
                                        "per_seed_diff": [round(float((p - b).mean()), 4) for p in per]}
                print(f"B calibrated {name:10s} vs submitted recipe | "
                      + " | ".join(f"{s} {v['diff']:+.3f} {v['ci95']} per seed {v['per_seed_diff']}" for s, v in vs[name].items()),
                      flush=True)
            report[key]["calibrated_vs_submitted"] = vs
    (HW1 / "results" / "grouped_ft.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
