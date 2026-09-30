"""Score fine-tuned MERT top 12 (seed 0) on the stratified folds and on the grouped folds of scripts/grouped_cv.py.

Same comparisons as scripts/cv_finetune.py, calibrated equal weight, each scheme scored against its own probe and
recipe: fine-tuned alone vs the probe, fine-tuned in place of the probe, fine-tuned added to the recipe.
Stratified uses ft_{key}_L12_s0 (one seed, to match the grouped run). Writes results/grouped_ft.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from grouped_cv import calibrate
from non_mert import ci, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
RECIPE = {"A": ("mert_ord10", ["qwen_zs"]), "B": ("mert_mixture", ["lang_mixture"])}


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for key, (probe, rest) in RECIPE.items():
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids = ref["sample_id"][ref["split"] == "train"]
        g = np.load(FEAT / f"cvg_{key}.npz")
        y = g["y_train"]
        cv = np.load(FEAT / (f"cv_{key}_ordinal.npz" if probe == "mert_ord10" else f"cv_{key}.npz"))
        tt = np.load(FEAT / f"{key}_alm_plain_train_test.npz")
        p = {s: i for i, s in enumerate(tt["sample_id"])}
        raw = tt["logp"][[p[s] for s in tr_ids]]
        qz = np.zeros_like(raw)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(y, y):
            qz[b] = raw[b] - raw[a].mean(0)
        strat = {probe: cv[f"oof_{probe}"], "qwen_zs": qz,
                 "lang_mixture": np.load(FEAT / f"cv_{key}.npz")["oof_lang_mixture"]}
        grouped = {c: g[f"grouped_{c}"] for c in (probe, *rest)}
        r = {}
        for tag, comp, ft_file in (("stratified", strat, f"ft_{key}_L12_s0"), ("grouped", grouped, f"ft_{key}_L12_g_s0")):
            z = np.load(FEAT / f"{ft_file}.npz")
            assert all(int(z[f"done_f{f}"]) for f in range(5))
            pos = {s: i for i, s in enumerate(z["oof_id"])}
            cal = {c: calibrate(comp[c], y) for c in (probe, *rest)}
            cal["ft"] = calibrate(z["oof"][[pos[s] for s in tr_ids]], y)
            S = lambda cs: per_clip(np.mean([cal[c] for c in cs], 0), y)
            row = lambda cs, base: {"S": round(float(S(cs).mean()), 4), "diff": round(float((S(cs) - S(base)).mean()), 4),
                                    "ci95": ci(S(cs) - S(base), rng)}
            r[tag] = {"probe": round(float(S([probe]).mean()), 4), "recipe": round(float(S([probe, *rest]).mean()), 4),
                      "ft_alone_vs_probe": row(["ft"], [probe]),
                      "ft_swapped_in": row(["ft", *rest], [probe, *rest]),
                      "ft_added": row([probe, *rest, "ft"], [probe, *rest])}
            print(key, tag, r[tag], flush=True)
        report[key] = r
    (HW1 / "results" / "grouped_ft.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
