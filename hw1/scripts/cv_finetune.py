"""Score the fine-tuned MERT-v2 top blocks (scripts/finetune_mert.py) in the 5-fold screen.

Seeds of one setting are averaged (log-prob mean) into one component. Compared, calibrated equal weight as everywhere:
the component alone vs the MERT probe it would replace, the recipe with the probe swapped for it, and the recipe with
it added. Only settings with all 5 folds and the full fit are scored. Writes results/cv_finetune.json.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from combo_tree import load
from non_mert import ci, norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
RECIPE = {"A": ["mert_ord10", "qwen_zs"], "B": ["mert_mixture", "lang_mixture"]}


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for key in ("A", "B"):
        files = sorted(glob.glob(str(FEAT / f"ft_{key}_L*_s*.npz")))
        if not files:
            continue
        comp, ytr, yva = load(key)
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
        settings = {}
        for f in files:
            z = dict(np.load(f))
            if not all(f"done_f{i}" in z for i in range(5)) or "val" not in z:
                continue
            p, vp = {s: i for i, s in enumerate(z["oof_id"])}, {s: i for i, s in enumerate(z["val_id"])}
            L, s = re.search(r"_L(\d+)_s(\d+)", f).groups()
            settings.setdefault(f"ft_L{L}", []).append((z["oof"][[p[i] for i in tr_ids]], z["val"][[vp[i] for i in va_ids]], int(s)))
        cal = {}

        def calib(o, v):
            o, v = norm(o), norm(v)
            t = minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
            return norm(o * t), norm(v * t)

        for c in RECIPE[key]:
            cal[c] = calib(*comp[c])
        res = {}
        for name, runs in settings.items():
            cal[name] = calib(np.mean([r[0] for r in runs], 0), np.mean([r[1] for r in runs], 0))
            sc = lambda cs, i, y: per_clip(np.mean([cal[c][i] for c in cs], 0), y)
            probe, rec = RECIPE[key][0], RECIPE[key]
            swap = [name] + rec[1:]
            r = {"seeds": sorted(r[2] for r in runs),
                 "per_seed_oof": [round(float(per_clip(x[0], ytr).mean()), 4) for x in runs]}
            for label, cs, base in (("alone", [name], [probe]), ("swap_into_recipe", swap, rec), ("add_to_recipe", rec + [name], rec)):
                fo, bo = sc(cs, 0, ytr), sc(base, 0, ytr)
                r[label] = {"S_oof": round(float(fo.mean()), 4), "S_val": round(float(sc(cs, 1, yva).mean()), 4),
                            "base": "+".join(base), "base_S_oof": round(float(bo.mean()), 4),
                            "base_S_val": round(float(sc(base, 1, yva).mean()), 4), "diff_ci95": ci(fo - bo, rng)}
            res[name] = r
            print(key, name, json.dumps(r), flush=True)
        out[key] = res
    (HW1 / "results" / "cv_finetune.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
