"""MuQ in the cross-validated screen (same 5 train folds as scripts/cv_fusion.py): all-layer average of per-layer
probes on the time-mean embedding, alone and fused (calibrated equal weight) with the current recipes.
Writes results/cv_muq.json and features/cv_{key}_muq.npz.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cv_fusion import logp
from hw1.data import LABELS
from hw1.probe import load_features
from non_mert import ci, norm, per_clip
from scipy.optimize import minimize_scalar

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"


def main() -> None:
    rng, report = np.random.default_rng(0), {}
    for key in ("A", "B"):
        f = load_features(FEAT / f"{key}_muq.npz")
        assert (f["sample_id"] == load_features(FEAT / f"{key}_mertv2.npz")["sample_id"]).all()
        X = f["feats"][:, :, : f["feats"].shape[2] // 2].astype(np.float32)
        y = np.array([LABELS[key].index(l) if l else -1 for l in f["label"]])
        tr, va = np.where(f["split"] == "train")[0], np.where(f["split"] == "validation")[0]
        oof = np.zeros((len(tr), 6))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]):
            oof[b] = logp("layers", X[tr[a]], y[tr[a]], X[tr[b]])
        val = logp("layers", X[tr], y[tr], X[va])
        np.savez(FEAT / f"cv_{key}_muq.npz", oof_muq=oof, val_muq=val)
        cv = dict(np.load(FEAT / f"cv_{key}.npz"))
        cv["oof_muq"], cv["val_muq"] = oof, val
        ytr, yva = cv["y_train"], cv["y_val"]

        def cal(pre, c):
            l = norm(cv[f"oof_{c}"])
            t = minimize_scalar(lambda t: -norm(l * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x
            return norm(cv[f"{pre}_{c}"] * t)

        current = ["mert_mixture"] if key == "A" else ["mert_mixture", "lang_mixture"]
        base = per_clip(np.mean([cal("oof", c) for c in current], 0), ytr)
        r = {"per_layer_oof": None, "current": {"S_oof": round(float(base.mean()), 4)}}
        for name, combo in [("muq", ["muq"]), ("mert+muq", ["mert_mixture", "muq"]),
                            ("current+muq", current + ["muq"]), ("muq+lang", ["muq", "lang_mixture"])]:
            so = per_clip(np.mean([cal("oof", c) for c in combo], 0), ytr)
            sv = per_clip(np.mean([cal("val", c) for c in combo], 0), yva)
            r[name] = {"S_oof": round(float(so.mean()), 4), "S_val": round(float(sv.mean()), 4), "diff_vs_current_ci95": ci(so - base, rng)}
        r.pop("per_layer_oof")
        report[key] = r
        print(key, json.dumps(r), flush=True)
    (HW1 / "results" / "cv_muq.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
