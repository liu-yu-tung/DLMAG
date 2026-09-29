"""Hybrid Demucs encoder features in the cross-validated comparison (same 5 train folds as scripts/cv_fusion.py).

Components: each single encoder layer (per-layer logistic probe), the all-layer average of per-layer probes, and
fusions of the all-layer average with the current recipes (equal-weight log-prob mean). Paired bootstrap CI against
the current recipe. Writes results/cv_demucs.json and features/cv_{key}_demucs.npz.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cv_fusion import logp
from hw1.data import LABELS
from hw1.probe import load_features

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
CURRENT = {"A": ["mert_mixture"], "B": ["mert_mixture", "lang_mixture"]}


def per_clip(lp, y):
    o = np.argsort(-lp, 1)[:, :3]
    return (o[:, 0] == y) + 0.5 * (o == y[:, None]).any(1)


def main() -> None:
    report = {}
    rng = np.random.default_rng(0)
    for key in ("A", "B"):
        f = load_features(FEAT / f"{key}_demucs_latent.npz")
        ref = load_features(FEAT / f"{key}_mertv2.npz")
        assert (f["sample_id"] == ref["sample_id"]).all()
        y = np.array([LABELS[key].index(l) if l else -1 for l in f["label"]])
        tr, va = np.where(f["split"] == "train")[0], np.where(f["split"] == "validation")[0]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]))
        cv = dict(np.load(FEAT / f"cv_{key}.npz"))
        ytr, yva = cv["y_train"], cv["y_val"]
        assert (ytr == y[tr]).all()
        save, res = {}, {}
        for layer in f["layers"]:
            X = f[str(layer)]
            oof = np.zeros((len(tr), 6))
            for a, b in folds:
                oof[b] = logp("flat", X[tr[a]], y[tr[a]], X[tr[b]])
            save[f"oof_{layer}"], save[f"val_{layer}"] = oof, logp("flat", X[tr], y[tr], X[va])
            res[str(layer)] = (per_clip(oof, ytr).mean(), per_clip(save[f"val_{layer}"], yva).mean())
            print(key, layer, [round(v, 3) for v in res[str(layer)]], flush=True)
        layers = [str(l) for l in f["layers"]]
        save["oof_demucs_all"] = np.mean([save[f"oof_{l}"] for l in layers], 0)
        save["val_demucs_all"] = np.mean([save[f"val_{l}"] for l in layers], 0)
        np.savez(FEAT / f"cv_{key}_demucs.npz", **save)

        cur = lambda pre: np.mean([cv[f"{pre}_{c}"] for c in CURRENT[key]], 0)
        cur_clip = per_clip(cur("oof"), ytr)
        out = {"current": {"components": "+".join(CURRENT[key]), "S_oof": float(cur_clip.mean()),
                           "S_val": float(per_clip(cur("val"), yva).mean())},
               "single_layers": {l: {"S_oof": round(float(a), 4), "S_val": round(float(b), 4)} for l, (a, b) in res.items()}}
        best = max(layers, key=lambda l: res[l][0])
        for name, comp in [("demucs_all", "demucs_all"), (f"best layer {best} (OOF-selected)", best)]:
            s_oof, s_val = per_clip(save[f"oof_{comp}"], ytr), per_clip(save[f"val_{comp}"], yva)
            f_oof = per_clip(np.mean([cv[f"oof_{c}"] for c in CURRENT[key]] + [save[f"oof_{comp}"]], 0), ytr)
            f_val = per_clip(np.mean([cv[f"val_{c}"] for c in CURRENT[key]] + [save[f"val_{comp}"]], 0), yva)
            d = f_oof - cur_clip
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            out[name] = {"alone": {"S_oof": round(float(s_oof.mean()), 4), "S_val": round(float(s_val.mean()), 4)},
                         "current_plus": {"S_oof": round(float(f_oof.mean()), 4), "S_val": round(float(f_val.mean()), 4),
                                          "diff_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}}
        report[key] = out
        print(key, json.dumps({k: v for k, v in out.items() if k != "single_layers"}), flush=True)
    (HW1 / "results" / "cv_demucs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
