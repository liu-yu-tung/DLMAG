"""Cross-validated fusion selection.

Step 1 (cached): for every component (one probe per input) get out-of-fold log-probs on train (5-fold) and
validation log-probs from a train-only fit -> features/cv_{A,B}.npz. Step 2: score every fusion of up to
--max-size components (equal-weight log-prob mean) on the out-of-fold set (n=1026 for A, 798 for B), then apply
the rule "fewest components within --tol of the best out-of-fold S". Validation is reported, never used to choose.
A logistic-regression stacker on the out-of-fold log-probs is scored the same way for reference.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from hw1 import final
from hw1.data import LABELS
from hw1.probe import load_features, make_model

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
EPS = 1e-9


def _norm(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _fit_layer(Xtr, ytr, Xte):
    return np.log(make_model("logreg").fit(Xtr, ytr).predict_proba(Xte) + EPS)


def logp(kind: str, Xtr, ytr, Xte) -> np.ndarray:
    if kind == "layers":
        z = np.mean(Parallel(n_jobs=6)(delayed(_fit_layer)(Xtr[:, l], ytr, Xte[:, l]) for l in range(Xtr.shape[1])), axis=0)
    elif kind == "flat":
        z = _fit_layer(Xtr, ytr, Xte)
    else:
        n = Xtr.shape[1]
        clf = make_model("logreg").fit(Xtr.reshape(-1, Xtr.shape[2]), np.repeat(ytr, n))
        z = np.log(clf.predict_proba(Xte.reshape(-1, Xte.shape[2])) + EPS).reshape(len(Xte), Xte.shape[1], -1).mean(1)
    return _norm(z)


def specs(key: str) -> tuple[dict[str, tuple[str, np.ndarray]], np.ndarray, np.ndarray]:
    ref = load_features(FEAT / f"{key}_mertv2.npz")
    y = np.array([LABELS[key].index(l) if l else -1 for l in ref["label"]])
    out = {"mert_mixture": ("layers", final.mert_means(ref["feats"]))}
    for stem in ["vocals", "accompaniment", "drums", "bass", "other"]:
        f = load_features(FEAT / f"{key}_mertv2_{stem}.npz")
        assert (f["sample_id"] == ref["sample_id"]).all()
        out[f"mert_{stem}"] = ("layers", final.mert_means(f["feats"]))
    if key == "A":
        out["handcrafted"] = ("chunks", load_features(FEAT / "A_handcrafted_chunks.npz")["c10"])
    else:
        out["handcrafted"] = ("flat", load_features(FEAT / "B_handcrafted.npz")["feats"])
    out["clap"] = ("flat", load_features(FEAT / f"{key}_clap.npz")["emb"])
    out["mixbalance"] = ("flat", load_features(FEAT / f"{key}_mixbalance.npz")["feats"])
    lid = load_features(FEAT / f"{key}_langid.npz")
    out["lang_vocals"], out["lang_mixture"] = ("flat", lid["vocals"]), ("flat", lid["mixture"])
    for _, X in out.values():
        assert len(X) == len(ref["sample_id"])
    return out, y, ref["split"]


def compute(key: str, path: Path) -> dict[str, np.ndarray]:
    comps, y, split = specs(key)
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]))
    out: dict[str, np.ndarray] = {"y_train": y[tr], "y_val": y[va]}
    for name, (kind, X) in comps.items():
        oof = np.zeros((len(tr), len(LABELS[key])))
        for a, b in folds:
            oof[b] = logp(kind, X[tr[a]], y[tr[a]], X[tr[b]])
        out[f"oof_{name}"] = oof
        out[f"val_{name}"] = logp(kind, X[tr], y[tr], X[va])
        print(key, name, flush=True)
    np.savez(path, **out)
    return out


def per_clip(lp: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-lp, 1)[:, :3]
    return (order[:, 0] == y) + 0.5 * (order == y[:, None]).any(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    report = {}
    for key in ("A", "B"):
        path = FEAT / f"cv_{key}.npz"
        z = dict(np.load(path)) if path.exists() and not args.recompute else compute(key, path)
        names = [k[4:] for k in z if k.startswith("oof_")]
        ytr, yva = z["y_train"], z["y_val"]
        fuse = lambda pre, combo, sl=slice(None): np.mean([z[f"{pre}_{c}"] for c in combo], axis=0)

        rows = []
        for size in range(1, args.max_size + 1):
            for combo in itertools.combinations(names, size):
                s_oof, s_va = per_clip(fuse("oof", combo), ytr), per_clip(fuse("val", combo), yva)
                rows.append({"components": "+".join(combo), "n": size, "S_oof": s_oof.mean(), "S_val": s_va.mean()})
        df = pd.DataFrame(rows).sort_values("S_oof", ascending=False).reset_index(drop=True)
        best = df.S_oof.iloc[0]
        chosen = df[df.S_oof >= best - args.tol].sort_values(["n", "S_oof"], ascending=[True, False]).iloc[0]

        current = "+".join(["mert_mixture", "handcrafted"] if key == "A" else ["mert_mixture"])
        cur_clip = per_clip(fuse("oof", current.split("+")), ytr)
        ch_clip = per_clip(fuse("oof", chosen.components.split("+")), ytr)
        rng = np.random.default_rng(0)
        d = ch_clip - cur_clip
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]

        stack_X = lambda pre: np.concatenate([z[f"{pre}_{c}"] for c in names], axis=1)
        stk = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, random_state=0))
        p_oof = cross_val_predict(stk, stack_X("oof"), ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_log_proba")
        stack = {"S_oof": float(per_clip(p_oof, ytr).mean()),
                 "S_val": float(per_clip(stk.fit(stack_X("oof"), ytr).predict_log_proba(stack_X("val")), yva).mean())}

        singles = {r.components: {"S_oof": round(r.S_oof, 4), "S_val": round(r.S_val, 4)} for r in df[df.n == 1].itertuples()}
        report[key] = {
            "n_train": int(len(ytr)), "components": names, "best": df.iloc[0].to_dict(),
            "chosen_by_rule": chosen.to_dict(), "current": {"components": current, "S_oof": float(cur_clip.mean()),
                                                              "S_val": float(per_clip(fuse("val", current.split("+")), yva).mean())},
            "chosen_minus_current_oof": {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
            "stacker_all_components": stack, "singles": singles,
        }
        df.to_csv(HW1 / "results" / "tables" / f"cv_fusion_{key}.csv", index=False, float_format="%.4f")
        print(f"\n== {key}: {len(ytr)} train clips, chance S 0.417")
        print(df.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("singles:", singles)
        print("current:", report[key]["current"])
        print("chosen by rule:", dict(chosen), "| minus current (OOF):", report[key]["chosen_minus_current_oof"])
        print("stacker:", stack)
    (HW1 / "results" / "cv_fusion.json").write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
