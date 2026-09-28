"""Benchmark and analyze hand-crafted features on validation. Writes results/handcrafted.json and plots."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_classif

from hw1.data import LABELS
from hw1.features.handcrafted import feature_group
from hw1.metrics import plot_confusion
from hw1.probe import fit_eval, load_features, split_xy

HW1 = Path(__file__).resolve().parents[1]
MODELS = ["logreg", "svm_rbf", "rf", "knn"]
TREND_FEATURES = [
    "energy_dynamic_range_db",
    "energy_rms_db_mean",
    "energy_band_0_60hz_ratio",
    "energy_band_60_150hz_ratio",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "rhythm_tempo_bpm",
    "rhythm_beat_interval_cv",
]


def _round(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def error_structure(key: str, y: np.ndarray, pred: np.ndarray) -> dict:
    wrong = pred != y
    out = {"n_errors": int(wrong.sum())}
    if key == "A":
        dist = np.abs(pred - y)[wrong]
        out["errors_within_1_decade"] = round(float((dist == 1).mean()), 4) if wrong.any() else None
        out["mean_decade_distance_of_errors"] = round(float(dist.mean()), 4) if wrong.any() else None
    else:
        us, uk = LABELS["B"].index("US"), LABELS["B"].index("UK")
        us_uk = ((y == us) & (pred == uk)) | ((y == uk) & (pred == us))
        out["errors_us_uk_share"] = round(float(us_uk[wrong].mean()), 4) if wrong.any() else None
    return out


def trend_plot(key: str, X: np.ndarray, y: np.ndarray, names: list[str], path: Path) -> dict:
    labels = LABELS[key]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    trends = {}
    for ax, feat in zip(axes.flat, TREND_FEATURES):
        col = X[:, names.index(feat)]
        means = [col[y == c].mean() for c in range(6)]
        sems = [col[y == c].std() / np.sqrt((y == c).sum()) for c in range(6)]
        trends[feat] = [round(float(m), 5) for m in means]
        ax.errorbar(range(6), means, yerr=sems, marker="o", capsize=3)
        ax.set_xticks(range(6), labels, rotation=45)
        ax.set_title(feat, fontsize=9)
    fig.suptitle(f"Dataset {key}: train-set class means (± s.e.m.)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return trends


def run(key: str) -> dict:
    feat = load_features(HW1 / "features" / f"{key}_handcrafted.npz")
    names = feat["names"].tolist()
    labels = LABELS[key]
    Xtr, ytr, _ = split_xy(feat, key, "train")
    Xva, yva, _ = split_xy(feat, key, "validation")
    res: dict = {"n_features": len(names), "n_train": len(ytr), "n_val": len(yva)}

    res["models"] = {}
    best = None
    for name in MODELS:
        _, probs, m = fit_eval(name, Xtr, ytr, Xva, yva, labels)
        res["models"][name] = _round({k: v for k, v in m.items() if k != "cm"})
        if best is None or m["S"] > best[2]["S"]:
            best = (name, probs, m)

    res["groups"] = {}
    for g in sorted({feature_group(n) for n in names}):
        cols = np.array([i for i, n in enumerate(names) if feature_group(n) == g])
        _, _, m = fit_eval("logreg", Xtr[:, cols], ytr, Xva[:, cols], yva, labels)
        res["groups"][g] = _round({"n": len(cols), **{k: v for k, v in m.items() if k != "cm"}})

    name, probs, m = best
    pred = probs.argmax(1)
    res["best"] = {"model": name, **_round({k: v for k, v in m.items() if k != "cm"}), "cm": m["cm"]}
    res["best"]["errors"] = error_structure(key, yva, pred)
    cm = np.array(m["cm"])
    plot_confusion(cm, labels, HW1 / "results" / f"handcrafted_{key}_cm.png", title=f"{key} hand-crafted {name}, validation")
    plot_confusion(cm, labels, HW1 / "results" / f"handcrafted_{key}_cm_norm.png", normalize=True, title=f"{key} hand-crafted {name}, validation")

    F, p = f_classif(Xtr, ytr)
    top = np.argsort(-F)[:12]
    res["anova_top_train"] = [{"feature": names[i], "F": round(float(F[i]), 1), "p": float(f"{p[i]:.2e}")} for i in top]
    res["trends_train"] = trend_plot(key, Xtr, ytr, names, HW1 / "results" / f"handcrafted_{key}_trends.png")
    return res


def main() -> None:
    (HW1 / "results").mkdir(exist_ok=True)
    out = {key: run(key) for key in ("A", "B")}
    path = HW1 / "results" / "handcrafted.json"
    path.write_text(json.dumps(out, indent=2))
    for key, r in out.items():
        print(f"== {key}")
        for n, m in r["models"].items():
            print(f"  {n:8s} top1 {m['top1']:.3f} top3 {m['top3']:.3f} S {m['S']:.3f}")
        for g, m in r["groups"].items():
            print(f"  group {g:9s} ({m['n']:2d}) top1 {m['top1']:.3f} top3 {m['top3']:.3f} S {m['S']:.3f}")
        print(f"  best {r['best']['model']} errors {r['best']['errors']}")
    print(path)


if __name__ == "__main__":
    main()
