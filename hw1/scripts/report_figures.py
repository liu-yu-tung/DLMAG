"""Validation figures and numbers for the report, from the saved checkpoints and cached features (no refitting).

Writes results/report/: confusion matrices (counts and row-normalized) for the final recipes, each component alone,
and the zero-shot Qwen2-Audio prompts; report_figures.json with top-1/top-3/S, error distances (A) and the most
confused market pairs (B).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_final import FEAT, HW1, cached

from hw1 import final
from hw1.data import LABELS
from hw1.metrics import evaluate, plot_confusion

OUT = HW1 / "results" / "report"


def summary(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    m = evaluate(probs, y, labels)
    return {k: round(m[k], 4) for k in ("top1", "top3", "S")} | {"cm": m["cm"]}


def plot_both(cm: list[list[int]], labels: list[str], stem: str, title: str) -> None:
    plot_confusion(np.array(cm), labels, OUT / f"{stem}_cm.png", title=title)
    plot_confusion(np.array(cm), labels, OUT / f"{stem}_cm_norm.png", normalize=True, title=title)


def decade_distance(cm: list[list[int]]) -> dict:
    """Top-1 errors by |predicted - true| in decades, and the share expected if a wrong prediction were uniform
    over the other five decades (weighted by the true-class counts of the errors)."""
    cm = np.array(cm)
    n = len(cm)
    dist = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    errors = {d: int(cm[dist == d].sum()) for d in range(1, n)}
    wrong_per_true = cm.sum(1) - np.diag(cm)
    chance = {d: float(sum(wrong_per_true[t] * (dist[t] == d).sum() / (n - 1) for t in range(n))) for d in range(1, n)}
    total = sum(errors.values())
    return {"errors": errors, "share": {d: round(e / total, 3) for d, e in errors.items()},
            "uniform_share": {d: round(c / total, 3) for d, c in chance.items()}}


def market_pairs(cm: list[list[int]], labels: list[str], top: int = 6) -> list[dict]:
    cm = np.array(cm)
    pairs = [{"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]]
    return sorted(pairs, key=lambda p: -p["count"])[:top]


def alm_zero_shot(key: str, prompt: str, ids: np.ndarray) -> np.ndarray:
    z = np.load(FEAT / f"{key}_alm_{prompt}_validation.npz")
    rows = dict(zip(z["sample_id"].tolist(), z["logp"]))
    return np.stack([rows[s] for s in ids.tolist()])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for key in ("A", "B"):
        labels = LABELS[key]
        feats, y, split, ids = cached(key)
        val = split == "validation"
        fv, yv = {c: v[val] for c, v in feats.items()}, y[val]
        ckpt = final.load(HW1 / "checkpoints" / f"{key}.joblib")

        recipes = {key: ckpt["recipe"]} | ({f"{key}_fallback": ckpt["fallback"]} if ckpt.get("fallback") else {})
        for name, recipe in recipes.items():
            r = summary(final.predict_proba(ckpt, fv, recipe), yv, labels)
            plot_both(r["cm"], labels, name, f"{key} validation: {' + '.join(recipe)}")
            r["recipe"] = recipe
            if key == "A":
                r["decade_distance"] = decade_distance(r["cm"])
            else:
                r["confused_pairs"] = market_pairs(r["cm"], labels)
            report[name] = r

        for comp in ckpt["recipe"]:
            r = summary(final.predict_proba(ckpt, fv, [comp]), yv, labels)
            plot_both(r["cm"], labels, f"{key}_{comp}", f"{key} validation: {comp} alone")
            report[f"{key}_{comp}"] = r

        for prompt in ("plain", "cues"):
            r = summary(alm_zero_shot(key, prompt, ids[val]), yv, labels)
            plot_both(r["cm"], labels, f"{key}_qwen_{prompt}", f"{key} validation: Qwen2-Audio zero-shot, {prompt} prompt")
            report[f"{key}_qwen_{prompt}"] = r

    (OUT / "report_figures.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, r in report.items():
        print(f"{name:24s} top1 {r['top1']:.3f} top3 {r['top3']:.3f} S {r['S']:.3f}")
    print(json.dumps(report["A"]["decade_distance"]))
    print(json.dumps(report["A_fallback"]["decade_distance"]))
    print(json.dumps(report["B"]["confused_pairs"]))


if __name__ == "__main__":
    main()
