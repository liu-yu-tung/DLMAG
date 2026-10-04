"""Per class: fine-tuned MERT top 12 (3-seed mean) next to the recipe's zero-shot part, with the probe and fusions.

A: per decade, next to Qwen2-Audio zero-shot. B (--dataset B): per market, next to the Whisper language ID of the
mixture. Two views, both calibrated as in grouped_ft_score.py: train from the grouped 5-fold out-of-fold log-probs
(honest estimate, artist-grouped proxy) and validation from the train fits. Per true class: Top-1, Top-3, S,
probability on the true class, the Top-1 overlap of FT and the zero-shot part, and on A the mean signed decade error of
the top pick (negative = guesses too early). Mean probability matrices (true x predicted) go to the figure.
Writes results/ft_qwen_per_decade.{json,png} (A) or results/ft_lang_per_market.{json,png} (B).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combo_tree import load
from hw1.data import LABELS
from non_mert import norm, per_clip

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
SEEDS = [0, 1, 2]
CFG = {"A": {"zs": "qwen", "zs_comp": "qwen_zs", "probe_comp": "mert_ord10", "unit": "decade", "out": "ft_qwen_per_decade",
             "names": {"qwen": "Qwen zero-shot", "recipe": "probe + Qwen", "recipe_ft": "probe + Qwen + FT"}},
       "B": {"zs": "lang", "zs_comp": "lang_mixture", "probe_comp": "mert_mixture", "unit": "market", "out": "ft_lang_per_market",
             "names": {"lang": "Whisper language", "recipe": "probe + Whisper", "recipe_ft": "probe + Whisper + FT"}}}


def take(x, ids, want):
    p = {s: i for i, s in enumerate(ids)}
    return x[[p[s] for s in want]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    key = ap.parse_args().dataset
    cfg = CFG[key]
    zs = cfg["zs"]
    names = {"ft": "FT top 12", "probe": "MERT probe", **cfg["names"]}
    comp, ytr, yva = load(key)
    ref = np.load(FEAT / f"{key}_mertv2.npz")
    tr_ids, va_ids = ref["sample_id"][ref["split"] == "train"], ref["sample_id"][ref["split"] == "validation"]
    g = np.load(FEAT / f"cvg_{key}.npz")
    assert (g["y_train"] == ytr).all()
    runs_s = [np.load(FEAT / f"ft_{key}_L12_s{s}.npz") for s in SEEDS]
    runs_g = [np.load(FEAT / f"ft_{key}_L12_g_s{s}.npz") for s in SEEDS]
    assert all(int(z[f"done_f{i}"]) for z in runs_g for i in range(5))
    ft_g = np.mean([take(z["oof"], z["oof_id"], tr_ids) for z in runs_g], 0)
    ft_s = np.mean([take(z["oof"], z["oof_id"], tr_ids) for z in runs_s], 0)
    ft_v = np.mean([take(z["val"], z["val_id"], va_ids) for z in runs_s], 0)

    def temp(o):
        o = norm(o)
        return minimize_scalar(lambda t: -norm(o * t)[np.arange(len(ytr)), ytr].mean(), bounds=(0.05, 20), method="bounded").x

    views = {}
    base = {"train (grouped OOF)": {zs: g[f"grouped_{cfg['zs_comp']}"], "ft": ft_g, "probe": g[f"grouped_{cfg['probe_comp']}"]}}
    tv = {zs: comp[cfg["zs_comp"]], "ft": (ft_s, ft_v), "probe": comp[cfg["probe_comp"]]}
    base["validation"] = {c: norm(norm(v) * temp(o)) for c, (o, v) in tv.items()}
    base["train (grouped OOF)"] = {c: norm(norm(o) * temp(o)) for c, o in base["train (grouped OOF)"].items()}
    for view, lp in base.items():
        lp["recipe"] = norm(np.mean([lp["probe"], lp[zs]], 0))
        lp["recipe_ft"] = norm(np.mean([lp["probe"], lp[zs], lp["ft"]], 0))
        views[view] = (lp, ytr if view.startswith("train") else yva)

    labels = LABELS[key]
    out = {}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for row, (view, (lp, y)) in enumerate(views.items()):
        res = {}
        for m, z in lp.items():
            top = np.argsort(-z, 1)[:, :3]
            s = per_clip(z, y)
            res[m] = {"S": round(float(s.mean()), 3), "per_class": {}}
            for k, lab in enumerate(labels):
                i = y == k
                res[m]["per_class"][lab] = {
                    "n": int(i.sum()), "top1": round(float((top[i, 0] == k).mean()), 3),
                    "top3": round(float((top[i] == k).any(1).mean()), 3), "S": round(float(s[i].mean()), 3),
                    "p_true": round(float(np.exp(z[i, k]).mean()), 3)}
                if key == "A":
                    res[m]["per_class"][lab]["bias"] = round(float((top[i, 0] - k).mean()), 2)
        f1, q1 = lp["ft"].argmax(1) == y, lp[zs].argmax(1) == y
        res["overlap"] = {lab: {"both": int((f1 & q1)[y == k].sum()), "ft_only": int((f1 & ~q1)[y == k].sum()),
                                f"{zs}_only": int((~f1 & q1)[y == k].sum()), "neither": int((~f1 & ~q1)[y == k].sum())}
                          for k, lab in enumerate(labels)}
        out[view] = res
        for col, m in enumerate([zs, "ft", "recipe", "recipe_ft"]):
            P = np.array([np.exp(lp[m][y == k]).mean(0) for k in range(len(labels))])
            ax = axes[row, col]
            ax.imshow(P, vmin=0, vmax=0.8, cmap="Blues")
            for a in range(len(labels)):
                for b in range(len(labels)):
                    ax.text(b, a, f"{P[a, b]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if P[a, b] > 0.45 else "black")
            short = [l[:4] if key == "A" else l[:3] for l in labels]
            ax.set_xticks(range(len(labels)), short, fontsize=8)
            ax.set_yticks(range(len(labels)), short, fontsize=8)
            ax.set_title(f"{names[m]}, {view}\nS {res[m]['S']:.3f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"true {cfg['unit']}")
            if row == 1:
                ax.set_xlabel("mean predicted probability")

        print(f"\n== {view} ==")
        print(f"{cfg['unit']:8s} {'n':>4s} | " + " | ".join(f"{names[m]:>22s}" for m in lp)
              + f" |  FT/{zs} top-1: both, FT only, {zs} only, neither")
        print(f"{'':8s} {'':>4s} | " + " | ".join(f"{'T1   T3    S   p_true':>22s}" for _ in lp))
        for k, lab in enumerate(labels):
            d = [res[m]["per_class"][lab] for m in lp]
            o = res["overlap"][lab]
            print(f"{lab:8s} {d[0]['n']:4d} | " + " | ".join(f"{x['top1']:.2f} {x['top3']:.2f} {x['S']:.2f} {x['p_true']:.2f}".rjust(22) for x in d)
                  + f" |  {o['both']:3d} {o['ft_only']:3d} {o[f'{zs}_only']:3d} {o['neither']:3d}")
        print(f"{'all':8s} {len(y):4d} | " + " | ".join(f"S {res[m]['S']:.3f}".rjust(22) for m in lp))
    fig.tight_layout()
    fig.savefig(HW1 / "results" / f"{cfg['out']}.png", dpi=120)
    (HW1 / "results" / f"{cfg['out']}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
