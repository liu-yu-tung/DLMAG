"""Stem experiments (validation S): which input carries the label, and does stem information add to the mixture?

Inputs: MERT-v2 all-layer average on the mixture, each Demucs stem and the accompaniment; 29 mix-balance
features. Mixtures of inputs are equal-weight log-probability averages (nothing is tuned on validation).
Also exports train-set class means and ANOVA F of the mix-balance features (H7).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_classif

from hw1 import final
from hw1.data import LABELS
from hw1.metrics import evaluate
from hw1.probe import chunk_probs, fit_eval, fuse_logprobs, layeravg_logprobs, load_features

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
INPUTS = ["mixture", "vocals", "accompaniment", "drums", "bass", "other"]


def _y(f: dict, key: str) -> np.ndarray:
    return np.array([LABELS[key].index(l) if l else -1 for l in f["label"]])


def components(key: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    ref = load_features(FEAT / f"{key}_mertv2.npz")
    split, y = ref["split"], _y(ref, key)
    tr, va = split == "train", split == "validation"
    comp: dict[str, np.ndarray] = {}
    for inp in INPUTS:
        path = FEAT / (f"{key}_mertv2.npz" if inp == "mixture" else f"{key}_mertv2_{inp}.npz")
        if not path.exists():
            print(f"skip {path.name} (missing)")
            continue
        f = load_features(path)
        if not (f["sample_id"] == ref["sample_id"]).all():
            raise ValueError(f"clip order differs in {path.name}")
        X = final.mert_means(f["feats"])
        comp[f"MERT {inp}"] = layeravg_logprobs(X[tr], y[tr], X[va])

    extra: dict = {}
    mb = load_features(FEAT / f"{key}_mixbalance.npz")
    if not (mb["sample_id"] == ref["sample_id"]).all():
        raise ValueError("clip order differs in mix-balance features")
    comp["mix balance"] = np.log(fit_eval("logreg", mb["feats"][tr], y[tr], mb["feats"][va], y[va], LABELS[key])[1] + 1e-9)
    F, p = f_classif(mb["feats"][tr], y[tr])
    names = mb["names"].tolist()
    extra["mixbalance_anova_train"] = sorted(
        [{"feature": n, "F": round(float(a), 2), "p": float(b)} for n, a, b in zip(names, F, p)], key=lambda d: -d["F"])
    extra["mixbalance_class_means_train"] = {
        n: [round(float(mb["feats"][tr & (y == c), i].mean()), 4) for c in range(6)] for i, n in enumerate(names)}

    lid = load_features(FEAT / f"{key}_langid.npz")
    if not (lid["sample_id"] == ref["sample_id"]).all():
        raise ValueError("clip order differs in language-ID features")
    for src in ("vocals", "mixture"):
        comp[f"language ({src})"] = np.log(fit_eval("logreg", lid[src][tr], y[tr], lid[src][va], y[va], LABELS[key])[1] + 1e-9)

    if key == "A":
        ch = load_features(FEAT / "A_handcrafted_chunks.npz")
        comp["hand-crafted 10 s chunks"] = np.log(chunk_probs(ch["c10"], ch["split"], y[tr]) + 1e-9)
    return comp, y[va], extra


def mixtures(key: str, comp: dict) -> dict[str, list[str]]:
    base = ["MERT mixture", "hand-crafted 10 s chunks"] if key == "A" else ["MERT mixture"]
    have = lambda *names: all(n in comp for n in names)
    mixes = {
        "MERT mixture + vocals": ["MERT mixture", "MERT vocals"],
        "MERT mixture + accompaniment": ["MERT mixture", "MERT accompaniment"],
        "MERT vocals + accompaniment": ["MERT vocals", "MERT accompaniment"],
        "MERT mixture + 4 stems": ["MERT mixture", "MERT vocals", "MERT drums", "MERT bass", "MERT other"],
        "MERT mixture + mix balance": ["MERT mixture", "mix balance"],
        "current recipe + mix balance": base + ["mix balance"],
        "current recipe + vocals": base + ["MERT vocals"],
        "current recipe + vocals + mix balance": base + ["MERT vocals", "mix balance"],
        "current recipe + language (vocals)": base + ["language (vocals)"],
        "current recipe + language (mixture)": base + ["language (mixture)"],
        "current recipe + mix balance + language (vocals)": base + ["mix balance", "language (vocals)"],
    }
    if key == "A":
        mixes["current recipe (MERT mixture + hand-crafted 10 s)"] = base
    return {n: parts for n, parts in mixes.items() if have(*parts)}


def plot_inputs(out: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [f"MERT {i}" for i in INPUTS] + ["mix balance"]
    x = np.arange(len(names))
    for j, (key, r) in enumerate(out.items()):
        vals = [r["methods"].get(n, {}).get("S", np.nan) for n in names]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.38, label=f"dataset {key}")
    ax.axhline(0.417, color="gray", ls="--", lw=0.8, label="chance (0.417)")
    ax.set_xticks(x, [n.replace("MERT ", "") for n in names], rotation=20)
    ax.set(ylabel="validation S", title="Which input carries the label? (MERT-v2 all-layer average per input)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trends(out: dict, path: Path, feats: list[str]) -> None:
    means = out["A"]["mixbalance_class_means_train"]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 3), constrained_layout=True)
    for ax, n in zip(np.atleast_1d(axes), feats):
        ax.plot(LABELS["A"], means[n], marker="o")
        ax.set_title(n, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Dataset A train: mix-balance class means by decade", fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    out = {}
    for key in ("A", "B"):
        comp, yva, extra = components(key)
        labels = LABELS[key]
        methods = {n: evaluate(fuse_logprobs([lp]), yva, labels) for n, lp in comp.items()}
        for n, parts in mixtures(key, comp).items():
            methods[n] = evaluate(fuse_logprobs([np.log(fuse_logprobs([comp[c]]) + 1e-9) for c in parts]), yva, labels)
        out[key] = {"methods": {n: {k: (round(v, 4) if k != "cm" else v) for k, v in m.items()} for n, m in methods.items()},
                    **extra}
    plot_inputs(out, HW1 / "results" / "stems_inputs.png")
    top = [d["feature"] for d in out["A"]["mixbalance_anova_train"][:4]]
    plot_trends(out, HW1 / "results" / "stems_A_mixbalance_trends.png", top)
    (HW1 / "results" / "stems.json").write_text(json.dumps(out, indent=2))
    for key, r in out.items():
        print(f"== {key}")
        for n, m in sorted(r["methods"].items(), key=lambda kv: -kv[1]["S"]):
            print(f"  {m['S']:.3f}  top1 {m['top1']:.3f} top3 {m['top3']:.3f}  {n}")
        print("  top mix-balance features (train ANOVA F):",
              ", ".join(f"{d['feature']} {d['F']}" for d in r["mixbalance_anova_train"][:6]))


if __name__ == "__main__":
    main()
