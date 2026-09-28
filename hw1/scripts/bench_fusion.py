"""Learned layer weighting and equal-weight late fusion of different information sources (validation S).

No fusion weight or component choice is fitted on validation: every mixture averages log-probabilities
with equal weights, and the MERT single best layer is reported only as a validation-selected reference.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hw1.data import LABELS
from hw1.layermix import fit_layermix
from hw1.metrics import evaluate, plot_confusion
from hw1.probe import chunk_probs, fit_eval, fuse_logprobs, load_features, split_xy

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
SEEDS = range(5)
D = 1024


def _split(feat: dict, key: str, x: np.ndarray):
    v = {**feat, "feats": x}
    return split_xy(v, key, "train"), split_xy(v, key, "validation")


def components(key: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    labels = LABELS[key]
    comp: dict[str, np.ndarray] = {}
    extra: dict = {}

    hc = load_features(FEAT / f"{key}_handcrafted.npz")
    (Xtr, ytr, _), (Xva, yva, _) = _split(hc, key, hc["feats"])
    comp["hand-crafted 30 s"] = fit_eval("logreg", Xtr, ytr, Xva, yva, labels)[1]
    chunk_file = FEAT / f"{key}_handcrafted_chunks.npz"
    if chunk_file.exists():
        ch = load_features(chunk_file)
        comp["hand-crafted 10 s chunks"] = chunk_probs(ch["c10"], ch["split"], ytr)

    mert = load_features(FEAT / f"{key}_mertv2.npz")
    layers = mert["feats"][:, :, :D].astype(np.float32)
    per_layer = []
    for l in range(layers.shape[1]):
        (Xtr, ytr, _), (Xva, _, _) = _split(mert, key, layers[:, l])
        per_layer.append(np.log(fit_eval("logreg", Xtr, ytr, Xva, yva, labels)[1] + 1e-9))
    comp["MERT-v2 all-layer average"] = fuse_logprobs(per_layer)
    best = int(np.argmax([evaluate(np.exp(p), yva, labels)["S"] for p in per_layer]))
    comp[f"MERT-v2 layer {best + 1} (val-selected, reference)"] = np.exp(per_layer[best])

    tr, va = mert["split"] == "train", mert["split"] == "validation"
    runs, weights = [], []
    for seed in SEEDS:
        fitted = fit_layermix(layers[tr], ytr, seed=seed)
        runs.append(np.log(fitted.predict_proba(layers[va]) + 1e-9))
        weights.append(fitted.layer_weights())
    comp["MERT-v2 learned layer weights"] = fuse_logprobs(runs)
    extra["layer_weights"] = np.mean(weights, axis=0).round(4).tolist()
    extra["layer_weights_seed_std"] = np.std(weights, axis=0).round(4).tolist()

    clap_file = FEAT / f"{key}_clap.npz"
    if clap_file.exists():
        clap = load_features(clap_file)
        (Xtr, ytr, _), (Xva, _, _) = _split(clap, key, clap["emb"])
        comp["CLAP audio embedding"] = fit_eval("logreg", Xtr, ytr, Xva, yva, labels)[1]
    return comp, yva, extra


def mixtures(comp: dict[str, np.ndarray]) -> dict[str, list[str]]:
    hc = "hand-crafted 10 s chunks" if "hand-crafted 10 s chunks" in comp else "hand-crafted 30 s"
    mixes = {}
    for mert in ("MERT-v2 all-layer average", "MERT-v2 learned layer weights"):
        mixes[f"{mert} + {hc}"] = [mert, hc]
        if "CLAP audio embedding" in comp:
            mixes[f"{mert} + CLAP"] = [mert, "CLAP audio embedding"]
            mixes[f"{mert} + CLAP + {hc}"] = [mert, "CLAP audio embedding", hc]
    return mixes


def plot_weights(out: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for key, r in out.items():
        w, s = np.array(r["layer_weights"]), np.array(r["layer_weights_seed_std"])
        x = np.arange(1, len(w) + 1)
        ax.errorbar(x, w, yerr=s, marker="o", ms=3, capsize=2, label=f"dataset {key}")
    ax.axhline(1 / 24, color="gray", ls="--", lw=0.8, label="uniform (1/24)")
    ax.set(xlabel="MERT-v2 layer", ylabel="learned weight", title="Learned layer weights (mean ± std over 5 seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    out = {}
    for key in ("A", "B"):
        comp, yva, extra = components(key)
        methods = {name: evaluate(p, yva, LABELS[key]) for name, p in comp.items()}
        for name, parts in mixtures(comp).items():
            methods[name] = evaluate(fuse_logprobs([np.log(comp[c] + 1e-9) for c in parts]), yva, LABELS[key])
        out[key] = {"methods": {n: {k: (round(v, 4) if k != "cm" else v) for k, v in m.items()} for n, m in methods.items()}, **extra}
        best = max((n for n in methods if "reference" not in n), key=lambda n: methods[n]["S"])
        out[key]["best_non_reference"] = best
        cm = np.array(methods[best]["cm"])
        plot_confusion(cm, LABELS[key], HW1 / "results" / f"fusion_{key}_cm_norm.png", normalize=True, title=f"{key}: {best}")
    plot_weights(out, HW1 / "results" / "mert_layer_weights.png")
    (HW1 / "results" / "fusion.json").write_text(json.dumps(out, indent=2))
    for key, r in out.items():
        print(f"== {key}")
        for n, m in sorted(r["methods"].items(), key=lambda kv: -kv[1]["S"]):
            print(f"  {m['S']:.3f}  top1 {m['top1']:.3f} top3 {m['top3']:.3f}  {n}")
        print("  layer weights:", r["layer_weights"])


if __name__ == "__main__":
    main()
