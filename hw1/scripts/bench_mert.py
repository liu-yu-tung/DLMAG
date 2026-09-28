"""Layer sweep of frozen MERT-v2 features with a logistic-regression probe (validation S)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hw1.data import LABELS
from hw1.metrics import plot_confusion
from hw1.probe import fit_eval, load_features, split_xy

HW1 = Path(__file__).resolve().parents[1]
TAG = "mertv2"
D = 1024


def layer_view(feat: dict, layer: int | None, part: str) -> dict:
    x = feat["feats"].astype(np.float32)
    x = x.mean(axis=1) if layer is None else x[:, layer]
    x = x[:, :D] if part == "mean" else x
    return {**feat, "feats": x}


def probe(feat: dict, key: str, layer: int | None, part: str):
    v = layer_view(feat, layer, part)
    Xtr, ytr, _ = split_xy(v, key, "train")
    Xva, yva, _ = split_xy(v, key, "validation")
    return fit_eval("logreg", Xtr, ytr, Xva, yva, LABELS[key])


def run(key: str) -> dict:
    feat = load_features(HW1 / "features" / f"{key}_{TAG}.npz")
    n_layers = feat["feats"].shape[1]
    sweep = {}
    for layer in range(n_layers):
        m = probe(feat, key, layer, "mean")[2]
        sweep[layer + 1] = {k: round(m[k], 4) for k in ("top1", "top3", "S")}
    best_layer = max(sweep, key=lambda l: sweep[l]["S"])
    res = {"sweep_mean": sweep, "best_layer": best_layer}
    for name, layer, part in [
        ("best layer mean+std", best_layer - 1, "mean+std"),
        ("layer-average mean", None, "mean"),
    ]:
        m = probe(feat, key, layer, part)[2]
        res[name] = {k: round(m[k], 4) for k in ("top1", "top3", "S")}
    _, probs, m = probe(feat, key, best_layer - 1, "mean")
    res["best_cm"] = m["cm"]
    title = f"{key} MERT-v2 L{best_layer} logreg, validation"
    plot_confusion(np.array(m["cm"]), LABELS[key], HW1 / "results" / f"mert_{key}_cm.png", title=title)
    plot_confusion(np.array(m["cm"]), LABELS[key], HW1 / "results" / f"mert_{key}_cm_norm.png", normalize=True, title=title)
    np.save(HW1 / "features" / f"{key}_{TAG}_val_probs.npy", probs)
    return res


def plot_sweep(out: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, r in out.items():
        layers = [int(l) for l in r["sweep_mean"]]
        ax.plot(layers, [r["sweep_mean"][l]["S"] for l in r["sweep_mean"]], marker="o", ms=3, label=f"dataset {key}")
    ax.axhline(0.417, color="gray", ls="--", lw=0.8, label="chance S")
    ax.set(xlabel="MERT-v2 layer", ylabel="validation S", title="Frozen MERT-v2 layer sweep (logreg on time-mean)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    out = {key: run(key) for key in ("A", "B")}
    plot_sweep(out, HW1 / "results" / "mert_layer_sweep.png")
    (HW1 / "results" / "mert.json").write_text(json.dumps(out, indent=2))
    for key, r in out.items():
        print(f"== {key} best layer {r['best_layer']}: {r['sweep_mean'][r['best_layer']]}")
        print("  S by layer:", " ".join(f"{l}:{v['S']:.3f}" for l, v in r["sweep_mean"].items()))
        for name in ("best layer mean+std", "layer-average mean"):
            print(f"  {name}: {r[name]}")


if __name__ == "__main__":
    main()
