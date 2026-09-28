"""Chunk-length and multi-scale experiments on hand-crafted features (validation S).

E1: classifier per chunk length, trained on train chunks, clip score = mean of chunk log-probs.
E2: one vector per clip combining scales: 30 s features + mean/std across shorter chunks.
E3: E1 per feature group, to see which features need which length.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hw1.data import LABELS
from hw1.features.handcrafted import feature_group
from hw1.metrics import evaluate
from hw1.probe import chunk_probs, load_features, make_model

HW1 = Path(__file__).resolve().parents[1]
LENGTHS = ["c5", "c10", "c15", "c30"]


def _y(feat: dict, key: str, split: str) -> np.ndarray:
    return np.array([LABELS[key].index(l) for l in feat["label"][feat["split"] == split]])


def chunk_arrays(key: str) -> tuple[dict, list[str]]:
    whole = load_features(HW1 / "features" / f"{key}_handcrafted.npz")
    chunks = load_features(HW1 / "features" / f"{key}_handcrafted_chunks.npz")
    if not (whole["sample_id"] == chunks["sample_id"]).all():
        raise ValueError("clip order differs between whole-clip and chunk features")
    arr = {c: chunks[c] for c in LENGTHS[:-1]}
    arr["c30"] = whole["feats"][:, None, :]
    for c, a in arr.items():
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite values in {c}")
    return {"arr": arr, "split": whole["split"], "label": whole["label"]}, whole["names"].tolist()


def chunk_classifier(data: dict, key: str, length: str, cols: np.ndarray, model: str = "logreg") -> dict:
    probs = chunk_probs(data["arr"][length][:, :, cols], data["split"], _y(data, key, "train"), model)
    return evaluate(probs, _y(data, key, "validation"), LABELS[key])


def multiscale(data: dict, key: str, scales: list[str], model: str = "logreg") -> dict:
    parts = [data["arr"]["c30"][:, 0]]
    for c in scales:
        parts += [data["arr"][c].mean(axis=1), data["arr"][c].std(axis=1)]
    X = np.concatenate(parts, axis=1)
    tr, va = data["split"] == "train", data["split"] == "validation"
    clf = make_model(model).fit(X[tr], _y(data, key, "train"))
    return evaluate(clf.predict_proba(X[va]), _y(data, key, "validation"), LABELS[key])


def run(key: str) -> dict:
    data, names = chunk_arrays(key)
    groups = np.array([feature_group(n) for n in names])
    all_cols = np.arange(len(names))
    s = lambda m: {k: round(m[k], 4) for k in ("top1", "top3", "S")}

    res: dict = {"E1_chunk_length": {c: s(chunk_classifier(data, key, c, all_cols)) for c in LENGTHS}}
    res["E2_multiscale"] = {
        "30s only": s(multiscale(data, key, [])),
        "30s + 10s stats": s(multiscale(data, key, ["c10"])),
        "30s + 5s stats": s(multiscale(data, key, ["c5"])),
        "30s + 10s + 5s stats": s(multiscale(data, key, ["c10", "c5"])),
        "30s + 15s + 10s + 5s stats": s(multiscale(data, key, ["c15", "c10", "c5"])),
    }
    res["E3_group_by_length"] = {
        g: {c: round(chunk_classifier(data, key, c, np.where(groups == g)[0])["S"], 4) for c in LENGTHS}
        for g in sorted(set(groups))
    }
    return res


def heatmap(key: str, table: dict, path: Path) -> None:
    groups = list(table)
    M = np.array([[table[g][c] for c in LENGTHS] for g in groups])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(LENGTHS)), [c[1:] + " s" for c in LENGTHS])
    ax.set_yticks(range(len(groups)), groups)
    for i in range(len(groups)):
        for j in range(len(LENGTHS)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white" if M[i, j] < M.mean() else "black")
    ax.set_xlabel("chunk length (clip score = mean over chunks)")
    ax.set_title(f"Dataset {key}: validation S by feature group and chunk length", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    out = {}
    for key in ("A",):
        out[key] = run(key)
        heatmap(key, out[key]["E3_group_by_length"], HW1 / "results" / f"chunks_{key}_group_length.png")
    (HW1 / "results" / "chunks.json").write_text(json.dumps(out, indent=2))
    for key, r in out.items():
        print(f"== {key}")
        for section in ("E1_chunk_length", "E2_multiscale"):
            print(f"  {section}")
            for name, m in r[section].items():
                print(f"    {name:28s} top1 {m['top1']:.3f} top3 {m['top3']:.3f} S {m['S']:.3f}")
        print("  E3_group_by_length (S)")
        for g, row in r["E3_group_by_length"].items():
            print(f"    {g:9s} " + "  ".join(f"{c}:{v:.3f}" for c, v in row.items()))


if __name__ == "__main__":
    main()
