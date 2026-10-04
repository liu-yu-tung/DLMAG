import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw1.data import LABELS


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    y = np.asarray(y)
    order = np.argsort(-probs, axis=1)[:, :k]
    hit = (order == y[:, None]).any(axis=1)
    return float(hit.mean())


def score_s(probs: np.ndarray, y: np.ndarray) -> float:
    top1 = topk_accuracy(probs, y, 1)
    top3 = topk_accuracy(probs, y, 3)
    return top1 + 0.5 * top3


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_confusion(
    cm: np.ndarray,
    labels: list[str],
    path: str | Path,
    normalize: bool = False,
    title: str = "",
) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        data = cm / row_sums
        fmt = "{:.2f}"
        kind = "row-normalized"
    else:
        data = cm
        fmt = "{:.0f}"
        kind = "counts"

    full_title = f"{title}\n({kind})" if title else kind

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(full_title, fontsize=10)

    thresh = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(data[i, j]),
                ha="center",
                va="center",
                color="white" if data[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    y_pred = np.argmax(probs, axis=1)
    cm = confusion(np.asarray(y), y_pred, n_classes=len(labels))
    return {
        "top1": topk_accuracy(probs, y, 1),
        "top3": topk_accuracy(probs, y, 3),
        "S": score_s(probs, y),
        "cm": cm.tolist(),
    }


def top3_labels(probs: np.ndarray, labels: list[str]) -> list[list[str]]:
    order = np.argsort(-probs, axis=1)[:, :3]
    return [[labels[i] for i in row] for row in order]


def write_predictions(path: str | Path, preds_by_dataset: dict[str, dict[str, list[str]]]) -> None:
    with open(path, "w") as f:
        json.dump(preds_by_dataset, f, indent=2)


def validate_predictions(obj: dict, manifests: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    for dataset_name, manifest in manifests.items():
        key = dataset_name.split("_")[-1]
        valid_labels = set(LABELS[key])
        test_ids = set(manifest.loc[manifest["split"] == "test", "sample_id"])

        preds = obj.get(dataset_name, {})
        pred_ids = list(preds.keys())

        missing = test_ids - set(pred_ids)
        extra = set(pred_ids) - test_ids
        if missing:
            errors.append(f"{dataset_name}: missing {len(missing)} sample_id(s)")
        if extra:
            errors.append(f"{dataset_name}: {len(extra)} unexpected sample_id(s)")

        seen = set()
        for sid, top3 in preds.items():
            if sid in seen:
                errors.append(f"{dataset_name}: duplicate sample_id {sid}")
            seen.add(sid)
            if len(top3) != 3 or len(set(top3)) != 3:
                errors.append(f"{dataset_name}: {sid} does not have 3 distinct labels")
            bad = [lab for lab in top3 if lab not in valid_labels]
            if bad:
                errors.append(f"{dataset_name}: {sid} has invalid label(s) {bad}")

    return errors
