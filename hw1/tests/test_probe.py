import numpy as np
import pytest

from hw1.data import LABELS
from hw1.probe import fit_eval, split_xy


def _fake_features(n_per_class: int = 30, dim: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = LABELS["A"]
    rows, lab, split = [], [], []
    for s in ("train", "validation"):
        for c, name in enumerate(labels):
            rows.append(rng.normal(loc=c * 3.0, scale=0.5, size=(n_per_class, dim)))
            lab += [name] * n_per_class
            split += [s] * n_per_class
    feats = np.concatenate(rows).astype(np.float32)
    ids = np.array([f"A_{i:04d}" for i in range(len(feats))])
    return {"feats": feats, "label": np.array(lab), "split": np.array(split), "sample_id": ids}


@pytest.mark.parametrize("name", ["logreg", "svm_rbf", "rf", "knn"])
def test_separable_data_is_learned(name):
    f = _fake_features()
    Xtr, ytr, _ = split_xy(f, "A", "train")
    Xva, yva, ids = split_xy(f, "A", "validation")
    assert len(ids) == len(yva) == 180
    _, probs, m = fit_eval(name, Xtr, ytr, Xva, yva, LABELS["A"])
    assert probs.shape == (180, 6)
    assert m["top1"] > 0.95 and m["top3"] == 1.0


def test_column_subset():
    f = _fake_features()
    X, _, _ = split_xy(f, "A", "train", cols=np.array([0, 2]))
    assert X.shape == (180, 2)
