"""Classifiers on cached clip-level features: train on train, score on validation."""

from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hw1.data import LABELS
from hw1.metrics import evaluate

SEED = 0


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_xy(feat: dict[str, np.ndarray], key: str, split: str, cols: np.ndarray | None = None):
    mask = feat["split"] == split
    X = feat["feats"][mask].astype(np.float32)
    if cols is not None:
        X = X[:, cols]
    X = X.reshape(len(X), -1)
    y = np.array([LABELS[key].index(l) if l else -1 for l in feat["label"][mask]])
    return X, y, feat["sample_id"][mask]


def make_model(name: str) -> Pipeline:
    clf = {
        "logreg": lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=SEED),
        "svm_rbf": lambda: CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=SEED), ensemble=False),
        "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }[name]()
    return make_pipeline(StandardScaler(), clf)


def fit_eval(name: str, Xtr, ytr, Xva, yva, labels: list[str]) -> tuple[Pipeline, np.ndarray, dict]:
    model = make_model(name).fit(Xtr, ytr)
    probs = model.predict_proba(Xva)
    return model, probs, evaluate(probs, yva, labels)


def chunk_probs(chunks: np.ndarray, split: np.ndarray, y_train: np.ndarray, model: str = "logreg") -> np.ndarray:
    """chunks (N, n_chunks, F). Fit on train chunks (each inherits its clip label); return validation clip
    probabilities as the renormalized geometric mean of chunk probabilities."""
    tr, va = split == "train", split == "validation"
    n = chunks.shape[1]
    clf = make_model(model).fit(chunks[tr].reshape(-1, chunks.shape[2]), np.repeat(y_train, n))
    logp = np.log(clf.predict_proba(chunks[va].reshape(-1, chunks.shape[2])) + 1e-9).reshape(va.sum(), n, -1)
    return fuse_logprobs([logp.mean(axis=1)])


def fuse_logprobs(logps: list[np.ndarray]) -> np.ndarray:
    """Equal-weight late fusion: mean of log-probabilities, renormalized to probabilities."""
    z = np.mean(logps, axis=0)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def layeravg_logprobs(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, model: str = "logreg") -> np.ndarray:
    """One probe per layer on (N, L, D) features; returns the mean of per-layer validation log-probabilities."""
    return np.mean([np.log(make_model(model).fit(Xtr[:, l], ytr).predict_proba(Xva[:, l]) + 1e-9)
                    for l in range(Xtr.shape[1])], axis=0)
