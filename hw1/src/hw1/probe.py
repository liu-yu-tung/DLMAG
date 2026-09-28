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
