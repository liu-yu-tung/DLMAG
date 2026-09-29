"""Submitted models: fit on train features, save one checkpoint per dataset, predict from features.

A recipe is a list of components; the clip score is the equal-weight mean of component log-probabilities
(no weight is tuned on validation). Components:
  mert_layeravg  one logistic regression per MERT-v2 layer on the time-mean embedding, log-probs averaged
  hc_c10         logistic regression on hand-crafted features of 10 s chunks, chunk log-probs averaged
  lang_mixture   logistic regression on Whisper language log-probs of the mixture (100 languages)
"""

from pathlib import Path

import joblib
import numpy as np

from hw1.data import LABELS
from hw1.probe import fuse_logprobs, make_model

RECIPES = {"A": ["mert_layeravg"], "B": ["mert_layeravg", "lang_mixture"]}
MERT_DIM = 1024
HC_CHUNK_S = 10
EPS = 1e-9


def mert_means(pooled: np.ndarray) -> np.ndarray:
    """(N, L, 2*D) mean+std pooled MERT features -> (N, L, D) time-means, rounded through fp16 as cached."""
    return pooled[:, :, :MERT_DIM].astype(np.float16).astype(np.float32)


def fit(key: str, feats: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """feats: component name -> train features. mert_layeravg (N, L, D); hc_c10 (N, n_chunks, F)."""
    models: dict[str, object] = {}
    for comp in RECIPES[key]:
        X = feats[comp]
        if comp == "mert_layeravg":
            models[comp] = [make_model("logreg").fit(X[:, l], y) for l in range(X.shape[1])]
        elif comp == "hc_c10":
            models[comp] = make_model("logreg").fit(X.reshape(-1, X.shape[2]), np.repeat(y, X.shape[1]))
        elif comp == "lang_mixture":
            models[comp] = make_model("logreg").fit(X, y)
        else:
            raise ValueError(f"unknown component {comp}")
    return {"key": key, "labels": LABELS[key], "recipe": RECIPES[key], "models": models}


def component_logprobs(ckpt: dict, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for comp in ckpt["recipe"]:
        X, m = feats[comp], ckpt["models"][comp]
        if comp == "mert_layeravg":
            out[comp] = np.mean([np.log(m[l].predict_proba(X[:, l]) + EPS) for l in range(X.shape[1])], axis=0)
        elif comp == "hc_c10":
            n, k, f = X.shape
            out[comp] = np.log(m.predict_proba(X.reshape(-1, f)) + EPS).reshape(n, k, -1).mean(axis=1)
        elif comp == "lang_mixture":
            out[comp] = np.log(m.predict_proba(X) + EPS)
    return out


def predict_proba(ckpt: dict, feats: dict[str, np.ndarray]) -> np.ndarray:
    comps = component_logprobs(ckpt, feats)
    return fuse_logprobs([np.log(fuse_logprobs([comps[c]]) + EPS) for c in ckpt["recipe"]])


def save(ckpt: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ckpt, path, compress=3)


def load(path: str | Path) -> dict:
    return joblib.load(path)
