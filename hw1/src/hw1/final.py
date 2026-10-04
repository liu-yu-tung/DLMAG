"""Submitted models: fit on train features, save one checkpoint per dataset, predict from features.

A recipe is a list of components; the clip score is the mean of component log-probabilities, each first scaled by a
temperature fitted on 5-fold out-of-fold log-probs of the train split (calibrated equal weight; no weight is tuned
on validation). Recipes without calibration use temperature 1. Components:
  mert_layeravg  one logistic regression per MERT-v2 layer on the time-mean embedding, log-probs averaged
  mert_ord10     the same, trained with ordinal soft labels (eps 0.1 on each neighbouring decade, via sample weights)
  hc_c10         logistic regression on hand-crafted features of 10 s chunks, chunk log-probs averaged
  lang_mixture   logistic regression on Whisper language log-probs of the mixture (100 languages)
  alm_qwen       zero-shot Qwen2-Audio label log-likelihoods minus their train-split mean per label
  mert_ft12      MERT-v2 blocks 13-24 fine-tuned with a linear head (scripts/finetune_mert.py --from-layer 12), log-probs
                 averaged over FT_SEEDS; nothing is fitted here. Train features are the out-of-fold log-probs of the same
                 5 stratified folds (used only for the temperature); predict.py runs the saved checkpoints.
FALLBACK gives the recipe used when a component cannot run (predict.py --no-alm).
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from hw1.data import LABELS
from hw1.probe import make_model

RECIPES = {"A": ["mert_ord10", "alm_qwen", "mert_ft12"], "B": ["mert_layeravg", "lang_mixture", "mert_ft12"]}
CALIBRATE = {"A": True, "B": True}
FALLBACK = {"A": ["mert_ord10", "mert_ft12"]}
FT_LAYER = 12
FT_SEEDS = [0, 1, 2]
MERT_DIM = 1024
HC_CHUNK_S = 10
ORD_EPS = 0.1
EPS = 1e-9


def mert_means(pooled: np.ndarray) -> np.ndarray:
    """(N, L, 2*D) mean+std pooled MERT features -> (N, L, D) time-means, rounded through fp16 as cached."""
    return pooled[:, :, :MERT_DIM].astype(np.float16).astype(np.float32)


def ordinal_expand(X: np.ndarray, y: np.ndarray, eps: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-label cross-entropy as weighted hard labels: each clip once with its class (weight 1 - eps * n_nb) and
    once per neighbouring class (weight eps)."""
    Xs, ys, ws = [X], [y], [np.where((y == 0) | (y == k - 1), 1 - eps, 1 - 2 * eps)]
    for d in (-1, 1):
        m = (y + d >= 0) & (y + d < k)
        Xs.append(X[m])
        ys.append(y[m] + d)
        ws.append(np.full(m.sum(), eps))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def _norm(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _fit_component(comp: str, X: np.ndarray, y: np.ndarray, k: int):
    if comp == "mert_layeravg":
        return [make_model("logreg").fit(X[:, l], y) for l in range(X.shape[1])]
    if comp == "mert_ord10":
        out = []
        for l in range(X.shape[1]):
            Xe, ye, we = ordinal_expand(X[:, l], y, ORD_EPS, k)
            out.append(make_model("logreg").fit(Xe, ye, logisticregression__sample_weight=we))
        return out
    if comp == "hc_c10":
        return make_model("logreg").fit(X.reshape(-1, X.shape[2]), np.repeat(y, X.shape[1]))
    if comp == "lang_mixture":
        return make_model("logreg").fit(X, y)
    if comp == "alm_qwen":
        return X.mean(axis=0)
    if comp == "mert_ft12":
        return None
    raise ValueError(f"unknown component {comp}")


def _component_logprobs(comp: str, m, X: np.ndarray) -> np.ndarray:
    if comp in ("mert_layeravg", "mert_ord10"):
        return np.mean([np.log(m[l].predict_proba(X[:, l]) + EPS) for l in range(X.shape[1])], axis=0)
    if comp == "hc_c10":
        n, k, f = X.shape
        return np.log(m.predict_proba(X.reshape(-1, f)) + EPS).reshape(n, k, -1).mean(axis=1)
    if comp == "lang_mixture":
        return np.log(m.predict_proba(X) + EPS)
    if comp == "alm_qwen":
        return X - m
    if comp == "mert_ft12":
        return X
    raise ValueError(f"unknown component {comp}")


def ft_checkpoints(key: str) -> list[str]:
    return [f"ft_{key}_L{FT_LAYER}_s{s}.pt" for s in FT_SEEDS]


def fit(key: str, feats: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """feats: component name -> train features. mert_* (N, L, D); hc_c10 (N, n_chunks, F); lang_mixture (N, 100);
    alm_qwen (N, 6) raw label log-likelihoods; mert_ft12 (N, k) out-of-fold log-probs, seed mean."""
    k = len(LABELS[key])
    comps = list(dict.fromkeys(RECIPES[key] + FALLBACK.get(key, [])))
    models = {c: _fit_component(c, feats[c], y, k) for c in comps}
    temps = {c: 1.0 for c in comps}
    if CALIBRATE[key]:
        for c in comps:
            oof = np.zeros((len(y), k))
            if c == "mert_ft12":
                oof = _norm(feats[c])
            else:
                for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(y, y):
                    oof[b] = _norm(_component_logprobs(c, _fit_component(c, feats[c][a], y[a], k), feats[c][b]))
            temps[c] = float(minimize_scalar(lambda t: -_norm(oof * t)[np.arange(len(y)), y].mean(),
                                             bounds=(0.05, 20), method="bounded").x)
    ckpt = {"key": key, "labels": LABELS[key], "recipe": RECIPES[key], "fallback": FALLBACK.get(key),
            "models": models, "temps": temps}
    if "mert_ft12" in comps:
        ckpt["ft"] = {"from_layer": FT_LAYER, "checkpoints": ft_checkpoints(key)}
    return ckpt


def component_logprobs(ckpt: dict, feats: dict[str, np.ndarray], recipe: list[str] | None = None) -> dict[str, np.ndarray]:
    return {c: _component_logprobs(c, ckpt["models"][c], feats[c]) for c in (recipe or ckpt["recipe"])}


def predict_proba(ckpt: dict, feats: dict[str, np.ndarray], recipe: list[str] | None = None) -> np.ndarray:
    recipe = recipe or ckpt["recipe"]
    comps = component_logprobs(ckpt, feats, recipe)
    temps = ckpt.get("temps", {})
    z = np.mean([_norm(_norm(comps[c]) * temps.get(c, 1.0)) for c in recipe], axis=0)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def save(ckpt: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ckpt, path, compress=3)


def load(path: str | Path) -> dict:
    return joblib.load(path)
