"""Fit the submitted models on train features only and save checkpoints/{A,B}.joblib.

Also reports validation metrics (must match results/fusion.json) and writes test predictions made from the
cached features, so predict.py (which extracts features from audio) can be checked against them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hw1 import final
from hw1.data import LABELS
from hw1.metrics import evaluate, top3_labels
from hw1.probe import load_features

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"


def alm_scores(key: str, ids: np.ndarray) -> np.ndarray:
    """Cached Qwen2-Audio plain-prompt label log-likelihoods (scripts/alm_qwen.py), in the given clip order."""
    rows = {}
    for f in (f"{key}_alm_plain_train_test.npz", f"{key}_alm_plain_validation.npz"):
        z = np.load(FEAT / f)
        rows.update(zip(z["sample_id"].tolist(), z["logp"]))
    missing = [s for s in ids.tolist() if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} clips have no cached Qwen scores")
    return np.stack([rows[s] for s in ids.tolist()])


def ft_scores(key: str, ids: np.ndarray) -> np.ndarray:
    """Fine-tuned top-block log-probs (scripts/finetune_mert.py), mean over final.FT_SEEDS, in the given clip order:
    out-of-fold for train clips, the full-train fit for validation and test clips."""
    runs = []
    for s in final.FT_SEEDS:
        z = np.load(FEAT / f"ft_{key}_L{final.FT_LAYER}_s{s}.npz")
        if not all(int(z[f"done_f{i}"]) for i in range(5)):
            raise ValueError(f"ft_{key}_L{final.FT_LAYER}_s{s}: not all folds done")
        rows = {}
        for part in ("oof", "val", "test"):
            rows.update(zip(z[f"{part}_id"].tolist(), z[part]))
        missing = [i for i in ids.tolist() if i not in rows]
        if missing:
            raise ValueError(f"{len(missing)} clips have no fine-tuned log-probs (seed {s})")
        runs.append(np.stack([rows[i] for i in ids.tolist()]))
    return np.mean(runs, axis=0)


def cached(key: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    mert = load_features(FEAT / f"{key}_mertv2.npz")
    need = set(final.RECIPES[key] + final.FALLBACK.get(key, []))
    feats = {c: final.mert_means(mert["feats"]) for c in ("mert_layeravg", "mert_ord10") if c in need}
    if "alm_qwen" in need:
        feats["alm_qwen"] = alm_scores(key, mert["sample_id"])
    if "mert_ft12" in need:
        feats["mert_ft12"] = ft_scores(key, mert["sample_id"])
    if "lang_mixture" in final.RECIPES[key]:
        lid = load_features(FEAT / f"{key}_langid.npz")
        if not (lid["sample_id"] == mert["sample_id"]).all():
            raise ValueError("clip order differs between MERT and language-ID features")
        feats["lang_mixture"] = lid["mixture"]
    y = np.array([LABELS[key].index(l) if l else -1 for l in mert["label"]])
    return feats, y, mert["split"], mert["sample_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=HW1 / "checkpoints")
    ap.add_argument("--cache-pred", type=Path, default=HW1 / "outputs" / "pred_from_cache.json")
    args = ap.parse_args()

    preds, report = {}, {}
    for key in ("A", "B"):
        feats, y, split, ids = cached(key)
        sel = lambda s: {c: v[split == s] for c, v in feats.items()}
        ckpt = final.fit(key, sel("train"), y[split == "train"])
        final.save(ckpt, args.ckpt_dir / f"{key}.joblib")

        m = evaluate(final.predict_proba(ckpt, sel("validation")), y[split == "validation"], LABELS[key])
        report[key] = {k: round(m[k], 4) for k in ("top1", "top3", "S")}
        print(f"{key} {'+'.join(ckpt['recipe'])}: validation {report[key]} | temperatures {ckpt['temps']}")
        if ckpt.get("fallback"):
            mf = evaluate(final.predict_proba(ckpt, sel("validation"), ckpt["fallback"]), y[split == "validation"], LABELS[key])
            report[f"{key}_fallback"] = {"recipe": ckpt["fallback"], **{k: round(mf[k], 4) for k in ("top1", "top3", "S")}}
            print(f"{key} fallback {'+'.join(ckpt['fallback'])}: validation {report[f'{key}_fallback']}")

        test = split == "test"
        preds[f"dataset_{key}"] = dict(zip(ids[test].tolist(), top3_labels(final.predict_proba(ckpt, sel("test")), LABELS[key])))

    args.cache_pred.parent.mkdir(parents=True, exist_ok=True)
    args.cache_pred.write_text(json.dumps(preds, indent=2))
    (HW1 / "results" / "final_validation.json").write_text(
        json.dumps({k: {"recipe": final.RECIPES[k], **v} if k in final.RECIPES else v for k, v in report.items()}, indent=2) + "\n")
    print("checkpoints in", args.ckpt_dir, "| cached-feature predictions in", args.cache_pred)


if __name__ == "__main__":
    main()
