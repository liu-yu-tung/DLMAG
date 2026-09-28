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


def cached(key: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    mert = load_features(FEAT / f"{key}_mertv2.npz")
    feats = {"mert_layeravg": final.mert_means(mert["feats"])}
    if "hc_c10" in final.RECIPES[key]:
        ch = load_features(FEAT / f"{key}_handcrafted_chunks.npz")
        if not (ch["sample_id"] == mert["sample_id"]).all():
            raise ValueError("clip order differs between MERT and hand-crafted features")
        feats["hc_c10"] = ch[f"c{final.HC_CHUNK_S}"]
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
        print(f"{key} {'+'.join(ckpt['recipe'])}: validation {report[key]}")

        test = split == "test"
        preds[f"dataset_{key}"] = dict(zip(ids[test].tolist(), top3_labels(final.predict_proba(ckpt, sel("test")), LABELS[key])))

    args.cache_pred.parent.mkdir(parents=True, exist_ok=True)
    args.cache_pred.write_text(json.dumps(preds, indent=2))
    (HW1 / "results" / "final_validation.json").write_text(
        json.dumps({k: {"recipe": final.RECIPES[k], **v} for k, v in report.items()}, indent=2) + "\n")
    print("checkpoints in", args.ckpt_dir, "| cached-feature predictions in", args.cache_pred)


if __name__ == "__main__":
    main()
