"""Inference: audio -> features -> checkpoints -> top-3 labels per clip, written as the submission JSON.

    python scripts/predict.py --data /path/to/data --out r12345678.json

--data is a folder holding dataset_A/ and dataset_B/ (each with manifest.csv and audio/). Only rows of
--split (default test) are predicted. Output: {"dataset_A": {sample_id: [top1, top2, top3]}, "dataset_B": ...}.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from hw1 import final
from hw1.data import load_audio, load_manifest
from hw1.features.handcrafted import extract_chunks
from hw1.features.mert import load_mert, pooled_layers
from hw1.metrics import top3_labels, validate_predictions

HW1 = Path(__file__).resolve().parents[1]


def mert_features(model, paths: list[str], batch: int) -> np.ndarray:
    batches = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    out = []
    with ThreadPoolExecutor(4) as pool:
        for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc="MERT"):
            lengths = [len(w) for w in wavs]
            x = np.zeros((len(wavs), max(lengths)), dtype=np.float32)
            for i, w in enumerate(wavs):
                x[i, : len(w)] = w
            out.append(pooled_layers(model, x, None if len(set(lengths)) == 1 else lengths))
    return final.mert_means(np.concatenate(out))


def hc_features(paths: list[str], jobs: int) -> np.ndarray:
    rows = Parallel(n_jobs=jobs)(
        delayed(lambda p: extract_chunks(load_audio(p), final.HC_CHUNK_S))(p) for p in tqdm(paths, desc="hand-crafted")
    )
    return np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder containing dataset_A/ and dataset_B/")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--ckpt-dir", type=Path, default=HW1 / "checkpoints")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_mert(device=args.device)
    preds, manifests = {}, {}
    for key in args.datasets:
        df = load_manifest(args.data / f"dataset_{key}")
        manifests[f"dataset_{key}"] = df
        df = df[df["split"] == args.split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows with split={args.split} in dataset_{key}")
        ckpt = final.load(args.ckpt_dir / f"{key}.joblib")
        paths = df["path"].tolist()
        feats = {"mert_layeravg": mert_features(model, paths, args.batch)}
        if "hc_c10" in ckpt["recipe"]:
            feats["hc_c10"] = hc_features(paths, args.jobs)
        probs = final.predict_proba(ckpt, feats)
        preds[f"dataset_{key}"] = dict(zip(df["sample_id"].tolist(), top3_labels(probs, ckpt["labels"])))
        print(f"dataset_{key}: {len(df)} clips, recipe {'+'.join(ckpt['recipe'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds, indent=2) + "\n")
    if args.split == "test":
        errors = validate_predictions(preds, manifests)
        if errors:
            raise SystemExit("invalid predictions:\n" + "\n".join(errors))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
