"""Hybrid Demucs encoder features for every clip -> features/{key}_demucs_latent.npz (one array per layer)."""

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.demucs_latent import encoder_features, layer_names
from hw1.features.separation import load_separator

HW1 = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    df = load_manifest(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    model = load_separator()
    names = layer_names(model)
    feats = {n: [] for n in names}
    t0 = time.time()
    for p in tqdm(df["path"], desc=f"demucs latent {args.dataset}"):
        f = encoder_features(model, load_audio(p))
        for n in names:
            feats[n].append(f[n])
    arrs = {n: np.stack(v).astype(np.float32) for n, v in feats.items()}
    for n, a in arrs.items():
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite values in {n}")
    path = HW1 / "features" / f"{args.dataset}_demucs_latent{'_limit' if args.limit else ''}.npz"
    np.savez(path, **arrs, layers=np.array(names), sample_id=df["sample_id"].to_numpy(str),
             split=df["split"].to_numpy(str), label=df["label"].to_numpy(str))
    print(path, {n: a.shape[1] for n, a in arrs.items()}, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
