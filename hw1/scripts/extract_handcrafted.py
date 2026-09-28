import argparse
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.handcrafted import extract

HW1 = Path(__file__).resolve().parents[1]


def _one(path: str) -> list[float]:
    return list(extract(load_audio(path)).values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="A, B, or a dataset folder path")
    ap.add_argument("--out", type=Path, default=HW1 / "features")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only, for timing")
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    key = str(df["sample_id"].iloc[0]).split("_")[0]
    names = list(extract(load_audio(df["path"].iloc[0])).keys())

    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs)(delayed(_one)(p) for p in tqdm(df["path"], desc=f"handcrafted {key}"))
    feats = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(feats).all():
        raise ValueError(f"non-finite features in {int((~np.isfinite(feats)).sum())} cells")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{key}_handcrafted{'_limit' if args.limit else ''}.npz"
    np.savez(
        out_path,
        feats=feats,
        names=np.array(names),
        sample_id=df["sample_id"].to_numpy(str),
        split=df["split"].to_numpy(str),
        label=df["label"].to_numpy(str),
    )
    print(f"{out_path} {feats.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
