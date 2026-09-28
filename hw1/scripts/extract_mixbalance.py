"""Mix-balance features for every clip from its stored Demucs stems -> features/{key}_mixbalance.npz."""

import argparse
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.mixbalance import extract
from hw1.features.separation import STEMS, stem_dir

HW1 = Path(__file__).resolve().parents[1]


def _one(key: str, sample_id: str, path: str) -> dict[str, float]:
    d = stem_dir(key, sample_id)
    return extract({s: load_audio(d / f"{s}.flac") for s in STEMS}, load_audio(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=HW1 / "features")
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(_one)(args.dataset, r.sample_id, r.path) for r in tqdm(df.itertuples(), total=len(df), desc="mixbalance"))
    names = list(rows[0])
    feats = np.asarray([[r[n] for n in names] for r in rows], dtype=np.float32)
    if not np.isfinite(feats).all():
        raise ValueError(f"non-finite features in {int((~np.isfinite(feats)).sum())} cells")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.dataset}_mixbalance.npz"
    np.savez(out, feats=feats, names=np.array(names), sample_id=df["sample_id"].to_numpy(str),
             split=df["split"].to_numpy(str), label=df["label"].to_numpy(str))
    print(f"{out} {feats.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
