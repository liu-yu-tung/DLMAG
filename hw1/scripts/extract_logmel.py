import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.logmel import EPS, HOP, N_FFT, N_MELS, SR, logmel

HW1 = Path(__file__).resolve().parents[1]


def _one(path: str) -> np.ndarray:
    return logmel(load_audio(path)).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="A, B, or a dataset folder path")
    ap.add_argument("--out", type=Path, default=HW1 / "features")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only, for timing")
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    key = str(df["sample_id"].iloc[0]).split("_")[0]
    suffix = "_limit" if args.limit else ""
    jobs = min(args.jobs, 4)

    t0 = time.time()
    mels = Parallel(n_jobs=jobs)(delayed(_one)(p) for p in tqdm(df["path"], desc=f"logmel {key}"))
    arr = np.stack(mels)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in log-mel (float16 overflow?)")

    args.out.mkdir(parents=True, exist_ok=True)
    npy = args.out / f"{key}_logmel{suffix}.npy"
    np.save(npy, arr)
    np.savez(
        args.out / f"{key}_logmel{suffix}_meta.npz",
        sample_id=df["sample_id"].to_numpy(str),
        split=df["split"].to_numpy(str),
        label=df["label"].to_numpy(str),
    )
    params = dict(sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, eps=EPS, transform="ln(eps + mel power)", dtype="float16")
    (args.out / f"{key}_logmel{suffix}.json").write_text(json.dumps({**params, "shape": list(arr.shape)}, indent=2))
    print(f"{npy} {arr.shape} min {arr.min():.2f} max {arr.max():.2f} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
