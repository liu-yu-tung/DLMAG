"""Whisper language log-probabilities for every clip, from the vocal stem and from the mixture.

Output: features/{key}_langid.npz with vocals (N, L) and mixture (N, L) log-probs and the language codes.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.langid import LangID
from hw1.features.separation import SR_IN, stem_dir

HW1 = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HW1 / "features")
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    lid = LangID()
    sources = {
        "vocals": [str(stem_dir(args.dataset, s) / "vocals.flac") for s in df["sample_id"]],
        "mixture": df["path"].tolist(),
    }
    out = {}
    t0 = time.time()
    for name, paths in sources.items():
        batches = [paths[i : i + args.batch] for i in range(0, len(paths), args.batch)]
        res = []
        with ThreadPoolExecutor(4) as pool:
            for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc=name):
                res.append(lid.log_probs(wavs, SR_IN))
        out[name] = np.concatenate(res).astype(np.float32)
        if not np.isfinite(out[name]).all():
            raise ValueError(f"non-finite log-probs for {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.dataset}_langid{'_limit' if args.limit else ''}.npz"
    np.savez(path, **out, codes=np.array(lid.codes), sample_id=df["sample_id"].to_numpy(str),
             split=df["split"].to_numpy(str), label=df["label"].to_numpy(str))
    top = {name: np.array(lid.codes)[v.argmax(1)] for name, v in out.items()}
    print(f"{path} in {time.time() - t0:.0f}s")
    for name, t in top.items():
        codes, counts = np.unique(t, return_counts=True)
        print(f"  top-1 language from {name}:", dict(sorted(zip(codes, counts.tolist()), key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
