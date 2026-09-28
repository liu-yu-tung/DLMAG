import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

from hw1.data import load_manifest, load_audio
from hw1.features.mert import DEFAULT_MODEL, load_mert, pooled_layers

HW1 = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="A, B, or a dataset folder path")
    ap.add_argument("--out", type=Path, default=HW1 / "features")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default=None, help="output suffix, default derived from model id")
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    key = str(df["sample_id"].iloc[0]).split("_")[0]
    tag = args.tag or ("mertv2" if "v2" in args.model.lower() else "mertv1")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{key}_{tag}.npz"

    model = load_mert(args.model, args.dtype)
    torch.cuda.reset_peak_memory_stats()
    paths = df["path"].tolist()
    batches = [paths[i : i + args.batch] for i in range(0, len(paths), args.batch)]
    feats: list[np.ndarray] = []
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        loaded = pool.map(lambda b: [load_audio(p) for p in b], batches)
        for wavs in tqdm(loaded, total=len(batches), desc=f"{key} {tag}"):
            lengths = [len(w) for w in wavs]
            x = np.zeros((len(wavs), max(lengths)), dtype=np.float32)
            for i, w in enumerate(wavs):
                x[i, : len(w)] = w
            feats.append(pooled_layers(model, x, None if len(set(lengths)) == 1 else lengths).astype(np.float16))
    elapsed = time.time() - t0
    arr = np.concatenate(feats)
    assert arr.shape[0] == len(df)

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "num_layers": int(arr.shape[1]),
        "hidden_size": int(arr.shape[2] // 2),
        "layout": "feats[n, layer, :D]=mean, feats[n, layer, D:]=std over valid frames",
        "seconds": round(elapsed, 1),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    np.savez(
        out_path,
        feats=arr,
        sample_id=df["sample_id"].to_numpy(str),
        split=df["split"].to_numpy(str),
        label=df["label"].to_numpy(str),
        meta=json.dumps(meta),
    )
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out_path, arr.shape, meta)


if __name__ == "__main__":
    main()
