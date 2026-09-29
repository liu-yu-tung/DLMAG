"""MuQ-large (OpenMuQ/MuQ-large-msd-iter, week 2b p.23) hidden states for every clip: per-layer mean and std over time.

Output: features/{key}_muq.npz with feats (N, L, 2*D) float16 (same layout as the MERT file), sample_id, split,
label. --check loads the model twice and compares the features of the first clips (remote-model buffer check).
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from muq import MuQ
from tqdm import tqdm

from hw1.data import load_audio, load_manifest

HW1 = Path(__file__).resolve().parents[1]
MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def load(device: str = "cuda"):
    return MuQ.from_pretrained(MODEL_ID).eval().to(device)


@torch.inference_mode()
def pooled(model, wavs: list[np.ndarray]) -> np.ndarray:
    n = min(len(w) for w in wavs)
    x = torch.as_tensor(np.stack([w[:n] for w in wavs]), dtype=torch.float32, device=next(model.parameters()).device)
    hs = torch.stack(model(x, output_hidden_states=True).hidden_states, 1).float()  # (B, L, T, D)
    return torch.cat([hs.mean(2), hs.std(2)], -1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["A", "B"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        wavs = [load_audio(p) for p in load_manifest("B")["path"].head(4)]
        a = pooled(load(), wavs)
        torch.cuda.empty_cache()
        b = pooled(load(), wavs)
        print("shape", a.shape, "finite", bool(np.isfinite(a).all()),
              "max rel diff across loads", float(np.abs(a - b).max() / np.abs(a).max()))
        return

    df = load_manifest(args.dataset)
    model = load()
    paths = df["path"].tolist()
    batches = [paths[i : i + args.batch] for i in range(0, len(paths), args.batch)]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(4) as pool:
        for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc=f"MuQ {args.dataset}"):
            out.append(pooled(model, wavs))
    feats = np.concatenate(out)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite MuQ features")
    path = HW1 / "features" / f"{args.dataset}_muq.npz"
    np.savez(path, feats=feats.astype(np.float16), sample_id=df["sample_id"].to_numpy(str),
             split=df["split"].to_numpy(str), label=df["label"].to_numpy(str))
    print(path, feats.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
