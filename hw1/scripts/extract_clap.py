from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hw1.data import LABELS, load_audio, load_manifest
from hw1.features.clap import build_prompts, class_text_embeddings, cosine_scores, embed_clips, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="A, B, or a dataset folder path")
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    key = args.dataset if args.dataset in ("A", "B") else None
    df = load_manifest(args.dataset)
    if key is None:
        key = str(df["sample_id"].iloc[0]).split("_")[0]
    labels = LABELS[key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_model(device)

    waveforms = [load_audio(p) for p in df["path"]]
    emb = embed_clips(model, processor, waveforms, device, batch_size=args.batch_size)

    prompts = build_prompts(key)
    text_p1 = class_text_embeddings(model, processor, prompts["p1"], device)
    text_p2 = class_text_embeddings(model, processor, prompts["p2"], device)
    zs_p1 = cosine_scores(emb, text_p1)
    zs_p2 = cosine_scores(emb, text_p2)

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}_clap.npz"
    np.savez(
        out_path,
        emb=emb,
        zs_p1=zs_p1.astype(np.float32),
        zs_p2=zs_p2.astype(np.float32),
        sample_id=df["sample_id"].to_numpy(dtype=str),
        split=df["split"].to_numpy(dtype=str),
        label=df["label"].to_numpy(dtype=str),
        labels=np.array(labels, dtype=str),
        prompts_p1=json.dumps(prompts["p1"]),
        prompts_p2=json.dumps(prompts["p2"]),
    )

    print(f"dataset={key} n={len(df)} emb_shape={emb.shape} out={out_path}")
    print(f"elapsed_s={elapsed:.1f} peak_gpu_mb={peak_mem:.0f}")


if __name__ == "__main__":
    main()
