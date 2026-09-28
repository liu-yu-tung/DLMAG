"""Separate every clip into vocals/drums/bass/other with Hybrid Demucs and store 16-bit FLAC stems.

Output: data/stems/dataset_{key}/{sample_id}/{stem}.flac (24 kHz mono, under data/, so never committed).
Resumable: clips whose four stems already exist are skipped.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from hw1.data import load_audio, load_manifest
from hw1.features.separation import SR_IN, STEMS, load_separator, separate, stem_dir


def _write(d: Path, stems: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, y in stems.items():
        sf.write(d / f"{name}.flac", y.clip(-1, 1), SR_IN, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = load_manifest(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    todo = [r for r in df.itertuples() if not all((stem_dir(args.dataset, r.sample_id) / f"{s}.flac").exists() for s in STEMS)]
    print(f"{args.dataset}: {len(df) - len(todo)} done, {len(todo)} to go")
    if not todo:
        return

    model = load_separator()
    if list(model.sources) != STEMS:
        raise ValueError(f"unexpected sources {model.sources}")
    t0 = time.time()
    with ThreadPoolExecutor(2) as loader, ThreadPoolExecutor(2) as writer:
        audio = loader.map(lambda r: (r, load_audio(r.path)), todo)
        pending = []
        for r, y in tqdm(audio, total=len(todo), desc=f"stems {args.dataset}"):
            pending.append(writer.submit(_write, stem_dir(args.dataset, r.sample_id), separate(model, y)))
        for p in pending:
            p.result()
    print(f"{len(todo)} clips in {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
