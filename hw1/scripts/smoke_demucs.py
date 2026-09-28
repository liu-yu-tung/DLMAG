"""One-clip-per-class Demucs check: speed, GPU memory, mixture reconstruction, stem energy shares.

Writes stems as WAV to --out for listening. Uses train clips only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hw1.data import LABELS, load_audio, load_split
from hw1.features.separation import SR_IN, load_separator, separate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-dataset", type=int, default=6, help="clips per dataset, one per class in label order")
    args = ap.parse_args()

    t0 = time.time()
    model = load_separator()
    print(f"load {time.time() - t0:.1f}s, sources {model.sources}")
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for key in ("A", "B"):
        df = load_split(key, "train")
        picks = [df[df["label"] == lab].iloc[0] for lab in LABELS[key]][: args.per_dataset]
        for r in picks:
            y = load_audio(r["path"])
            torch.cuda.synchronize()
            t = time.time()
            stems = separate(model, y)
            torch.cuda.synchronize()
            dt = time.time() - t
            recon = sum(stems.values())
            snr = 10 * np.log10(np.sum(y**2) / max(np.sum((y - recon) ** 2), 1e-12))
            energy = {k: float(np.sum(v**2)) for k, v in stems.items()}
            total = sum(energy.values())
            share = {k: round(v / total, 3) for k, v in energy.items()}
            rows.append({"sample_id": r["sample_id"], "label": r["label"], "seconds": round(dt, 2),
                         "recon_snr_db": round(float(snr), 1), "energy_share": share})
            d = args.out / f"{r['sample_id']}_{r['label']}"
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / "mix.wav", y, SR_IN)
            for k, v in stems.items():
                sf.write(d / f"{k}.wav", v, SR_IN)
            print(json.dumps(rows[-1]))
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB; "
          f"mean {np.mean([r['seconds'] for r in rows[1:]]):.2f} s/clip after warm-up")
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
