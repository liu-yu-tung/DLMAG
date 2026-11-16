import argparse
import sys
from pathlib import Path

import numpy as np

HW3 = Path(__file__).resolve().parents[2]
CACHE = HW3 / "outputs/cache/runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed generated clips with the same encoders and preprocessing as the MusicCaps caches, so the attribute probes "
        "(scripts/analysis/attribute_probes.py) can score them. mert: run with the hw1 venv; clap90: run with the av-benchmark venv. "
        "Writes outputs/cache/runs/<set>__<run>_<kind>.npy and .ids.txt (rows in id order).")
    parser.add_argument("--kind", choices=("mert", "clap90"), required=True)
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run folders under outputs/<set>/")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clip_paths(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("*.wav"), key=lambda path: int(path.stem))
    assert paths, f"{folder}: no clips"
    return paths


def main():
    args = parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.kind == "mert":
        sys.path.insert(0, str(HW3 / "scripts"))
        from evaluation.fd_mert import DEFAULT_MODEL, embed, load_mert
        model = load_mert(DEFAULT_MODEL, "fp32", args.device)
        encode = lambda paths: embed(model, paths, 4)  # noqa: E731
    else:
        sys.path.insert(0, str(HW3 / "scripts"))
        from retrieval.retrieval import load_model, normalize, read_clip
        import torch
        model = load_model(args.device)

        def encode(paths):
            with torch.inference_mode():
                return normalize(np.concatenate([model.get_audio_embedding_from_data(x=np.stack([read_clip(p) for p in paths[i:i + 8]]),
                                                                                       use_tensor=False) for i in range(0, len(paths), 8)]))
    for run in args.runs.split(","):
        out = CACHE / f"{args.set_name}__{run}_{args.kind}.npy"
        if out.exists():
            continue
        paths = clip_paths(HW3 / "outputs" / args.set_name / run)
        features = encode(paths)
        np.save(out, features.astype(np.float32))
        out.with_suffix(".ids.txt").write_text("\n".join(p.stem for p in paths) + "\n")
        print(run, features.shape, "->", out)


if __name__ == "__main__":
    main()
