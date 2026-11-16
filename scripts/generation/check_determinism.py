import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HW3 = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the same clip ids from two runs of the same settings and seeds. "
        "Writes whether the files are bit-identical and the largest sample difference.")
    parser.add_argument("--first", type=Path, required=True, help="audio folder of the original run")
    parser.add_argument("--second", type=Path, required=True, help="audio folder of the repeated run")
    parser.add_argument("--ids", required=True, help="comma-separated clip ids")
    parser.add_argument("--out", type=Path, default=HW3 / "results/determinism.json")
    return parser.parse_args()


def main():
    args = parse_args()
    clips = []
    for item_id in args.ids.split(","):
        paths = [folder / f"{item_id}.wav" for folder in (args.first, args.second)]
        first, second = (sf.read(path, dtype="float64")[0] for path in paths)
        assert first.shape == second.shape, (item_id, first.shape, second.shape)
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        clips.append({
            "id": item_id, "identical_file": digests[0] == digests[1],
            "max_abs_difference": float(np.abs(first - second).max()),
            "rms_first": float(np.sqrt((first**2).mean())),
            "rms_difference": float(np.sqrt(((first - second) ** 2).mean())),
        })
    record = {"first": str(args.first), "second": str(args.second), "clips": clips,
              "all_identical": all(clip["identical_file"] for clip in clips),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
