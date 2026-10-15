"""Trim leading silence from existing WAV files in place (renders made before render.py trimmed them).

uv run python scripts/trim_wavs.py outputs/plan_fill/* outputs/exp2/* outputs/ref/heldout_crop32
"""

import argparse
from pathlib import Path

from hw2.render import trim_leading_silence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    args = p.parse_args()
    n, total = 0, 0.0
    for d in map(Path, args.dirs):
        for wav in sorted(d.rglob("*.wav")):
            cut = trim_leading_silence(wav)
            n, total = n + (cut > 0), total + cut
            if cut > 0.5:
                print(f"{wav}: cut {cut:.2f}s")
    print(f"trimmed {n} files, {total:.1f}s in total")


if __name__ == "__main__":
    main()
