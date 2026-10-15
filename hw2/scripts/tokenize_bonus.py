"""Tokenize POP909 training songs into bar-interleaved MELODY/PIANO sequences for the bonus task.

uv run python scripts/tokenize_bonus.py
uv run python scripts/tokenize_bonus.py --lookahead --out_dir data/tokens/bonus_la
uv run python scripts/tokenize_bonus.py --chords --out_dir data/tokens/bonus_chd
"""

import argparse
import json
from pathlib import Path

from hw2.bonus import tokenize_song, write_vocab
from hw2.remi import DATA_ROOT, load_vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--out_dir", type=str, default=str(DATA_ROOT.parent / "tokens" / "bonus_ppb16"))
    p.add_argument("--lookahead", action="store_true", help="melody one bar ahead of the piano")
    p.add_argument("--chords", action="store_true", help="add chord tokens from chord_midi.txt")
    args = p.parse_args()
    variant = {"lookahead": args.lookahead, "chords": args.chords}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base, _ = load_vocab(DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}" / "tok2idx.json")
    tok2idx = write_vocab(base, out / "tok2idx.json", variant)
    (out / "variant.json").write_text(json.dumps(variant))

    songs = []
    for d in sorted((DATA_ROOT / "training").iterdir()):
        tokens = tokenize_song(d / f"{d.name}.mid", d / "beat_midi.txt", args.pos_per_beat, variant, d / "chord_midi.txt")
        if tokens is None:
            print(f"skip {d.name}")
            continue
        oov = [t for t in tokens if t not in tok2idx]
        assert not oov, (d.name, oov[:5])
        songs.append({"song_name": d.name, "tokens": tokens})
    (out / "sequences.json").write_text(json.dumps(songs))
    lengths = sorted(len(s["tokens"]) for s in songs)
    print(f"{len(songs)} songs, vocab {len(tok2idx)}, tokens total {sum(lengths)}, median {lengths[len(lengths) // 2]}")


if __name__ == "__main__":
    main()
