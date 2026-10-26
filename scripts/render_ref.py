"""Render held-out real songs as listening anchors.

- crop: more of the random 32-bar windows in outputs/ref/heldout_crop32/sequences.json.
- open: the first 32 bars of held-out songs (a song start, like generation from <BOS>) -> outputs/ref/heldout_open32.
  Songs already rendered as crops are skipped, so a listener never hears the same song twice.

uv run python scripts/render_ref.py --set open --n 12
"""

import argparse
import json
import random

from hw2.data import split_songs
from hw2.remi import DATA_ROOT, HW2_ROOT, load_sequences, split_bars, tokens_to_midi
from hw2.render import midi_to_wav

REF = HW2_ROOT / "outputs" / "ref"
N_BARS = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["crop", "open"], default="crop")
    p.add_argument("--n", type=int, default=12, help="new songs to render")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_val", type=int, default=45, help="must match train.py, so the songs are held out")
    p.add_argument("--pos_per_beat", type=int, default=16)
    args = p.parse_args()
    crops = REF / "heldout_crop32"
    if args.set == "crop":
        out, pieces = crops, json.loads((crops / "sequences.json").read_text())
    else:
        out = REF / "heldout_open32"
        _, val = split_songs(load_sequences(DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}" / "sequences.json"), args.n_val)
        pieces = [{"song_name": s["song_name"], "tokens": ["<BOS>"] + [t for bar in split_bars(s["tokens"])[:N_BARS] for t in bar] + ["<EOS>"]}
                  for s in val if not (crops / f"{s['song_name']}.wav").exists()]
    out.mkdir(parents=True, exist_ok=True)
    todo = sorted(x["song_name"] for x in pieces if not (out / f"{x['song_name']}.wav").exists())
    pick = set(random.Random(args.seed).sample(todo, min(args.n, len(todo))))
    for x in pieces:
        if x["song_name"] in pick:
            tokens_to_midi(x["tokens"], out / f"{x['song_name']}.mid", args.pos_per_beat)
            midi_to_wav(out / f"{x['song_name']}.mid", out / f"{x['song_name']}.wav")
    if args.set == "open":
        done = json.loads((out / "sequences.json").read_text()) if (out / "sequences.json").exists() else []
        names = {x["song_name"] for x in done}
        (out / "sequences.json").write_text(json.dumps(done + [x for x in pieces if x["song_name"] in pick and x["song_name"] not in names]))
    print(f"rendered {sorted(pick)} -> {out}")


if __name__ == "__main__":
    main()
