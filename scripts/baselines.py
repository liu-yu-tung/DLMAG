"""Real-data metrics and sanity baselines.

Sets, each one piece per training song:
- real_full: whole songs, as in the TA example.
- real_crop32: a random 32-bar window (the length of a task-a output).
- shuffled_crop32: the same windows with bars in random order.
- loop_crop32: one non-empty bar of the window repeated 32 times.
- random_notes32: 32 bars of random notes, same note count per bar as the window, uniform pitch and position.

uv run python scripts/baselines.py
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from hw2.metrics import evaluate, summarize
from hw2.remi import DATA_ROOT, HW2_ROOT, load_sequences, load_vocab, split_bars

N_BARS = 32


def flatten(bars):
    return ["<BOS>"] + [t for bar in bars for t in bar] + ["<EOS>"]


def random_bar(n_notes, rng, pos_per_beat):
    events = sorted((rng.randrange(4 * pos_per_beat), rng.randrange(21, 109)) for _ in range(n_notes))
    bar = ["Bar"]
    for pos, pitch in events:
        bar += [f"Pos_{pos}", f"Pitch_{pitch}", "Vel_81", f"Dur_{pos_per_beat}"]
    return bar


def build_sets(songs, pos_per_beat, seed):
    rng = random.Random(seed)
    sets = {k: [] for k in ["real_full", "real_crop32", "shuffled_crop32", "loop_crop32", "random_notes32"]}
    for song in songs:
        name, bars = song["song_name"], split_bars(song["tokens"])
        sets["real_full"].append({"song_name": name, "tokens": song["tokens"]})
        start = rng.randrange(max(1, len(bars) - N_BARS + 1))
        window = bars[start : start + N_BARS]
        sets["real_crop32"].append({"song_name": name, "tokens": flatten(window)})
        shuffled = window[:]
        rng.shuffle(shuffled)
        sets["shuffled_crop32"].append({"song_name": name, "tokens": flatten(shuffled)})
        non_empty = [b for b in window if any(t.startswith("Pitch_") for t in b)]
        sets["loop_crop32"].append({"song_name": name, "tokens": flatten([rng.choice(non_empty)] * N_BARS)})
        counts = [sum(t.startswith("Pitch_") for t in b) for b in window]
        sets["random_notes32"].append({"song_name": name, "tokens": flatten([random_bar(n, rng, pos_per_beat) for n in counts])})
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_per_beat", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(HW2_ROOT / "outputs" / "baselines"))
    parser.add_argument("--summary_csv", type=str, default=str(HW2_ROOT / "results" / "baselines.csv"))
    args = parser.parse_args()

    tok_dir = DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}"
    tok2idx, _ = load_vocab(tok_dir / "tok2idx.json")
    songs = load_sequences(tok_dir / "sequences.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, pieces in build_sets(songs, args.pos_per_beat, args.seed).items():
        (out_dir / f"{name}.json").write_text(json.dumps(pieces))
        df = evaluate(pieces, tok2idx, seed=args.seed)
        df.to_csv(out_dir / f"{name}_metrics.csv", index=False)
        rows.append({"set": name, **summarize(df)})
        print(rows[-1])

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(args.summary_csv, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
