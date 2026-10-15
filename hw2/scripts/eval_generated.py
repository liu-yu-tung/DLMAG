"""Evaluate generated sets: H1/H4/GS (TA functions), sampled-token NLL, copy rate vs. training songs.

Each argument is a folder with sequences.json (and gen_info.json) from generate_uncond.py / generate_cont.py.
Appends one row per folder to --summary_csv and writes <folder>/metrics.csv.

uv run python scripts/eval_generated.py outputs/exp1/* --summary_csv results/exp1.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hw2.data import split_songs
from hw2.metrics import evaluate, summarize
from hw2.novelty import CopyIndex
from hw2.remi import DATA_ROOT, load_sequences, load_vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--summary_csv", type=str, required=True)
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--n_val", type=int, default=45, help="must match train.py, so copy rate uses the training split only")
    p.add_argument("--skip_bars", type=int, default=0, help="bars to skip in copy rate (8 for task b prompts)")
    args = p.parse_args()

    tok_dir = DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}"
    tok2idx, _ = load_vocab(tok_dir / "tok2idx.json")
    train_songs, _ = split_songs(load_sequences(tok_dir / "sequences.json"), args.n_val)
    index = CopyIndex(train_songs)

    rows = []
    for d in map(Path, args.dirs):
        pieces = load_sequences(d / "sequences.json")
        df = evaluate(pieces, tok2idx)
        copies = pd.DataFrame([index.score(s["tokens"], args.skip_bars) for s in pieces])
        df = pd.concat([df, copies], axis=1)
        info = json.loads((d / "gen_info.json").read_text()) if (d / "gen_info.json").exists() else {}
        if info:
            df["nll"] = [x["nll"] for x in info["pieces"]]
            df["n_tokens"] = [x["n_tokens"] for x in info["pieces"]]
        df.to_csv(d / "metrics.csv", index=False)
        row = {"dir": str(d), "config": info.get("config", ""), "ckpt": info.get("args", {}).get("ckpt", ""), **summarize(df)}
        for col in ["nll", "n_tokens", "bar_copy", "bar_copy_any_key", "ngram_copy", "note8_copy"]:
            if col in df:
                row[f"{col}_mean"] = float(np.mean(df[col]))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    out = Path(args.summary_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows).round(4)
    if out.exists():
        new = pd.concat([pd.read_csv(out), new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
