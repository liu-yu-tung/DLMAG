"""Exp 4: how fast a continuation drifts away from its 8-bar prompt.

For each bar after the prompt: cosine similarity of its pitch-class histogram to the prompt's, and
groove similarity (1 - onset XOR distance, as in GS) to the closest prompt bar.
Reference: held-out real songs, 8 bars from a random point followed by their real next 24 bars.

uv run python scripts/exp4_prompt_fade.py outputs/task_b/* --out results/exp4.csv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from hw2.data import split_songs
from hw2.remi import DATA_ROOT, load_sequences, split_bars

PROMPT_BARS = 8


def pc_hist(bars):
    h = np.zeros(12)
    for bar in bars:
        for t in bar:
            if t.startswith("Pitch_"):
                h[int(t[6:]) % 12] += 1
    return h


def onsets(bar, n_pos):
    v = np.zeros(n_pos)
    pos = 0
    for t in bar:
        if t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and pos < n_pos:
            v[pos] = 1
    return v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else np.nan


def curve(bars, n_pos=128):
    prompt, cont = bars[:PROMPT_BARS], bars[PROMPT_BARS:]
    ph = pc_hist(prompt)
    pon = [onsets(b, n_pos) for b in prompt]
    rows = []
    for i, bar in enumerate(cont):
        o = onsets(bar, n_pos)
        rows.append({"bar": PROMPT_BARS + i + 1, "pc_sim": cosine(ph, pc_hist([bar])),
                     "groove_sim": max(1 - np.abs(o - p).sum() / n_pos for p in pon) if o.any() else np.nan})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="*")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_val", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []
    for d in map(Path, args.dirs):
        for piece in load_sequences(d / "sequences.json"):
            for r in curve(split_bars(piece["tokens"])):
                rows.append({"set": d.name, "piece": piece["song_name"], **r})

    _, val = split_songs(load_sequences(DATA_ROOT.parent / "tokens" / "ppb16" / "sequences.json"), args.n_val)
    rng = random.Random(args.seed)
    for s in val:
        bars = split_bars(s["tokens"])
        if len(bars) < 32:
            continue
        start = rng.randrange(len(bars) - 32 + 1)
        for r in curve(bars[start : start + 32]):
            rows.append({"set": "real_heldout", "piece": s["song_name"], **r})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["set", pd.cut(df["bar"], [8, 16, 24, 32])], observed=True)[["pc_sim", "groove_sim"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
