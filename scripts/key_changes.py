"""Key changes inside songs: local key per bar by Viterbi over 12 major scales (a minor key counts as its
relative major, the same frame as the relative chords), with a switch penalty.

Per bar cost of scale t = 1 - (share of the bar's sounding time, all tracks, on the 7 notes of t's major scale).
Bars with no notes cost 0 for every scale. A key change costs --penalty, so a new key must fit clearly better
for several bars. Check: share of chord time (chord_midi.txt) whose triad lies in the scale, with one key per
song vs the local key.

Writes outputs/stats/key_segments.csv, results/key_changes.csv, results/figures/key_changes.png.
uv run python scripts/key_changes.py [--penalty 4]   (songs are split over all CPU cores)
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi

from hw2.pop909 import KEY_PENALTY, SCALES, chord_token, key_path, read_beats
from hw2.remi import DATA_ROOT, HW2_ROOT

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def chord_pcs(tok):
    root, q = int(tok.split("_")[1]), tok.split("_")[2]
    third = {"maj": 4, "min": 3, "dim": 3, "aug": 4, "sus": None}[q]
    fifth = {"dim": 6, "aug": 8}.get(q, 7)
    return [root, (root + fifth) % 12] + ([] if third is None else [(root + third) % 12])


def song_keys(d, penalty):
    pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
    notes = [n for i in pm.instruments for n in i.notes]
    times, orders = read_beats(d / "beat_midi.txt")
    down = [times[i] for i, o in enumerate(orders) if o == 1.0]
    edges = list(zip(down, down[1:]))
    hist = np.zeros((len(edges), 12))
    for n in notes:
        for b, (a, e) in enumerate(edges):
            s, f = max(n.start, a), min(n.end, e)
            if f > s:
                hist[b, n.pitch % 12] += f - s
    tot = hist.sum(1, keepdims=True)
    share = np.divide(hist, tot, out=np.zeros_like(hist), where=tot > 0)
    cost = np.where(tot > 0, 1 - share @ SCALES.T, 0.0)
    local = key_path(cost, penalty)
    global_key = int((hist.sum(0) @ SCALES.T).argmax())

    fit = {"global": 0.0, "local": 0.0, "total": 0.0}
    for line in (d / "chord_midi.txt").read_text().splitlines():
        a, e, label = line.split()
        a, e, tok = float(a), float(e), chord_token(label)
        if tok == "Chord_N":
            continue
        b = int(np.clip(np.searchsorted(down, a, side="right") - 1, 0, len(edges) - 1))
        pcs = chord_pcs(tok)
        fit["total"] += e - a
        fit["global"] += (e - a) * all(SCALES[global_key][p] for p in pcs)
        fit["local"] += (e - a) * all(SCALES[local[b]][p] for p in pcs)

    segs, start = [], 0
    for b in range(1, len(local) + 1):
        if b == len(local) or local[b] != local[start]:
            segs.append({"song": d.name, "start_bar": start, "bars": b - start, "key": NAMES[local[start]], "tonic": local[start],
                         "n_bars_song": len(local)})
            start = b
    return segs, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--penalty", type=float, default=KEY_PENALTY)
    args = p.parse_args()
    songs = sorted(p for p in (DATA_ROOT / "training").iterdir() if p.is_dir())
    segs, fit = [], Counter()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for s, f in ex.map(partial(song_keys, penalty=args.penalty), songs, chunksize=8):
            segs += s
            fit.update(f)
    segs = pd.DataFrame(segs)
    out = HW2_ROOT / "outputs" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    segs.to_csv(out / "key_segments.csv", index=False)

    changes = []
    for song, g in segs.groupby("song"):
        g = g.sort_values("start_bar")
        for prev, cur in zip(g.itertuples(), g.iloc[1:].itertuples()):
            iv = (cur.tonic - prev.tonic) % 12
            changes.append({"song": song, "bar": cur.start_bar, "position": cur.start_bar / cur.n_bars_song,
                            "interval": iv - 12 if iv > 6 else iv, "bars_before": prev.bars, "bars_after": cur.bars})
    changes = pd.DataFrame(changes)
    per_song = segs.groupby("song").size() - 1
    n = len(per_song)
    summary = {
        "penalty": args.penalty, "songs": n,
        "songs_with_change": int((per_song > 0).sum()), "share_with_change": (per_song > 0).mean(),
        "changes_total": len(changes),
        "songs_1_change": int((per_song == 1).sum()), "songs_2_changes": int((per_song == 2).sum()), "songs_3plus": int((per_song >= 3).sum()),
        "chord_fit_global": fit["global"] / fit["total"], "chord_fit_local": fit["local"] / fit["total"],
    }
    if len(changes):
        summary["median_position"] = changes["position"].median()
        summary["share_after_70pct"] = (changes["position"] >= 0.7).mean()
        summary["median_bars_after"] = changes["bars_after"].median()
    pd.DataFrame([summary]).round(4).to_csv(HW2_ROOT / "results" / "key_changes.csv", index=False)

    iv_counts = changes["interval"].value_counts().reindex(range(-5, 7), fill_value=0) if len(changes) else pd.Series(0, index=range(-5, 7))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar([str(i) for i in per_song.clip(upper=4).value_counts().sort_index().index], per_song.clip(upper=4).value_counts().sort_index().values)
    ax[0].set(title="Key changes per song (4 = 4+)", xlabel="changes", ylabel="songs")
    ax[1].bar([f"{i:+d}" for i in iv_counts.index], iv_counts.values, color="C1")
    ax[1].set(title="Interval of the change", xlabel="semitones, new tonic minus old", ylabel="changes")
    if len(changes):
        ax[2].hist(changes["position"], bins=20, range=(0, 1), color="C2")
    ax[2].set(title="Where in the song", xlabel="bar of the change / song length", ylabel="changes")
    fig.suptitle(f"Local key by Viterbi over major scales, switch penalty {args.penalty:g}")
    fig.tight_layout()
    (HW2_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(HW2_ROOT / "results" / "figures" / "key_changes.png", dpi=110)

    for k, v in summary.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    if len(changes):
        print("intervals:", iv_counts[iv_counts > 0].to_dict())
        print("bars_after p10/p50/p90:", np.percentile(changes["bars_after"], [10, 50, 90]))


if __name__ == "__main__":
    main()
