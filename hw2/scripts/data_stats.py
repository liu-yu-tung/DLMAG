"""Training-set statistics behind the plan/fill pipeline design (see results/findings.md).

- chord_from_notes.csv: for each maj/min segment of chord_midi.txt, the triad that sounds longest (dur) or is
  pressed most (count) in the given tracks, compared with the label; weighted by segment time. Uses the true
  chord boundaries, so it is an upper bound for this rule.
- track_stats.csv: per bar (downbeat to downbeat of beat_midi.txt) and track, notes with onset in the bar and
  their mean duration in beats (total sounding time / presses).
- token_budget.csv: tokens per bar, song and 32-bar window (stride 8) of each section of the bonus_chd tokens.

uv run python scripts/data_stats.py   (songs are split over all CPU cores)
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from hw2.bonus import chord_token, read_beats
from hw2.remi import DATA_ROOT, HW2_ROOT

TRACKS = ("PIANO", "BRIDGE", "MELODY")
SOURCES = {"all": TRACKS, "piano+bridge": ("PIANO", "BRIDGE"), "piano": ("PIANO",), "bridge": ("BRIDGE",), "melody": ("MELODY",)}
TRIADS = {(r, q): [r, (r + (4 if q == "maj" else 3)) % 12, (r + 7) % 12] for r in range(12) for q in ("maj", "min")}


def pc_hist(notes, a, b, weight):
    h = np.zeros(12)
    for n in notes:
        s, e = max(n.start, a), min(n.end, b)
        if e > s:
            h[n.pitch % 12] += (e - s) if weight == "dur" else 1.0
    return h


def song_stats(d):
    """Chord-match sums per (source, weight) and per-bar track rows for one song folder."""
    totals = {(s, w): np.zeros(4) for s in SOURCES for w in ("dur", "count")}
    bars = []
    pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
    tr = {i.name: i.notes for i in pm.instruments}
    for line in (d / "chord_midi.txt").read_text().splitlines():
        a, b, label = line.split()
        a, b = float(a), float(b)
        tok = chord_token(label)
        if not tok.endswith(("_maj", "_min")):
            continue
        root, q = int(tok.split("_")[1]), tok.split("_")[2]
        for s, names in SOURCES.items():
            notes = [n for k in names for n in tr.get(k, [])]
            for w in ("dur", "count"):
                h = pc_hist(notes, a, b, w)
                if h.sum():
                    guess = max(TRIADS, key=lambda k: h[TRIADS[k]].sum())
                    t = b - a
                    totals[(s, w)] += [t, t * (guess == (root, q)), t * (int(np.argmax(h)) == root), t * (root in np.argsort(h)[-3:])]
    times, orders = read_beats(d / "beat_midi.txt")
    down = [i for i, o in enumerate(orders) if o == 1.0]
    for i0, i1 in zip(down, down[1:]):
        a, b = times[i0], times[i1]
        beat = (b - a) / (i1 - i0)
        for t in TRACKS:
            ns = [n for n in tr.get(t, []) if a <= n.start < b]
            bars.append({"song": d.name, "track": t, "notes": len(ns),
                         "mean_dur_beats": np.mean([(n.end - n.start) / beat for n in ns]) if ns else np.nan})
    return totals, bars


def chord_and_track_stats(train_dir, workers):
    totals = {(s, w): np.zeros(4) for s in SOURCES for w in ("dur", "count")}
    bars = []
    songs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    with ProcessPoolExecutor(workers) as ex:
        for song_acc, song_bars in ex.map(song_stats, songs, chunksize=8):
            for k, v in song_acc.items():
                totals[k] += v
            bars += song_bars
    rows = [{"notes_from": s, "weight": w, "triad_match": v[1] / v[0], "longest_is_root": v[2] / v[0], "root_in_top3": v[3] / v[0],
             "seconds": v[0]} for (s, w), v in totals.items()]
    bars = pd.DataFrame(bars)
    stats = []
    for t, g in bars.groupby("track", sort=False):
        n, m = g["notes"], g["mean_dur_beats"].dropna()
        stats.append({"track": t, "bars": len(g), "empty_share": (n == 0).mean(),
                      **{f"notes_p{p}": np.percentile(n, p) for p in (10, 50, 90)},
                      **{f"mean_dur_beats_p{p}": np.percentile(m, p) for p in (10, 50, 90)}})
    return pd.DataFrame(rows), pd.DataFrame(stats)


def token_budget(seq_path, ctx=2048):
    songs = json.loads(Path(seq_path).read_text())
    per_bar, per_song, per_win = {k: [] for k in ("plan", "melody", "piano")}, {k: [] for k in ("plan", "melody", "piano", "all")}, {k: [] for k in ("plan", "melody", "piano", "all")}
    for s in songs:
        bars, sec = [], None
        for t in s["tokens"][1:]:
            if t == "Bar":
                bars.append({"plan": 1, "melody": 0, "piano": 0})
                sec = "plan"
                continue
            sec = {"Mel": "melody", "Acc": "piano"}.get(t, sec)
            bars[-1][sec] += 1
        for k in per_bar:
            per_bar[k] += [b[k] for b in bars]
        tot = {k: sum(b[k] for b in bars) for k in ("plan", "melody", "piano")}
        for k, v in {**tot, "all": sum(tot.values())}.items():
            per_song[k].append(v)
        for i in range(0, len(bars) - 31, 8):
            win = {k: sum(b[k] for b in bars[i:i + 32]) for k in ("plan", "melody", "piano")}
            for k, v in {**win, "all": sum(win.values())}.items():
                per_win[k].append(v)
    rows = []
    for scope, d in (("bar", per_bar), ("song", per_song), ("window32", per_win)):
        for k, v in d.items():
            rows.append({"scope": scope, "section": k, "p50": np.percentile(v, 50), "p90": np.percentile(v, 90),
                         "p99": np.percentile(v, 99), "max": max(v), "share_over_ctx": np.mean(np.array(v) > ctx)})
    return pd.DataFrame(rows)


def main():
    out = HW2_ROOT / "results"
    chords, tracks = chord_and_track_stats(DATA_ROOT / "training", os.cpu_count())
    budget = token_budget(DATA_ROOT.parent / "tokens" / "bonus_chd" / "sequences.json")
    for name, df in (("chord_from_notes", chords), ("track_stats", tracks), ("token_budget", budget)):
        df.round(4).to_csv(out / f"{name}.csv", index=False)
        print(name, "\n", df.round(3).to_string(index=False), "\n", flush=True)


if __name__ == "__main__":
    main()
