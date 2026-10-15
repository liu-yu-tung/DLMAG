"""Distributions of the training set for the plan features: song length, tempo, meter, key, chords relative to
the key, chord progressions, chord changes per bar, and per track notes per bar, mean duration and pitch.

Key per song: Krumhansl-Kessler profile match on the duration-weighted pitch-class histogram of all tracks.
Relative chords: root as a degree above the key tonic, with minor keys moved to their relative major (so the
minor tonic is vi), so all songs share one frame.

Writes outputs/stats/{song,bar}_features.csv, results/figures/data_distributions.png, results/progressions.csv.
uv run python scripts/plot_data_stats.py   (songs are split over all CPU cores)
"""

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi

from hw2.bonus import chord_token, read_beats
from hw2.remi import DATA_ROOT, HW2_ROOT

TRACKS = ("PIANO", "BRIDGE", "MELODY")
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
DEGREES = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(hist):
    """(tonic, 'maj'|'min') with the highest correlation to the rotated Krumhansl-Kessler profiles."""
    best = max(((np.corrcoef(hist, np.roll(p, t))[0, 1], t, m) for t in range(12) for p, m in ((KK_MAJ, "maj"), (KK_MIN, "min"))))
    return best[1], best[2]


def chord_name(tok, major_tonic):
    """Chord_<root>_<q> -> degree in the relative-major frame, e.g. 'IV', 'vi', 'Vsus'."""
    if tok == "Chord_N":
        return "N"
    root, q = int(tok.split("_")[1]), tok.split("_")[2]
    deg = DEGREES[(root - major_tonic) % 12]
    return {"maj": deg, "min": deg.lower(), "dim": deg.lower() + "o", "aug": deg + "+", "sus": deg + "sus"}[q]


def song_features(d):
    pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
    tr = {i.name: i.notes for i in pm.instruments}
    hist = np.zeros(12)
    for n in (n for t in TRACKS for n in tr.get(t, [])):
        hist[n.pitch % 12] += n.end - n.start
    tonic, mode = estimate_key(hist)
    major_tonic = (tonic + 3) % 12 if mode == "min" else tonic

    chords = []
    for line in (d / "chord_midi.txt").read_text().splitlines():
        a, b, label = line.split()
        chords.append((float(a), float(b), chord_name(chord_token(label), major_tonic)))
    seq = [c for _, _, c in chords if c != "N"]
    seq = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    changes = [(s, c) for i, (s, _, c) in enumerate(chords) if i == 0 or c != chords[i - 1][2]]
    chord_time = Counter()
    for a, b, c in chords:
        chord_time[c] += b - a

    times, orders = read_beats(d / "beat_midi.txt")
    down = [i for i, o in enumerate(orders) if o == 1.0]
    bars = []
    for i0, i1 in zip(down, down[1:]):
        a, b = times[i0], times[i1]
        beat = (b - a) / (i1 - i0)
        row = {"song": d.name, "beats": i1 - i0,
               "chord_changes": sum(a <= s < b for s, c in changes),
               "chord_at_start": next((c for s, e, c in chords if s <= a < e), "N")}
        for t in TRACKS:
            ns = [n for n in tr.get(t, []) if a <= n.start < b]
            row[f"{t}_notes"] = len(ns)
            row[f"{t}_mean_dur"] = np.mean([(n.end - n.start) / beat for n in ns]) if ns else np.nan
            row[f"{t}_mean_pitch"] = np.mean([n.pitch for n in ns]) if ns else np.nan
        bars.append(row)
    beat_iv = np.diff(times)
    song = {"song": d.name, "bars": len(bars), "bpm": 60 / np.median(beat_iv) if len(beat_iv) else np.nan,
            "beats_per_bar": Counter(r["beats"] for r in bars).most_common(1)[0][0] if bars else np.nan,
            "key": NAMES[tonic] + ("m" if mode == "min" else ""), "mode": mode, "n_chord_types": len(set(seq))}
    return song, bars, chord_time, Counter(zip(seq, seq[1:])), Counter(tuple(seq[i:i + 4]) for i in range(len(seq) - 3))


def main():
    songs_dir = sorted(p for p in (DATA_ROOT / "training").iterdir() if p.is_dir())
    songs, bars, chord_time, bigrams, fourgrams = [], [], Counter(), Counter(), Counter()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for s, b, ct, bg, fg in ex.map(song_features, songs_dir, chunksize=8):
            songs.append(s)
            bars += b
            chord_time.update(ct)
            bigrams.update(bg)
            fourgrams.update(fg)
    songs, bars = pd.DataFrame(songs), pd.DataFrame(bars)
    stats_dir = HW2_ROOT / "outputs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    songs.to_csv(stats_dir / "song_features.csv", index=False)
    bars.to_csv(stats_dir / "bar_features.csv", index=False)

    n_bg, n_fg = sum(bigrams.values()), sum(fourgrams.values())
    prog = [{"kind": "2-chord", "progression": " -> ".join(k), "count": v, "share": v / n_bg} for k, v in bigrams.most_common(15)]
    prog += [{"kind": "4-chord", "progression": " -> ".join(k), "count": v, "share": v / n_fg} for k, v in fourgrams.most_common(15)]
    pd.DataFrame(prog).round(4).to_csv(HW2_ROOT / "results" / "progressions.csv", index=False)

    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    ax = ax.ravel()
    ax[0].hist(songs["bars"], bins=30, color="C0")
    ax[0].set(title="Song length", xlabel="bars", ylabel="songs")
    ax[1].hist(songs["bpm"], bins=30, color="C0")
    ax[1].set(title="Tempo (median beat interval)", xlabel="BPM", ylabel="songs")
    bpb = bars["beats"].value_counts().sort_index()
    ax[2].bar(bpb.index.astype(str), bpb.values / bpb.sum(), color="C0")
    ax[2].set(title="Beats per bar (all bars)", xlabel="beats", ylabel="share of bars")
    keys = songs["key"].value_counts()
    order = [n for n in NAMES] + [n + "m" for n in NAMES]
    ax[3].bar(order, [keys.get(k, 0) for k in order], color=["C0"] * 12 + ["C1"] * 12)
    ax[3].set(title="Estimated key (blue major, orange minor)", ylabel="songs")
    ax[3].tick_params(axis="x", rotation=90)
    tot = sum(chord_time.values())
    top = chord_time.most_common(16)
    ax[4].bar([k for k, _ in top], [v / tot for _, v in top], color="C2")
    ax[4].set(title="Chords relative to key (time share)", ylabel="share of song time")
    ax[4].tick_params(axis="x", rotation=90)
    cc = bars["chord_changes"].clip(upper=5).value_counts().sort_index()
    ax[5].bar([str(i) if i < 5 else "5+" for i in cc.index], cc.values / cc.sum(), color="C2")
    ax[5].set(title="Chord changes per bar", xlabel="new chord labels starting in the bar", ylabel="share of bars")
    for t, c in zip(TRACKS, ("C0", "C1", "C3")):
        ax[6].hist(bars[f"{t}_notes"].clip(upper=40), bins=np.arange(0, 42), histtype="step", lw=2, color=c, label=t, density=True)
        ax[7].hist(np.log2(bars[f"{t}_mean_dur"].dropna()), bins=50, histtype="step", lw=2, color=c, label=t, density=True)
        ax[8].hist(bars[f"{t}_mean_pitch"].dropna(), bins=np.arange(30, 100), histtype="step", lw=2, color=c, label=t, density=True)
    ax[6].set(title="Notes per bar (onsets, clipped at 40)", xlabel="notes", ylabel="density")
    ax[7].set(title="Mean note duration per bar", xlabel="beats (log2 scale)", ylabel="density")
    ax[7].set_xticks(range(-2, 4), [f"{2.0 ** k:g}" for k in range(-2, 4)])
    ax[8].set(title="Mean pitch per bar", xlabel="MIDI pitch (60 = middle C)", ylabel="density")
    for a in ax[6:]:
        a.legend()
    fig.suptitle(f"POP909 training set: {len(songs)} songs, {len(bars)} bars")
    fig.tight_layout()
    out = HW2_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "data_distributions.png", dpi=110)

    print(songs[["bars", "bpm", "n_chord_types"]].describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string())
    print("\nbeats per bar share:", (bpb / bpb.sum()).round(3).to_dict())
    print("songs per beats_per_bar:", songs["beats_per_bar"].value_counts().to_dict())
    print("mode share:", songs["mode"].value_counts(normalize=True).round(3).to_dict())
    print("top keys:", keys.head(6).to_dict())
    print("chords (time share):", {k: round(v / tot, 3) for k, v in top})
    print("chord changes per bar share:", (cc / cc.sum()).round(3).to_dict())
    print("\n", pd.DataFrame(prog).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
