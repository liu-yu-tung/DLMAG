"""Does the first theme come back? Intro = bars before the first melody note, outro = bars after the last one; the
body is what lies between. Compares the first 8 body bars (theme) with every other 8-bar window of the body, by
pitch-class content per bar (cosine, bars aligned), for all notes, the melody and the piano. Also a lag profile over
the body: bar i vs bar i+L, by pitch-class cosine and by exact note match (same start and pitch).

uv run python scripts/return_stats.py outputs/plan_fill_v2/gaps --out results/return.csv
"""

import argparse
import json
import sys
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from silence_stats import WINDOW, bar_profile, window_similarity

from hw2.data import split_songs
from hw2.remi import HW2_ROOT, load_sequences, notes_from_tokens

PPB = 16
STEP = 480 // PPB
REPEAT = 0.85
PARTS = ("all", "melody", "piano")
MAX_LAG = 24


def note_sets(notes, bar_starts) -> list[set]:
    sets = [set() for _ in bar_starts]
    starts = np.array(bar_starts)
    for s, _, p, _ in notes:
        b = int(np.searchsorted(starts, s, side="right")) - 1
        if 0 <= b < len(sets):
            sets[b].add(((s - bar_starts[b]) // STEP, p))
    return sets


def lag_rows(pc: dict, sets: dict, lo: int, hi: int, source: str, piece: str) -> list[dict]:
    rows = []
    for part in pc:
        x = pc[part][lo:hi + 1]
        norm = np.linalg.norm(x, axis=1)
        for lag in range(1, min(MAX_LAG, hi - lo) + 1):
            a, b = slice(0, len(x) - lag), slice(lag, len(x))
            ok = (norm[a] > 0) & (norm[b] > 0)
            cos = ((x[a] * x[b]).sum(1)[ok] / (norm[a][ok] * norm[b][ok])) if ok.any() else np.array([])
            inter = sum(len(sets[part][lo + i] & sets[part][lo + i + lag]) for i in range(len(x) - lag))
            union = sum(len(sets[part][lo + i] | sets[part][lo + i + lag]) for i in range(len(x) - lag))
            rows.append({"source": source, "piece": piece, "part": part, "lag": lag, "cosine": cos.mean() if len(cos) else np.nan,
                         "exact": inter / union if union else np.nan, "pairs": int(ok.sum())})
    return rows


def song_row(notes, melody_notes, piano_notes, bar_starts, source, piece) -> tuple[dict | None, list[dict]]:
    by_part = {"all": notes, "melody": melody_notes, "piano": piano_notes}
    pcs = {k: bar_profile(v, bar_starts)["pc"] if v else np.zeros((len(bar_starts), 12)) for k, v in by_part.items()}
    has = pcs["melody"].sum(1) > 0
    if melody_notes is None:
        has = pcs["all"].sum(1) > 0
        pcs = {"all": pcs["all"]}
    if not has.any():
        return None, []
    entry, exit_ = int(np.argmax(has)), len(has) - 1 - int(np.argmax(has[::-1]))
    body = exit_ - entry + 1
    lags = lag_rows(pcs, {k: note_sets(by_part[k], bar_starts) for k in pcs}, entry, exit_, source, piece)
    row = {"source": source, "piece": piece, "bars": len(has), "intro_bars": entry, "outro_bars": len(has) - 1 - exit_, "body_bars": body}
    if body < 3 * WINDOW:
        return row, lags
    theme, last = entry, exit_ - WINDOW + 1
    later = range(theme + WINDOW, last + 1)
    for name, pc in pcs.items():
        sims = np.array([window_similarity(pc, theme, b) for b in later])
        pos = (np.array(later) - theme) / max(1, last - theme)
        row[f"{name}_theme_end"] = window_similarity(pc, theme, last)
        row[f"{name}_best_last16"] = np.nanmax(sims[np.array(later) >= exit_ - 2 * WINDOW + 1])
        mid = sims[np.array(later) < exit_ - 2 * WINDOW + 1]
        row[f"{name}_best_middle"] = np.nanmax(mid) if len(mid) and np.isfinite(mid).any() else np.nan
        row[f"{name}_mean_other"] = np.nanmean(sims)
        row[f"{name}_best_pos"] = pos[np.nanargmax(sims)] if np.isfinite(sims).any() else np.nan
        hits, b = 0, theme + WINDOW
        while b <= last:
            if window_similarity(pc, theme, b) >= REPEAT:
                hits, b = hits + 1, b + WINDOW
            else:
                b += 1
        row[f"{name}_returns"] = hits
    return row, lags


def real(n_val: int, crop_bars: int | None) -> tuple[list[dict], list[dict]]:
    tok_dir = HW2_ROOT / "data" / "tokens"
    seqs = load_sequences(tok_dir / "ppb16" / "sequences.json")
    fill = {s["song_name"]: s for s in json.loads((tok_dir / "fill" / "songs.json").read_text())}
    _, val = split_songs(seqs, n_val)
    val_names = {s["song_name"] for s in val}
    rows, lags = [], []
    for song in seqs:
        name = song["song_name"]
        notes, _, bar_starts = notes_from_tokens(song["tokens"], PPB)
        blocks = fill.get(name, {}).get("blocks", [])
        if len(blocks) != len(bar_starts):
            continue
        streams = {"melody": [], "piano": []}
        for i, b in enumerate(blocks):
            for key, out in streams.items():
                t = b[key]
                for j in range(0, len(t) - 3, 4):
                    start = bar_starts[i] + int(t[j][4:]) * STEP
                    out.append((start, start + int(t[j + 3][4:]) * STEP, int(t[j + 1][6:]), int(t[j + 2][4:])))
        melody, piano = streams["melody"], streams["piano"]
        split = "heldout" if name in val_names else "train"
        if crop_bars:
            cut = bar_starts[crop_bars] if crop_bars < len(bar_starts) else None
            if cut is None:
                continue
            notes = [n for n in notes if n[0] < cut]
            melody = [n for n in melody if n[0] < cut]
            piano = [n for n in piano if n[0] < cut]
            bar_starts = bar_starts[:crop_bars]
        row, lag = song_row(notes, melody, piano, bar_starts, f"real_{split}_" + (f"first{crop_bars}" if crop_bars else "whole_song"), name)
        if row:
            rows.append(row)
            lags += lag
    return rows, lags


def generated(folder: Path) -> tuple[list[dict], list[dict]]:
    rows, lags = [], []
    for song in load_sequences(folder / "sequences.json"):
        notes, _, bar_starts = notes_from_tokens(song["tokens"], PPB)
        midi = folder / f"{song['song_name']}.mid"
        tracks = {i.name.upper(): i for i in miditoolkit.MidiFile(str(midi)).instruments}
        if "MELODY" in tracks:
            melody, piano = ([(n.start, n.end, n.pitch, n.velocity) for n in tracks[t].notes] for t in ("MELODY", "PIANO"))
        else:
            melody, piano = None, None
        row, lag = song_row(notes, melody, piano, bar_starts, str(folder.relative_to(HW2_ROOT)), song["song_name"])
        if row:
            rows.append(row)
            lags += lag
    return rows, lags


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path, help="generated folders with sequences.json; without a MELODY track only all notes count and the body is the whole piece")
    p.add_argument("--n_val", type=int, default=45, help="must match train.py")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "return.csv")
    args = p.parse_args()
    (rows, lags), (rows32, lags32) = real(args.n_val, None), real(args.n_val, 32)
    rows += rows32
    for f in args.folders:
        r, lg = generated(f.resolve())
        rows, lags = rows + r, lags + lg
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_name(args.out.stem + "_pieces.csv"), index=False)
    g = df.groupby("source", sort=False)
    intro = g[["bars", "intro_bars", "outro_bars", "body_bars"]].median().add_suffix("_p50")
    intro["intro_p25"], intro["intro_p75"], intro["intro_p90"] = g["intro_bars"].quantile(.25), g["intro_bars"].quantile(.75), g["intro_bars"].quantile(.9)
    intro["outro_p75"] = g["outro_bars"].quantile(.75)
    intro["with_body_24"] = g["all_theme_end"].count()
    cols = [c for c in df.columns if c.startswith(tuple(f"{part}_" for part in PARTS))]
    sims = g[cols].mean()
    for name in PARTS:
        sims[f"{name}_end_beats_middle"] = g.apply(lambda d: (d[f"{name}_best_last16"] > d[f"{name}_best_middle"])[d[f"{name}_best_middle"].notna()].mean())
        sims[f"{name}_returns_ge1"] = g.apply(lambda d: (d[f"{name}_returns"] >= 1)[d[f"{name}_returns"].notna()].mean())
    out = pd.concat([intro, sims], axis=1).round(3)
    out.to_csv(args.out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("intro / outro / body length in bars:")
        print(intro.round(1).to_string())
        for name in PARTS:
            print(f"\nfirst 8 body bars (theme) vs later 8-bar windows, {name} notes:")
            print(sims[[c for c in sims.columns if c.startswith(name)]].round(3).to_string())
    print(f"-> {args.out}")

    lag = pd.DataFrame(lags)
    prof = lag.groupby(["source", "part", "lag"], sort=False)[["cosine", "exact"]].mean()
    prof["pieces"] = lag.groupby(["source", "part", "lag"], sort=False)["cosine"].count()
    prof.round(4).to_csv(args.out.with_name(args.out.stem + "_lags.csv"))
    show = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for metric in ("cosine", "exact"):
            print(f"\nlag profile inside the body, {metric} (bar i vs bar i+lag):")
            t = prof[metric].unstack("lag")
            print(t[[c for c in show if c in t.columns]].round(3).to_string())
    plot_lags(prof, args.out.parent / "figures" / f"{args.out.stem}_lags.png")


def plot_lags(prof: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sources = list(prof.index.get_level_values("source").unique())
    colors = dict(zip(sources, plt.cm.tab10.colors))
    fig, axes = plt.subplots(2, len(PARTS), figsize=(13, 7), sharex=True)
    for col, part in enumerate(PARTS):
        for row, metric in enumerate(("cosine", "exact")):
            ax = axes[row, col]
            for src in sources:
                if (src, part) not in prof.index.droplevel("lag"):
                    continue
                d = prof.loc[(src, part)]
                d = d[d["pieces"] >= 5]
                ax.plot(d.index, d[metric], "-" if src.startswith("real") else "--", marker=".", color=colors[src],
                        lw=2 if src.startswith("real") else 1.2)
            ax.set_title(f"{part}: {'pitch-class cosine' if metric == 'cosine' else 'exact note match'}")
            ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("lag (bars)")
    handles = [plt.Line2D([], [], color=colors[s], ls="-" if s.startswith("real") else "--", marker=".") for s in sources]
    fig.legend(handles, [s.replace("outputs/", "") for s in sources], loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
