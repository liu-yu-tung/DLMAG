"""Silence, melody gaps, 8-bar window stats and start/end similarity per 32-bar piece, generated vs real POP909
(all on the TA tick grid). Windows wrap around: with offset 4, the last window is bars 29-32 then 1-4.

uv run python scripts/silence_stats.py outputs/exp2/loss_0.40 outputs/plan_fill/* --out results/silence.csv
"""

import argparse
import json
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd

from hw2.data import split_songs
from hw2.remi import HW2_ROOT, TICKS_PER_BEAT, load_sequences, notes_from_tokens

PPB = 16
SLOT = TICKS_PER_BEAT // 4
BARS = 32
WINDOW = 8


def runs(mask: np.ndarray) -> list[int]:
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def bar_profile(notes, bar_starts, melody_onsets=None) -> dict:
    """Per bar of the whole song: sounding share of its 16ths, onsets, mean onset velocity, melody onset present
    (NaN without a melody track)."""
    starts = np.array(bar_starts + [bar_starts[-1] + (bar_starts[-1] - bar_starts[-2] if len(bar_starts) > 1 else 4 * TICKS_PER_BEAT)])
    arr = np.array([(s, e, v) for s, e, _, v in notes], int).reshape(-1, 3)
    n_slots = starts[-1] // SLOT + 1
    cover = np.zeros(n_slots + 1, int)
    np.add.at(cover, np.minimum(arr[:, 0] // SLOT, n_slots), 1)
    np.add.at(cover, np.minimum(np.maximum(arr[:, 0] // SLOT + 1, -(-arr[:, 1] // SLOT)), n_slots), -1)
    sounding = np.cumsum(cover)[:n_slots] > 0
    slot_bars = np.searchsorted(starts, np.arange(n_slots) * SLOT, side="right") - 1
    n_bars = len(bar_starts)
    ok = (slot_bars >= 0) & (slot_bars < n_bars)
    sound = np.bincount(slot_bars[ok], sounding[ok], n_bars) / np.maximum(1, np.bincount(slot_bars[ok], minlength=n_bars))
    onset_bars = np.searchsorted(starts, arr[:, 0], side="right") - 1
    keep = (onset_bars >= 0) & (onset_bars < n_bars)
    count = np.bincount(onset_bars[keep], minlength=n_bars).astype(float)
    vel_sum = np.bincount(onset_bars[keep], arr[keep, 2], n_bars)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.where(count > 0, vel_sum / count, np.nan)
    pc = np.zeros((n_bars, 12))
    weight = np.minimum(arr[keep, 1], starts[onset_bars[keep] + 1]) - arr[keep, 0]
    pitches = np.array([p for _, _, p, _ in notes], int)[keep]
    np.add.at(pc, (onset_bars[keep], pitches % 12), weight)
    melody = np.full(n_bars, np.nan)
    if melody_onsets is not None:
        mb = np.searchsorted(starts, np.array(melody_onsets, int), side="right") - 1
        melody = np.bincount(mb[(mb >= 0) & (mb < n_bars)], minlength=n_bars)[:n_bars].astype(bool).astype(float)
    return {"sound": sound, "notes": count, "velocity": velocity, "melody": melody, "pc": pc}


def crop(profile: dict, first_bar: int, bars: int) -> dict:
    return {k: v[first_bar:first_bar + bars] for k, v in profile.items()}


def window_rows(profile: dict, source: str, piece: str) -> list[dict]:
    """One row per offset: mean, weakest window and spread over the 8-bar windows starting at that offset, wrapping
    past the last bar, so every offset covers every bar once."""
    bars = len(profile["sound"])
    rows = []
    for offset in range(WINDOW):
        wins = [np.arange(offset + k * WINDOW, offset + (k + 1) * WINDOW) % bars for k in range(bars // WINDOW)]
        if not wins:
            continue
        row = {"source": source, "piece": piece, "offset": offset, "n_windows": len(wins)}
        for k, v in profile.items():
            if v.ndim > 1:
                continue
            w = np.array([np.nanmean(v[i]) if np.isfinite(v[i]).any() else np.nan for i in wins])
            if np.isnan(w).all():
                continue
            row[f"{k}_mean"], row[f"{k}_min"], row[f"{k}_range"] = np.nanmean(w), np.nanmin(w), np.nanmax(w) - np.nanmin(w)
        rows.append(row)
    return rows


def window_similarity(pc: np.ndarray, a: int, b: int) -> float:
    """Mean cosine of pitch-class content, bar a+i against bar b+i, over 8 bars where both have notes."""
    x, y = pc[a:a + WINDOW], pc[b:b + WINDOW]
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    ok = (nx > 0) & (ny > 0)
    return float(((x[ok] * y[ok]).sum(1) / (nx[ok] * ny[ok])).mean()) if ok.any() else np.nan


def closure(pc: np.ndarray, source: str, piece: str) -> dict:
    """First 8 bars vs last 8 bars, against every other pair of grid windows (plus the last window) in the piece."""
    n = len(pc)
    starts = sorted(set(list(range(0, n - WINDOW + 1, WINDOW)) + [n - WINDOW]))
    first, last = starts[0], starts[-1]
    others = [window_similarity(pc, a, b) for i, a in enumerate(starts) for b in starts[i + 1:]
              if (a, b) != (first, last) and b - a >= WINDOW]
    return {"source": source, "piece": piece, "start_end": window_similarity(pc, first, last),
            "other_pairs": np.nanmean(others) if others else np.nan, "best_other": np.nanmax(others) if others else np.nan}


def piece_stats(notes, bar_starts, melody_onsets=None, first_bar=0, bars=BARS) -> dict:
    lo = bar_starts[first_bar]
    hi = bar_starts[first_bar + bars] if first_bar + bars < len(bar_starts) else max(e for _, e, *_ in notes)
    notes = [(s - lo, e - lo) for s, e, *_ in notes if lo <= s < hi]
    starts = [b - lo for b in bar_starts[first_bar:first_bar + bars]]
    if not notes:
        return {"lead_beats": np.nan}
    n_slots = -(-(hi - lo) // SLOT)
    sounding = np.zeros(n_slots, bool)
    for s, e in notes:
        sounding[s // SLOT:min(n_slots, max(s // SLOT + 1, -(-e // SLOT)))] = True
    first = min(s for s, _ in notes) // SLOT
    last = min(n_slots, -(-max(e for _, e in notes) // SLOT))
    gaps = runs(~sounding[first:last])
    bar_slots = (starts[1] - starts[0]) // SLOT if len(starts) > 1 else 16
    onset_bars = np.searchsorted(starts, [s for s, _ in notes], side="right") - 1
    row = {
        "lead_beats": first * SLOT / TICKS_PER_BEAT,
        "silent_share": float((~sounding[first:last]).mean()),
        "gaps_1beat": sum(g >= 4 for g in gaps),
        "gaps_1bar": sum(g >= bar_slots for g in gaps),
        "longest_gap_beats": max(gaps, default=0) / 4,
        "empty_bars": len(starts) - len(set(onset_bars)),
    }
    if melody_onsets is not None:
        has = np.zeros(len(starts), bool)
        for s in melody_onsets:
            if lo <= s < hi:
                has[np.searchsorted(starts, s - lo, side="right") - 1] = True
        row["melody_bars"] = int(has.sum())
        row["melody_entry_bar"] = int(np.argmax(has)) + 1 if has.any() else np.nan
        inner = has[np.argmax(has):] if has.any() else has
        row["longest_melody_gap_bars"] = max(runs(~inner), default=0)
        row["melody_gaps_2bar"] = sum(g >= 2 for g in runs(~inner))
    return row


def generated(folder: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows, windows, profiles, closures = [], [], [], []
    for song in load_sequences(folder / "sequences.json"):
        notes, _, bar_starts = notes_from_tokens(song["tokens"], PPB)
        first_bar = 8 if "prompt_len" in song else 0
        melody = None
        midi_path = folder / f"{song['song_name']}.mid"
        if midi_path.exists():
            tracks = {i.name.upper(): i for i in miditoolkit.MidiFile(str(midi_path)).instruments}
            if "MELODY" in tracks:
                melody = [n.start for n in tracks["MELODY"].notes]
        bars = min(BARS, len(bar_starts) - first_bar)
        source = str(folder.relative_to(HW2_ROOT))
        rows.append({"source": source, "piece": song["song_name"], **piece_stats(notes, bar_starts, melody, first_bar, bars)})
        prof = crop(bar_profile(notes, bar_starts, melody), first_bar, bars)
        windows += window_rows(prof, source, song["song_name"])
        profiles.append({"source": source, **{f"{k}_{i + 1}": prof[k][i] for k in ("sound", "notes") for i in range(bars)}})
        closures.append(closure(prof["pc"], source, song["song_name"]))
    return rows, windows, profiles, closures


def real(n_val: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    tok_dir = HW2_ROOT / "data" / "tokens"
    seqs = {s["song_name"]: s for s in load_sequences(tok_dir / "ppb16" / "sequences.json")}
    fill = {s["song_name"]: s for s in json.loads((tok_dir / "fill" / "songs.json").read_text())}
    _, val = split_songs(list(seqs.values()), n_val)
    val_names = {s["song_name"] for s in val}
    rows, windows, profiles, closures = [], [], [], []
    for name, song in seqs.items():
        notes, _, bar_starts = notes_from_tokens(song["tokens"], PPB)
        melody = None
        if name in fill and len(fill[name]["blocks"]) == len(bar_starts):
            melody = [bar_starts[i] for i, b in enumerate(fill[name]["blocks"]) if b["melody"]]
        song_profile = bar_profile(notes, bar_starts, melody)
        split = "heldout" if name in val_names else "train"
        closures.append(closure(song_profile["pc"], f"real_{split}_whole_song", name))
        split = "heldout" if name in val_names else "train"
        for first_bar, kind in [(0, "start")] + [(b, "inside") for b in range(8, len(bar_starts) - BARS, 8)]:
            if first_bar + BARS > len(bar_starts):
                continue
            source, piece = f"real_{split}_{kind}", f"{name}@{first_bar}"
            rows.append({"source": source, "piece": piece, **piece_stats(notes, bar_starts, melody, first_bar)})
            prof = crop(song_profile, first_bar, BARS)
            windows += window_rows(prof, source, piece)
            profiles.append({"source": source, **{f"{k}_{i + 1}": prof[k][i] for k in ("sound", "notes") for i in range(BARS)}})
            closures.append(closure(prof["pc"], source, piece))
    return rows, windows, profiles, closures


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path, help="generated folders with sequences.json (TASK B: the 8 prompt bars are skipped)")
    p.add_argument("--n_val", type=int, default=45, help="must match train.py")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "silence.csv")
    args = p.parse_args()
    rows, windows, profiles, closures = real(args.n_val)
    for f in args.folders:
        r, w, pr, c = generated(f.resolve())
        rows, windows, profiles, closures = rows + r, windows + w, profiles + pr, closures + c
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_name(args.out.stem + "_pieces.csv"), index=False)
    agg = df.groupby("source", sort=False).agg(
        n=("piece", "size"),
        lead_beats_p50=("lead_beats", "median"),
        lead_over_1bar=("lead_beats", lambda x: (x >= 4).mean()),
        silent_share=("silent_share", "mean"),
        gaps_1beat=("gaps_1beat", "mean"),
        gaps_1bar=("gaps_1bar", "mean"),
        longest_gap_beats=("longest_gap_beats", "mean"),
        empty_bars=("empty_bars", "mean"),
        melody_entry_bar_p50=("melody_entry_bar", "median"),
        melody_bars=("melody_bars", "mean"),
        longest_melody_gap_bars=("longest_melody_gap_bars", "mean"),
        melody_gaps_2bar=("melody_gaps_2bar", "mean"),
    ).round(3)
    agg.to_csv(args.out)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(agg.to_string())
    print(f"-> {args.out}")

    win = pd.DataFrame(windows)
    cols = [c for c in win.columns if c.endswith(("_mean", "_min", "_range"))]
    by_offset = win.groupby(["source", "offset"], sort=False)[cols].mean().round(3)
    by_offset.insert(0, "n_pieces", win.groupby(["source", "offset"], sort=False).size())
    by_offset.to_csv(args.out.with_name(args.out.stem + "_windows.csv"))
    ties = win.groupby(["source", "piece"])["notes_range"].transform(lambda x: (x == x.max()).sum()) > 1
    best = win[~ties].loc[lambda d: d.groupby(["source", "piece"])["notes_range"].idxmax()].groupby("source", sort=False)["offset"]
    print("\n8-bar windows, offset 0 (windows start at bars 1, 9, 17, 25):")
    show = ["sound_mean", "sound_min", "sound_range", "notes_mean", "notes_min", "notes_range", "velocity_mean", "velocity_range", "melody_min", "melody_range"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(by_offset.xs(0, level="offset")[show].to_string())
        print("\nby offset, sound_range (contrast between 8-bar windows) and sound_min (weakest window):")
        print(by_offset["sound_range"].unstack("offset").round(3).to_string())
        print(by_offset["sound_min"].unstack("offset").round(3).to_string())
        print("\noffset with the largest notes-per-bar contrast per piece (share of pieces; tied pieces left out):")
        print(best.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())

    prof = pd.DataFrame(profiles).groupby("source", sort=False).mean()
    prof.round(3).to_csv(args.out.with_name(args.out.stem + "_bars.csv"))
    clo = pd.DataFrame(closures)
    clo.to_csv(args.out.with_name(args.out.stem + "_closure.csv"), index=False)
    summary = clo.groupby("source", sort=False).agg(n=("piece", "size"), start_end=("start_end", "mean"), other_pairs=("other_pairs", "mean"),
                                                    best_other=("best_other", "mean"))
    summary["start_end_above_others"] = clo.assign(a=clo["start_end"] > clo["other_pairs"]).groupby("source", sort=False)["a"].mean()
    summary["start_end_is_best"] = clo.assign(a=clo["start_end"] >= clo["best_other"]).groupby("source", sort=False)["a"].mean()
    print("\nstart/end similarity (bars 1-8 vs last 8 bars) against other 8-bar window pairs, pitch-class cosine per bar:")
    print(summary.round(3).to_string())

    for k, label in (("notes", "notes per bar"), ("sound", "sounding share")):
        first = pd.DataFrame({p: prof[f"{k}_{p}"] for p in range(1, WINDOW + 1)})
        later = pd.DataFrame({p: prof[[f"{k}_{b}" for b in range(WINDOW + p, BARS + 1, WINDOW) if f"{k}_{b}" in prof]].mean(axis=1)
                              for p in range(1, WINDOW + 1)})
        with pd.option_context("display.width", 250):
            print(f"\n{label} by bar position: bars 1-8 | mean over bars 9-16, 17-24, 25-32 at the same position")
            print(pd.concat({"bars 1-8": first, "bars 9-32": later}, axis=1).round(1 if k == "notes" else 3).to_string())


if __name__ == "__main__":
    main()
