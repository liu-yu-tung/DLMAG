"""Melody against the chords and the piano, in the C-based song, first 32 bars. Per piece: share of melody time on
chord tones (labels are triads, so a chord's 7th counts as a non-chord tone), chord tones at onsets on strong and weak
beats, in-key share, and the share of melody time that rubs against a sounding piano note, by the real distance
(1 = minor second, 11 = major seventh, 13 = minor ninth). Summaries per set: pooled, the per-piece median and the
worst 10%, and the share of pieces past the real songs' worst 10%, since a few rough pieces sound unlike every
piece being slightly rough. Real songs come from the fill token set (held out and training).

uv run python scripts/harmony_stats.py outputs/cand_a/pf_* --out results/harmony.csv
"""

import argparse
import json
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd

from hw2.data import split_songs
from hw2.plan import PPB, chord_tones, signed
from hw2.plan_generate import fill_prefix, note_events, parse_plan
from hw2.remi import HW2_ROOT, TICKS_PER_BEAT

BARS = 32
STEP = TICKS_PER_BEAT // PPB
MAJOR = (0, 2, 4, 5, 7, 9, 11)
CLASHES = {1: "m2", 11: "M7", 13: "m9"}
KINDS = [*CLASHES.values(), "wider"]
REAL = "real_train"
OPENING = 8


def bar_counts(meter: int, prefix: list[str], key: int, melody, piano) -> dict:
    """Melody time in slots (1/PPB beat), clipped to the bar; piano includes notes held over from the bar before."""
    half, end = max(1, meter // 2) * PPB, meter * PPB
    tones = [chord_tones(t) for t in prefix if t.startswith("Chord_")]
    scale = {(key + d) % 12 for d in MAJOR}
    c = {"time": 0, "chord_time": 0, "labelled_time": 0, "key_time": 0, "strong": 0, "strong_chord": 0, "weak": 0,
         "weak_chord": 0, "piano_labelled_time": 0, "piano_chord_time": 0, **{f"{k}_time": 0 for k in KINDS}}
    for pos, pitch, _, dur in melody:
        for t in range(pos, min(pos + max(dur, 1), end)):
            here = tones[int(t >= half)] if len(tones) > int(t >= half) else None
            c["time"] += 1
            c["key_time"] += pitch % 12 in scale
            if here is not None:
                c["labelled_time"] += 1
                c["chord_time"] += pitch % 12 in here
            gaps = [abs(pitch - q) for p, q, _, d in piano if p <= t < p + max(d, 1) and (pitch - q) % 12 in (1, 11)]
            if gaps:
                c[f"{CLASHES.get(min(gaps), 'wider')}_time"] += 1
        here = tones[int(pos >= half)] if len(tones) > int(pos >= half) else None
        if here is not None:
            side = "strong" if pos in (0, half) else "weak"
            c[side] += 1
            c[f"{side}_chord"] += pitch % 12 in here
    for pos, pitch, _, dur in piano:
        for t in range(max(pos, 0), min(pos + max(dur, 1), end)):
            here = tones[int(t >= half)] if len(tones) > int(t >= half) else None
            if here is not None:
                c["piano_labelled_time"] += 1
                c["piano_chord_time"] += pitch % 12 in here
    return c


def piece_counts(source: str, piece: str, meter: int, prefixes: list[list[str]], bars: list[dict]) -> dict:
    total, key, bar_len = {}, 0, meter * PPB
    for i, (prefix, notes) in enumerate(zip(prefixes[:BARS], bars[:BARS])):
        key = next((int(t.split("_")[1]) for t in reversed(prefix) if t.startswith("KeyShift_")), key)
        held = [(p - bar_len, q, v, d) for p, q, v, d in bars[i - 1]["piano"] if p + d > bar_len] if i else []
        for k, v in bar_counts(meter, prefix, key, notes["melody"], notes["piano"] + held).items():
            total[k] = total.get(k, 0) + v
            if i < OPENING:
                total[f"open_{k}"] = total.get(f"open_{k}", 0) + v
    return {"source": source, "piece": piece, **total}


def real(tokens_dir: Path, n_val: int) -> list[dict]:
    train, val = split_songs(json.loads((tokens_dir / "songs.json").read_text()), n_val)
    rows = []
    for name, songs in (("real_heldout", val), (REAL, train)):
        for song in songs:
            meter = int(song["header"][0].split("_")[1])
            bars = [{s: note_events(b[s]) for s in ("melody", "piano")} for b in song["blocks"]]
            rows.append(piece_counts(name, song["song_name"], meter, [b["prefix"] for b in song["blocks"]], bars))
    return rows


def source_of(folder: Path) -> str:
    folder = folder.resolve()
    return str(folder.relative_to(HW2_ROOT)) if folder.is_relative_to(HW2_ROOT) else str(folder)


def generated(folder: Path) -> list[dict]:
    rows = []
    for entry in json.loads((folder / "plans.json").read_text()):
        plan = parse_plan(entry["tokens"])
        shift, bar_ticks = signed(plan.key), plan.meter * TICKS_PER_BEAT
        bars = [{"melody": [], "piano": []} for _ in plan.bars]
        edge = 0
        for inst in miditoolkit.MidiFile(str(folder / f"{entry['song_name']}.mid")).instruments:
            stream = inst.name.lower()
            for n in inst.notes:
                b = n.start // bar_ticks
                edge += n.pitch in (21, 108)
                if stream in ("melody", "piano") and b < len(bars):
                    bars[b][stream].append(((n.start - b * bar_ticks) // STEP, n.pitch - shift, n.velocity, (n.end - n.start) // STEP))
        rows.append({**piece_counts(source_of(folder), entry["song_name"], plan.meter, [fill_prefix(b) for b in plan.bars], bars),
                     "range_edge_notes": edge})
    return rows


def shares(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"chord_time": d["chord_time"] / d["labelled_time"], "in_key_time": d["key_time"] / d["time"],
                        "chord_strong": d["strong_chord"] / d["strong"], "chord_weak": d["weak_chord"] / d["weak"]})
    for k in KINDS:
        out[f"{k}_time"] = d[f"{k}_time"] / d["time"]
    out["harsh_time"] = (d["m2_time"] + d["m9_time"]) / d["time"]
    out["piano_chord_time"] = d["piano_chord_time"] / d["piano_labelled_time"]
    out["chord_time_open"] = d["open_chord_time"] / d["open_labelled_time"].replace(0, np.nan)
    out["piano_chord_time_open"] = d["open_piano_chord_time"] / d["open_piano_labelled_time"].replace(0, np.nan)
    return out


def summary(pieces: pd.DataFrame) -> pd.DataFrame:
    per = pd.concat([pieces[["source", "piece"]], shares(pieces)], axis=1)
    worst_real = {"chord_time": per.loc[per.source == REAL, "chord_time"].quantile(0.1),
                  "harsh_time": per.loc[per.source == REAL, "harsh_time"].quantile(0.9)}
    rows = []
    for source, g in per.groupby("source", sort=False):
        pooled = shares(pieces[pieces.source == source].sum(numeric_only=True).to_frame().T).iloc[0]
        row = {"source": source, "pieces": len(g), **{f"{k}_pooled": v for k, v in pooled.items()}}
        for k in ("chord_time", "harsh_time", "M7_time", "chord_strong", "piano_chord_time", "chord_time_open", "piano_chord_time_open"):
            row[f"{k}_median"] = g[k].median()
        row["chord_time_worst10"] = g["chord_time"].quantile(0.1)
        row["harsh_time_worst10"] = g["harsh_time"].quantile(0.9)
        row["pieces_below_real_chord_worst10"] = (g["chord_time"] < worst_real["chord_time"]).mean()
        row["pieces_above_real_harsh_worst10"] = (g["harsh_time"] > worst_real["harsh_time"]).mean()
        row["range_edge_notes"] = pieces.loc[pieces.source == source, "range_edge_notes"].sum() if "range_edge_notes" in pieces else 0
        rows.append(row)
    return pd.DataFrame(rows), per


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path, help="plan-fill folders with plans.json")
    p.add_argument("--tokens_dir", type=Path, default=HW2_ROOT / "data" / "tokens" / "fill_rep")
    p.add_argument("--n_val", type=int, default=45, help="must match train.py")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "harmony.csv")
    args = p.parse_args()
    rows = real(args.tokens_dir, args.n_val)
    for folder in args.folders:
        rows += generated(folder)
    table, per = summary(pd.DataFrame(rows))
    table.to_csv(args.out, index=False)
    per.to_csv(args.out.with_name(args.out.stem + "_pieces.csv"), index=False)
    cols = ["source", "pieces", "chord_time_median", "chord_time_worst10", "pieces_below_real_chord_worst10", "chord_strong_median",
            "harsh_time_median", "pieces_above_real_harsh_worst10", "M7_time_median", "chord_time_open_median",
            "piano_chord_time_median", "piano_chord_time_open_median", "in_key_time_pooled"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
