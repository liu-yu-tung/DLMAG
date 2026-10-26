"""Does the fill model follow its plan? Per bar: planned chord vs the triad the piano holds longest (each half bar),
planned bins vs the bins of the written notes, planned repeat label vs the label of the written melody.

uv run python scripts/adherence.py outputs/plan_fill_rep/free --out results/adherence.csv
"""

import argparse
import json
from pathlib import Path

import miditoolkit
import numpy as np
import pandas as pd

from hw2.data import split_songs
from hw2.plan import PPB, STAT_NAMES, repeat_kind, repeat_label, signed, stat_bin
from hw2.plan_generate import fill_prefix, note_events, parse_plan
from hw2.remi import HW2_ROOT, TICKS_PER_BEAT

STEP = TICKS_PER_BEAT // PPB
TRIADS = {f"Chord_{r}_{q}": [r, (r + (4 if q == "maj" else 3)) % 12, (r + 7) % 12] for r in range(12) for q in ("maj", "min")}
KINDS = ("exact", "var", "rhythm")


def longest_triad(notes, lo: int, hi: int) -> str | None:
    hist = np.zeros(12)
    for pos, pitch, _, dur in notes:
        s, e = max(pos, lo), min(pos + max(dur, 1), hi)
        if e > s:
            hist[pitch % 12] += e - s
    return max(TRIADS, key=lambda c: hist[TRIADS[c]].sum()) if hist.any() else None


def piece_rows(source: str, piece: str, meter: int, prefixes: list[list[str]], bars: list[dict], edges: dict) -> list[dict]:
    """prefixes: fill prefix per bar (C-based chords); bars: {"melody": [...], "piano": [...]} note events per bar."""
    half, end = max(1, meter // 2) * PPB, meter * PPB
    melody_bars = [b["melody"] for b in bars]
    rows = []
    for i, (prefix, notes) in enumerate(zip(prefixes, bars)):
        row = {"source": source, "piece": piece, "bar": i, "has_melody": bool(notes["melody"])}
        chords = [t for t in prefix if t.startswith("Chord_")]
        for slot, (lo, hi) in enumerate(((0, half), (half, end))):
            if slot < len(chords) and chords[slot] in TRIADS:
                got = longest_triad(notes["piano"], lo, hi)
                row[f"chord_{slot}"] = np.nan if got is None else float(got == chords[slot])
        p, m = notes["piano"], notes["melody"]
        written = [len(p), np.mean([d for *_, d in p]) / PPB if p else 0.0, len(m), np.mean([d for *_, d in m]) / PPB if m else 0.0]
        for name, value in zip(STAT_NAMES, written):
            planned = next((int(t.split("_")[1]) for t in prefix if t.startswith(name + "_")), None)
            if planned is not None:
                row[f"{name}_diff"] = stat_bin(value, edges[name]) - planned
        planned_rep = next((t for t in prefix if t.startswith("Rep_")), None)
        row["rep_written"] = repeat_label(melody_bars, i) if m else None
        if planned_rep:
            row["rep_planned"] = planned_rep
            if planned_rep != "Rep_none" and m:
                _, kind, lag = planned_rep.split("_")
                row["rep_kind_at_lag"] = repeat_kind(m, melody_bars[i - int(lag)]) if i >= int(lag) else None
        rows.append(row)
    return rows


def generated(folder: Path, edges: dict) -> list[dict]:
    source = str(folder.relative_to(HW2_ROOT)) if folder.is_relative_to(HW2_ROOT) else str(folder)
    rows = []
    for entry in json.loads((folder / "plans.json").read_text()):
        plan = parse_plan(entry["tokens"])
        shift, bar_ticks = signed(plan.key), plan.meter * TICKS_PER_BEAT
        bars = [{"melody": [], "piano": []} for _ in plan.bars]
        for inst in miditoolkit.MidiFile(str(folder / f"{entry['song_name']}.mid")).instruments:
            stream = inst.name.lower()
            for n in inst.notes:
                b = n.start // bar_ticks
                if stream in ("melody", "piano") and b < len(bars):
                    bars[b][stream].append(((n.start - b * bar_ticks) // STEP, n.pitch - shift, n.velocity, (n.end - n.start) // STEP))
        rows += piece_rows(source, entry["song_name"], plan.meter, [fill_prefix(b) for b in plan.bars], bars, edges)
    return rows


def real(tokens_dir: Path, n_val: int, edges: dict) -> list[dict]:
    _, val = split_songs(json.loads((tokens_dir / "songs.json").read_text()), n_val)
    rows = []
    for song in val:
        meter = int(song["header"][0].split("_")[1])
        bars = [{s: note_events(b[s]) for s in ("melody", "piano")} for b in song["blocks"]]
        rows += piece_rows("real_heldout", song["song_name"], meter, [b["prefix"] for b in song["blocks"]], bars, edges)
    return rows


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source", sort=False)
    out = pd.DataFrame({"pieces": g["piece"].nunique(), "bars": g.size()})
    slots = pd.concat([df[["source", c]].rename(columns={c: "match"}) for c in ("chord_0", "chord_1") if c in df])
    out["chord_match"] = slots.groupby("source", sort=False)["match"].mean()
    for name in STAT_NAMES:
        col = f"{name}_diff"
        out[f"{name}_exact"] = g[col].apply(lambda d: (d == 0)[d.notna()].mean())
        out[f"{name}_within1"] = g[col].apply(lambda d: (d.abs() <= 1)[d.notna()].mean())
    melody = df[df["has_melody"]]
    kind = melody["rep_written"].str.split("_").str[1]
    for k in ("none",) + KINDS:
        out[f"written_{k}"] = (kind == k).groupby(melody["source"], sort=False).mean()
    if "rep_kind_at_lag" in df:
        asked = melody[melody["rep_planned"].notna() & (melody["rep_planned"] != "Rep_none")]
        gk = asked.groupby("source", sort=False)
        out["asked_repeat_bars"] = gk.size()
        out["asked_label_written"] = gk.apply(lambda d: (d["rep_written"] == d["rep_planned"]).mean())
        out["asked_kind_at_lag"] = gk.apply(lambda d: (d["rep_kind_at_lag"] == d["rep_planned"].str.split("_").str[1]).mean())
        out["asked_any_at_lag"] = gk["rep_kind_at_lag"].apply(lambda s: s.notna().mean())
        none = melody[melody["rep_planned"] == "Rep_none"]
        out["asked_none_written_none"] = (none["rep_written"] == "Rep_none").groupby(none["source"], sort=False).mean()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path, help="plan-fill output folders (plans.json + NN.mid)")
    p.add_argument("--tokens_dir", type=Path, default=HW2_ROOT / "data" / "tokens" / "fill_rep", help="real reference and bins")
    p.add_argument("--n_val", type=int, default=45, help="must match train.py")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "adherence.csv")
    args = p.parse_args()
    edges = json.loads((args.tokens_dir / "bins.json").read_text())
    rows = real(args.tokens_dir, args.n_val, edges)
    for f in args.folders:
        rows += generated(f.resolve(), edges)
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_name(args.out.stem + "_bars.csv"), index=False)
    out = summary(df).round(3)
    out.to_csv(args.out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.T.to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
