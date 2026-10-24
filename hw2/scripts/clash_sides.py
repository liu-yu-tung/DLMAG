"""Semitone clashes (melody against a sounding piano note a minor 2nd or minor 9th away), split by which note is off
the chord label at that slot: the melody's, the piano's, or both (two triad tones are never a semitone apart). Shares
of melody time, first 32 bars, C-based song; real songs from the fill token set.

uv run python scripts/clash_sides.py outputs/sweep/fill_rep_60k/last outputs/cand_a/pf_fillrel60k
"""

import argparse
import json
from pathlib import Path

import miditoolkit
import pandas as pd

from harmony_stats import BARS, STEP
from hw2.data import split_songs
from hw2.plan import PPB, chord_tones, signed
from hw2.plan_generate import fill_prefix, note_events, parse_plan
from hw2.remi import HW2_ROOT, TICKS_PER_BEAT

KINDS = ["melody_off", "piano_off", "both_off"]


def clash_counts(meter, prefixes, bars):
    half, end = max(1, meter // 2) * PPB, meter * PPB
    c = {"melody_time": 0, "clash": 0, **{k: 0 for k in KINDS}}
    for i, (prefix, notes) in enumerate(zip(prefixes[:BARS], bars[:BARS])):
        tones = [chord_tones(t) for t in prefix if t.startswith("Chord_")]
        held = [(p - end, q, v, d) for p, q, v, d in bars[i - 1]["piano"] if p + d > end] if i else []
        piano = notes["piano"] + held
        for pos, pitch, _, dur in notes["melody"]:
            for t in range(pos, min(pos + max(dur, 1), end)):
                here = tones[int(t >= half)] if len(tones) > int(t >= half) else None
                c["melody_time"] += 1
                hits = [q for p, q, _, d in piano if p <= t < p + max(d, 1) and abs(pitch - q) in (1, 13)]
                if not hits or here is None:
                    continue
                c["clash"] += 1
                melody_off, piano_off = pitch % 12 not in here, any(q % 12 not in here for q in hits)
                assert melody_off or piano_off, "two triad tones a semitone apart"
                c["both_off" if melody_off and piano_off else "melody_off" if melody_off else "piano_off"] += 1
    return c


def generated(folder: Path) -> list[dict]:
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
        rows.append({"source": str(folder), "piece": entry["song_name"], **clash_counts(plan.meter, [fill_prefix(b) for b in plan.bars], bars)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path, help="plan-fill folders with plans.json")
    p.add_argument("--tokens_dir", type=Path, default=HW2_ROOT / "data" / "tokens" / "fill_rep")
    p.add_argument("--n_val", type=int, default=45, help="must match train.py")
    p.add_argument("--n_real", type=int, default=300, help="training songs to measure")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "clash_sides.csv")
    args = p.parse_args()
    train, _ = split_songs(json.loads((args.tokens_dir / "songs.json").read_text()), args.n_val)
    rows = []
    for song in train[: args.n_real]:
        meter = int(song["header"][0].split("_")[1])
        bars = [{s: note_events(b[s]) for s in ("melody", "piano")} for b in song["blocks"]]
        rows.append({"source": "real_train", "piece": song["song_name"], **clash_counts(meter, [b["prefix"] for b in song["blocks"]], bars)})
    for folder in args.folders:
        rows += generated(folder)
    d = pd.DataFrame(rows)
    tot = d.groupby("source", sort=False)[["melody_time", "clash", *KINDS]].sum()
    table = pd.DataFrame({"pieces": d.groupby("source", sort=False).size(), "clash_time": tot["clash"] / tot["melody_time"]})
    for k in KINDS:
        table[f"{k}_time"] = tot[k] / tot["melody_time"]
    table.reset_index().to_csv(args.out, index=False)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
