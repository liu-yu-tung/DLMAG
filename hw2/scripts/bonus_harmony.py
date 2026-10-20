"""BONUS TASK: melody against chords read from the piano (harmony_stats.py --chords_from_piano), real piano vs
generated piano under the same prompt melodies, both on the prompt's beat grid. Generated folders hold one
subfolder per seed.

uv run python scripts/bonus_harmony.py outputs/bonus_ablation/melody/t1_p0.95_free
"""

import argparse
from pathlib import Path

import pandas as pd
import pretty_midi

from harmony_stats import piece_counts, shares
from hw2.pop909 import TA, bar_events, read_beats
from hw2.remi import DATA_ROOT, HW2_ROOT

PPB = 16


def prompt_bars(melody_notes, piano_notes, beat_path):
    beat_times, orders = read_beats(beat_path)
    info = TA.get_beat_info(orders)
    down = [i for i, o in enumerate(orders) if o == 1.0]
    first = info[down[0]][0]
    melody, piano = bar_events(melody_notes, beat_times, info, PPB), bar_events(piano_notes, beat_times, info, PPB)
    bars = [{"melody": melody.get(first + b, []), "piano": piano.get(first + b, [])} for b in range(len(down) - 1)]
    return int(max(orders)), bars


def track_notes(midi, name):
    return next(i.notes for i in midi.instruments if i.name.upper() == name)


def counts(source, piece, meter, bars):
    return piece_counts(source, piece, meter, [[] for _ in bars], bars, chords_from_piano=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--prompt_dir", type=Path, default=DATA_ROOT / "prompts" / "bonus")
    p.add_argument("--real_pieces", type=Path, default=HW2_ROOT / "results" / "harmony_piano_chords_pieces.csv",
                   help="harmony_stats.py --chords_from_piano per-piece table, for the real worst-10%% line")
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "bonus_harmony.csv")
    args = p.parse_args()
    rows = []
    for d in sorted(args.prompt_dir.iterdir()):
        melody = pretty_midi.PrettyMIDI(str(d / "melody.mid")).instruments[0].notes
        piano = pretty_midi.PrettyMIDI(str(d / "piano.mid")).instruments[0].notes
        rows.append(counts("real_piano", d.name, *prompt_bars(melody, piano, d / "beat_midi.txt")))
        for folder in args.folders:
            for seed in sorted(s for s in folder.iterdir() if s.is_dir()):
                midi = pretty_midi.PrettyMIDI(str(seed / f"{d.name}.mid"))
                bars = prompt_bars(track_notes(midi, "MELODY"), track_notes(midi, "PIANO"), d / "beat_midi.txt")
                rows.append(counts(str(folder), f"{d.name}_{seed.name}", *bars))
    raw = pd.DataFrame(rows)
    per = pd.concat([raw[["source", "piece"]], shares(raw)], axis=1)
    line = pd.read_csv(args.real_pieces).query("source == 'real_train'")["chord_time"].quantile(0.1)
    table = per.groupby("source", sort=False).agg(
        pieces=("piece", "size"), melody_on_piano_chords_median=("chord_time", "median"),
        piano_on_own_chords_median=("piano_chord_time", "median"), m2_time_median=("m2_time", "median"),
        harsh_time_median=("harsh_time", "median"))
    table["pieces_below_real_chord_worst10"] = per.assign(below=per.chord_time < line).groupby("source", sort=False)["below"].mean()
    table["real_train_line"] = line
    table.reset_index().to_csv(args.out, index=False)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
