"""Reference points for eval_bonus.py: what CA / OA look like for trivial accompaniments.

- gt: the ground-truth PIANO track (CA should equal CA-ceiling, OA should be 1)
- melody_8vb: the melody itself, an octave lower, as the PIANO track
- other_song: the ground-truth PIANO of the next prompt, bar-aligned by rescaling its bars to this prompt's bars

uv run python scripts/bonus_baselines.py
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi

from hw2.remi import DATA_ROOT, HW2_ROOT


def downbeat_times(beat_file):
    rows = [line.split() for line in Path(beat_file).read_text().splitlines() if line.split()]
    return np.array([float(r[0]) for r in rows if float(r[2]) == 1.0])


def write(path, melody, piano_notes):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody.name = "MELODY"
    piano = pretty_midi.Instrument(program=0, name="PIANO")
    piano.notes = piano_notes
    midi.instruments += [melody, piano]
    midi.write(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_dir", type=str, default=str(DATA_ROOT / "prompts" / "bonus"))
    p.add_argument("--out_dir", type=str, default=str(HW2_ROOT / "outputs" / "bonus_baselines"))
    args = p.parse_args()
    prompts = sorted(d for d in Path(args.prompt_dir).iterdir() if d.is_dir())
    out = Path(args.out_dir)

    for i, d in enumerate(prompts):
        melody = pretty_midi.PrettyMIDI(str(d / "melody.mid")).instruments[0]
        gt = pretty_midi.PrettyMIDI(str(d / "piano.mid")).instruments[0].notes
        (out / "gt").mkdir(parents=True, exist_ok=True)
        write(out / "gt" / f"{d.name}.mid", melody, gt)

        low = [pretty_midi.Note(n.velocity, max(21, n.pitch - 12), n.start, n.end) for n in melody.notes]
        (out / "melody_8vb").mkdir(parents=True, exist_ok=True)
        write(out / "melody_8vb" / f"{d.name}.mid", pretty_midi.PrettyMIDI(str(d / "melody.mid")).instruments[0], low)

        other = prompts[(i + 1) % len(prompts)]
        src_db, dst_db = downbeat_times(other / "beat_midi.txt"), downbeat_times(d / "beat_midi.txt")
        n_bars = min(len(src_db), len(dst_db))

        def remap(t):
            return float(np.interp(t, src_db[:n_bars], dst_db[:n_bars]))

        moved = [
            pretty_midi.Note(n.velocity, n.pitch, remap(n.start), remap(n.end))
            for n in pretty_midi.PrettyMIDI(str(other / "piano.mid")).instruments[0].notes
            if n.start < src_db[n_bars - 1]
        ]
        (out / "other_song").mkdir(parents=True, exist_ok=True)
        write(out / "other_song" / f"{d.name}.mid", pretty_midi.PrettyMIDI(str(d / "melody.mid")).instruments[0], moved)

    ta = HW2_ROOT / "scripts" / "ta"
    for name in ["gt", "melody_8vb", "other_song"]:
        print(f"== {name}")
        res = subprocess.run(
            ["python", "eval_bonus.py", "--prompt_dir", str(Path(args.prompt_dir).resolve()), "--gen_midi_dir", str((out / name).resolve()),
             "--output_csv", str((out / f"{name}_metrics.csv").resolve())],
            cwd=ta, capture_output=True, text=True, check=True,
        )
        print("\n".join(line for line in res.stdout.splitlines() if line.startswith("Average")))


if __name__ == "__main__":
    main()
