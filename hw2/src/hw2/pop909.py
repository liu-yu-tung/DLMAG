"""POP909 readers on the TA tokenizer's grid, the chord label rule, and the local-key Viterbi. Shared by the bonus and
plan-fill tokenizers (hw2.bonus, hw2.plan) and the dataset analyses in scripts/."""

import importlib.util
from pathlib import Path

import numpy as np

from hw2.remi import HW2_ROOT

CHORD_QUALITIES = ["maj", "min", "dim", "aug", "sus"]
ROOTS = {"C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6,
         "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}
# Major scales, one row per tonic; a minor key counts as its relative major (scripts/key_changes.py checks the fit).
SCALES = np.array([[(pc - t) % 12 in (0, 2, 4, 5, 7, 9, 11) for pc in range(12)] for t in range(12)], dtype=float)
KEY_PENALTY = 4.0


def load_ta():
    spec = importlib.util.spec_from_file_location("ta_tokenization", HW2_ROOT / "scripts" / "ta" / "tokenization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TA = load_ta()


def read_beats(path: Path):
    times, orders = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts:
            times.append(round(float(parts[0]), 6))
            orders.append(float(parts[2]) if len(parts) > 2 else 0.0)
    return times, orders


def track(midi, name):
    return next((i for i in midi.instruments if i.name.strip().upper() == name), None)


def bar_events(notes, beat_times, beat_info, ppb):
    """{bar: [(pos, pitch, vel, dur), ...]} with the TA tokenizer's grid, velocity and duration rules."""
    max_pos = 8 * ppb
    bars = {}
    for n in notes:
        bar, pos, b_len = TA.time_to_grid(n.start, beat_times, beat_info, ppb)
        if pos >= max_pos:
            continue
        dur = max(1, min(max_pos * 2, int(((n.end - n.start) / b_len) * ppb)))
        vel = max(1, n.velocity // 4 * 4 + 1)
        bars.setdefault(bar, []).append((pos, n.pitch, vel, dur))
    return {b: sorted(ev) for b, ev in bars.items()}


def note_tokens(events, pitch_tag: str = "Pitch"):
    out = []
    for pos, pitch, vel, dur in events:
        out += [f"Pos_{pos}", f"{pitch_tag}_{pitch}", f"Vel_{vel}", f"Dur_{dur}"]
    return out


def chord_token(label: str) -> str:
    """POP909 label (e.g. C:min7, Ab:maj/3, D:sus4(b7), N) -> Chord_<root>_<maj|min|dim|aug|sus> or Chord_N."""
    if label == "N" or ":" not in label:
        return "Chord_N"
    root, quality = label.split(":", 1)
    if root not in ROOTS:
        return "Chord_N"
    quality = quality.split("/")[0].split("(")[0]
    if "sus" in quality:
        q = "sus"
    elif quality.startswith(("dim", "hdim")):
        q = "dim"
    elif quality.startswith("aug"):
        q = "aug"
    elif quality.startswith("min"):
        q = "min"
    else:
        q = "maj"
    return f"Chord_{ROOTS[root]}_{q}"


def key_path(cost: np.ndarray, penalty: float = KEY_PENALTY) -> list[int]:
    """Cheapest key path through per-bar costs (bars x 12) when a change costs `penalty`."""
    best_cost, back = cost[0].copy(), np.zeros((len(cost), 12), dtype=int)
    for b in range(1, len(cost)):
        move = best_cost.min() + penalty
        back[b] = np.where(best_cost <= move, np.arange(12), best_cost.argmin())
        best_cost = np.minimum(best_cost, move) + cost[b]
    path = [int(best_cost.argmin())]
    for b in range(len(cost) - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    return path[::-1]
