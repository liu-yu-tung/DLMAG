"""BONUS TASK tokens: bar-interleaved MELODY/PIANO, with the ablation variants. Token layout in docs/bonus.md."""

import json
from pathlib import Path

import numpy as np
import pretty_midi

from hw2.pop909 import CHORD_QUALITIES, TA, bar_events, chord_token, note_tokens, read_beats, track

SPECIAL = ["Mel", "Acc"]
CHORD_TOKENS = ["Chd", "Chord_N"] + [f"Chord_{r}_{q}" for r in range(12) for q in CHORD_QUALITIES]


def chord_bar_events(chord_path, beat_times, beat_info, ppb):
    """{bar: [(pos, Chord_*), ...]}, with the chord sounding at each bar start repeated at Pos_0."""
    starts = {}
    for line in Path(chord_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            bar, pos, _ = TA.time_to_grid(float(parts[0]), beat_times, beat_info, ppb)
            if pos < 8 * ppb:  # same 8-beat limit as notes; a dropped change leaves the previous chord in place
                starts.setdefault(bar, []).append((pos, chord_token(parts[2])))
    out, current = {}, None
    last_bar = max(b for b, _ in beat_info) if beat_info else -1
    for b in range(min(starts, default=0), last_bar + 1):
        events = sorted(starts.get(b, []))
        if current is not None and (not events or events[0][0] > 0):
            events = [(0, current)] + events
        if events:
            out[b] = events
            current = events[-1][1]
    return out


def chord_tokens(events):
    out = []
    for pos, tok in events:
        out += [f"Pos_{pos}", tok]
    return out


def prelude(melody_bars: list[list[str]], variant: dict) -> list[str]:
    """Tokens right after <BOS>; melody_bars[b] = melody note tokens of bar b."""
    if variant.get("lookahead"):
        return ["Mel"] + (melody_bars[0] if melody_bars else [])
    return []


def bar_prefix(b: int, melody_bars: list[list[str]], chord_bars: list[list[str]] | None, variant: dict) -> list[str]:
    """Forced part of bar b: Bar [Chd chords] Mel <melody of b, or b+1 with lookahead> Acc."""
    out = ["Bar"]
    if variant.get("chords"):
        out += ["Chd"] + (chord_bars[b] if chord_bars and b < len(chord_bars) else [])
    m = b + 1 if variant.get("lookahead") else b
    out += ["Mel"] + (melody_bars[m] if m < len(melody_bars) else []) + ["Acc"]
    return out


def interleave(melody_bars, piano_bars, first_bar, last_bar, variant=None, chord_events=None):
    variant = variant or {}
    bars = range(first_bar, last_bar + 1)
    melody = [note_tokens(melody_bars.get(b, [])) for b in bars]
    chords = [chord_tokens(chord_events.get(b, [])) for b in bars] if chord_events is not None else None
    tokens = ["<BOS>"] + prelude(melody, variant)
    for i, b in enumerate(bars):
        tokens += bar_prefix(i, melody, chords, variant) + note_tokens(piano_bars.get(b, []))
    return tokens + ["<EOS>"]


def tokenize_song(midi_path, beat_path, ppb, variant=None, chord_path=None):
    variant = variant or {}
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    beat_times, orders = read_beats(beat_path)
    info = TA.get_beat_info(orders)
    melody_track, piano_track = track(midi, "MELODY"), track(midi, "PIANO")
    if melody_track is None or piano_track is None:
        return None
    melody_bars = bar_events(melody_track.notes, beat_times, info, ppb)
    piano_bars = bar_events(piano_track.notes, beat_times, info, ppb)
    used = set(melody_bars) | set(piano_bars)
    if not used:
        return None
    chords = chord_bar_events(chord_path, beat_times, info, ppb) if variant.get("chords") else None
    return interleave(melody_bars, piano_bars, min(used), max(used), variant, chords)


def tokenize_prompt(melody_path, beat_path, ppb, chord_path=None):
    """Melody (and chord) tokens per bar of a bonus prompt, numbered from the first downbeat (as eval_bonus.py does)."""
    midi = pretty_midi.PrettyMIDI(str(melody_path))
    beat_times, orders = read_beats(beat_path)
    info = TA.get_beat_info(orders)
    melody_bars = bar_events(midi.instruments[0].notes, beat_times, info, ppb)
    down = [i for i, o in enumerate(orders) if o == 1.0]
    first = info[down[0]][0]
    n_bars = len(down) - 1
    melody = [note_tokens(melody_bars.get(first + b, [])) for b in range(n_bars)]
    chords = None
    if chord_path is not None:
        events = chord_bar_events(chord_path, beat_times, info, ppb)
        chords = [chord_tokens(events.get(first + b, [])) for b in range(n_bars)]
    return melody, chords, beat_times, down


def piano_notes_to_seconds(bar_notes, beat_times, down, ppb):
    """bar_notes[b] = [(pos, pitch, vel, dur), ...] -> pretty_midi notes timed by the prompt's beats."""
    beat_times = np.asarray(beat_times)
    idx = np.arange(len(beat_times), dtype=float)
    last = beat_times[-1] - beat_times[-2]

    def beat_to_sec(beat):
        if beat <= idx[-1]:
            return float(np.interp(beat, idx, beat_times))
        return float(beat_times[-1] + (beat - idx[-1]) * last)

    notes = []
    for b, events in enumerate(bar_notes):
        start_beat = down[b]
        for pos, pitch, vel, dur in events:
            on = start_beat + pos / ppb
            notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=beat_to_sec(on), end=beat_to_sec(on + dur / ppb)))
    return notes


def write_vocab(base_tok2idx: dict[str, int], path: Path, variant: dict | None = None):
    tok2idx = dict(base_tok2idx)
    for tok in SPECIAL + (CHORD_TOKENS if (variant or {}).get("chords") else []):
        tok2idx.setdefault(tok, len(tok2idx))
    path.write_text(json.dumps(tok2idx, indent=1))
    return tok2idx
