"""Round-trip check of the REMI decoder: tokens -> MIDI -> TA tokenizer -> tokens.

For each song, decode its tokens to MIDI, write a beat_midi.txt on the decoded grid
(downbeat at every bar start), re-tokenize with the TA tokenizer, and compare events.

Result on the 896 training songs (16 per beat, one tempo): every (bar, pos, pitch, velocity) comes back.
Durations come back equal or shorter: the TA tokenizer floors a float ratio (often 1 slot short), and a note
struck again while the same pitch still sounds (5% of notes) is cut when MIDI pairs note-on/off.

uv run python scripts/check_roundtrip.py --pos_per_beat 16
"""

import argparse
import contextlib
import importlib.util
import io
import tempfile
from collections import Counter
from pathlib import Path

import pretty_midi

from hw2.remi import DATA_ROOT, TICKS_PER_BEAT, load_sequences, notes_from_tokens, tokens_to_midi

HW2_ROOT = Path(__file__).resolve().parents[1]
# The TA tokenizer floors the onset's position within its beat, and rounds beat times to 1 us; a note exactly on a
# grid line could land just before the beat time and slip back one slot. Writing beats 5 us early avoids that.
BEAT_EPS = 5e-6


def load_ta_tokenizer():
    spec = importlib.util.spec_from_file_location("ta_tokenization", HW2_ROOT / "scripts" / "ta" / "tokenization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_song_folder(tokens, song_dir: Path, pos_per_beat: int):
    song_dir.mkdir(parents=True)
    midi = tokens_to_midi(tokens, song_dir / f"{song_dir.name}.mid", pos_per_beat)
    _, _, bar_starts = notes_from_tokens(tokens, pos_per_beat)
    bar_ends = bar_starts[1:] + [max([n.end for n in midi.instruments[0].notes] + [bar_starts[-1]]) + 4 * TICKS_PER_BEAT]
    # Read the file back so beat times use the tempo as stored (integer microseconds per beat), as the tokenizer sees it
    tick_to_sec = pretty_midi.PrettyMIDI(str(song_dir / f"{song_dir.name}.mid")).tick_to_time

    lines = []
    for start, end in zip(bar_starts, bar_ends):
        for k, tick in enumerate(range(start, end, TICKS_PER_BEAT)):
            flag = "1.0" if k == 0 else "0.0"
            lines.append(f"{tick_to_sec(tick) - BEAT_EPS:.6f} {flag} {flag}")
    (song_dir / "beat_midi.txt").write_text("\n".join(lines) + "\n")
    (song_dir / "chord_midi.txt").write_text("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_per_beat", type=int, default=16)
    parser.add_argument("--sequences", type=str, default=None)
    parser.add_argument(
        "--keep_tempo_changes",
        action="store_true",
        help="Keep every Tempo token. By default only the first is kept: the TA tokenizer places onsets by linear "
        "interpolation between beat times, which a mid-beat tempo change breaks, so the strict grid check uses one tempo.",
    )
    args = parser.parse_args()
    seq_path = args.sequences or DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}" / "sequences.json"
    songs = load_sequences(seq_path)
    if not args.keep_tempo_changes:
        for song in songs:
            first = next((t for t in song["tokens"] if t.startswith("Tempo_")), None)
            song["tokens"] = [t for t in song["tokens"] if not t.startswith("Tempo_") or t is first]
            song["tokens"] = drop_orphan_pos(song["tokens"])
    ta = load_ta_tokenizer()

    with tempfile.TemporaryDirectory() as tmp:
        for song in songs:
            write_song_folder(song["tokens"], Path(tmp) / song["song_name"], args.pos_per_beat)
        with contextlib.redirect_stdout(io.StringIO()):
            redone = ta.process_pop909_dataset(tmp, args.pos_per_beat, 8 * args.pos_per_beat)

    redone = {s["song_name"]: s["tokens"] for s in redone}
    n_exact = 0
    counts = Counter()
    examples = []
    for song in songs:
        a, b = song["tokens"], redone.get(song["song_name"], [])
        n_exact += a == b
        ea, eb = events(a), events(b)
        counts["notes"] += len(ea["note"])
        counts["onset_pitch_vel_match"] += sum((Counter(ea["note"]) & Counter(eb["note"])).values())
        durs_b = {}
        for key, dur in zip(eb["note"], eb["dur"]):
            durs_b.setdefault(key, []).append(dur)
        for key, dur in zip(ea["note"], ea["dur"]):
            if durs_b.get(key):
                other = durs_b[key].pop(0)
                counts["dur_exact"] += dur == other
                counts["dur_within_1"] += abs(dur - other) <= 1
        counts["tempos"] += len(ea["tempo"])
        counts["tempo_match"] += sum((Counter(ea["tempo"]) & Counter(eb["tempo"])).values())
        counts["bars"] += a.count("Bar")
        counts["bars_redo"] += b.count("Bar")
        if Counter(ea["note"]) != Counter(eb["note"]) and len(examples) < 3:
            missing = sorted((Counter(ea["note"]) - Counter(eb["note"])).elements())[:3] + [(len(ea["note"]), len(eb["note"]), ea["tempo"][:2], eb["tempo"][:2])]
            extra = sorted((Counter(eb["note"]) - Counter(ea["note"])).elements())[:3]
            examples.append((song["song_name"], missing, extra))
    print(f"songs with identical tokens: {n_exact}/{len(songs)}")
    print(f"bars: {counts['bars']} -> {counts['bars_redo']}")
    for key in ("onset_pitch_vel_match", "dur_exact", "dur_within_1"):
        print(f"{key}: {counts[key]}/{counts['notes']} ({counts[key] / counts['notes']:.4%})")
    print(f"tempo_match: {counts['tempo_match']}/{counts['tempos']}")
    for name, missing, extra in examples:
        print(f"  song {name}: (bar, pos, pitch, vel) missing {missing}, extra {extra}")


def drop_orphan_pos(tokens):
    """Remove Pos_ tokens left without an event after dropping Tempo tokens."""
    return [t for i, t in enumerate(tokens) if not (t.startswith("Pos_") and (i + 1 == len(tokens) or not tokens[i + 1].startswith(("Pitch_", "Tempo_"))))]


def events(tokens):
    """Note events (bar, pos, pitch, vel) with their durations, and tempo events (bar, pos, tempo)."""
    out = {"note": [], "dur": [], "tempo": []}
    bar, pos = -1, 0
    for i, tok in enumerate(tokens):
        if tok == "Bar":
            bar, pos = bar + 1, 0
        elif tok.startswith("Pos_"):
            pos = int(tok[4:])
        elif tok.startswith("Tempo_"):
            out["tempo"].append((bar, pos, tok))
        elif tok.startswith("Pitch_") and i + 2 < len(tokens):
            out["note"].append((bar, pos, tok, tokens[i + 1]))
            out["dur"].append(int(tokens[i + 2][4:]))
    return out


if __name__ == "__main__":
    main()
