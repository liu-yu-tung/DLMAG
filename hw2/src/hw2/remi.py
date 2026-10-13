import json
from pathlib import Path

import miditoolkit

HW2_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = HW2_ROOT / "data" / "POP909"

TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 120


def load_vocab(path: str | Path) -> tuple[dict[str, int], list[str]]:
    tok2idx = json.loads(Path(path).read_text())
    idx2tok = [None] * len(tok2idx)
    for tok, i in tok2idx.items():
        idx2tok[i] = tok
    return tok2idx, idx2tok


def load_sequences(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def split_bars(tokens: list[str]) -> list[list[str]]:
    bars = []
    for tok in tokens:
        if tok == "Bar":
            bars.append(["Bar"])
        elif bars:
            bars[-1].append(tok)
    return bars


def infer_beats_per_bar(tokens: list[str], pos_per_beat: int, default: int = 4) -> int:
    """Piece-level bar length: 4 beats unless a position needs more (POP909 has 28 six-beat songs)."""
    max_pos = max((int(t[4:]) for t in tokens if t.startswith("Pos_")), default=0)
    return max(default, -(-(max_pos + 1) // pos_per_beat))


def notes_from_tokens(tokens: list[str], pos_per_beat: int, beats_per_bar: int | None = None):
    """Parse TA-style REMI tokens into (notes, tempos) on a tick grid.

    notes: (start_tick, end_tick, pitch, velocity); tempos: (tick, qpm).
    Incomplete note triples (Pitch without Vel/Dur) are dropped.
    """
    if beats_per_bar is None:
        beats_per_bar = infer_beats_per_bar(tokens, pos_per_beat)
    step = TICKS_PER_BEAT // pos_per_beat
    bar_ticks = beats_per_bar * TICKS_PER_BEAT
    notes, tempos = [], []
    bar = -1
    pos = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "Bar":
            bar += 1
            pos = 0
        elif tok.startswith("Pos_"):
            pos = int(tok[4:])
        elif tok.startswith("Tempo_"):
            tempos.append((max(bar, 0) * bar_ticks + pos * step, int(tok[6:])))
        elif tok.startswith("Pitch_") and i + 2 < len(tokens):
            vel_tok, dur_tok = tokens[i + 1], tokens[i + 2]
            if vel_tok.startswith("Vel_") and dur_tok.startswith("Dur_"):
                start = max(bar, 0) * bar_ticks + pos * step
                end = start + int(dur_tok[4:]) * step
                notes.append((start, end, int(tok[6:]), int(vel_tok[4:])))
                i += 2
        i += 1
    return notes, tempos


def tokens_to_midi(
    tokens: list[str],
    out_path: str | Path | None,
    pos_per_beat: int,
    beats_per_bar: int | None = None,
    track_name: str = "PIANO",
) -> miditoolkit.MidiFile:
    notes, tempos = notes_from_tokens(tokens, pos_per_beat, beats_per_bar)
    midi = miditoolkit.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, tempos[0][1] if tempos else DEFAULT_TEMPO))
    midi.tempo_changes = [miditoolkit.TempoChange(q, t) for t, q in tempos]
    inst = miditoolkit.Instrument(program=0, is_drum=False, name=track_name)
    inst.notes = [miditoolkit.Note(velocity=v, pitch=p, start=s, end=e) for s, e, p, v in notes]
    midi.instruments.append(inst)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        midi.dump(str(out_path))
    return midi
