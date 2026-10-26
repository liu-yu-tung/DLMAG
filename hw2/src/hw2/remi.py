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


def bar_beats_used(bar: list[str], pos_per_beat: int) -> int:
    """Beats needed to hold the last Pos_ in a bar (0 for an empty bar)."""
    pos = [int(t[4:]) for t in bar if t.startswith("Pos_")]
    return max(pos) // pos_per_beat + 1 if pos else 0


def infer_beats_per_bar(tokens: list[str], pos_per_beat: int, default: int = 4) -> int:
    """Piece meter: 6 if most bars use beat 5 or later (the 28 six-beat POP909 songs), else 4.

    About half the 4-beat songs also have a few 2/5/6-beat bars from the beat annotation;
    bar_lengths handles those per bar.
    """
    bars = [b for b in split_bars(tokens) if len(b) > 1]
    if bars and sum(bar_beats_used(b, pos_per_beat) > default for b in bars) >= len(bars) / 2:
        return max(6, default)
    return default


def bar_lengths(tokens: list[str], pos_per_beat: int, beats_per_bar: int | None = None) -> list[int]:
    """Beats per bar: the piece meter, stretched for a bar whose notes run past it."""
    if beats_per_bar is None:
        beats_per_bar = infer_beats_per_bar(tokens, pos_per_beat)
    return [max(beats_per_bar, bar_beats_used(b, pos_per_beat)) for b in split_bars(tokens)]


def notes_from_tokens(tokens: list[str], pos_per_beat: int, beats_per_bar: int | None = None):
    """Parse TA-style REMI tokens into (notes, tempos, bar_starts) on a tick grid.

    notes: (start_tick, end_tick, pitch, velocity); tempos: (tick, qpm); bar_starts: tick of each Bar.
    Tokens before the first Bar count as bar 0. Incomplete note triples (Pitch without Vel/Dur) are dropped.
    """
    step = TICKS_PER_BEAT // pos_per_beat
    bar_starts = [0]
    for beats in bar_lengths(tokens, pos_per_beat, beats_per_bar):
        bar_starts.append(bar_starts[-1] + beats * TICKS_PER_BEAT)
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
            tempos.append((bar_starts[max(bar, 0)] + pos * step, int(tok[6:])))
        elif tok.startswith("Pitch_") and i + 2 < len(tokens):
            vel_tok, dur_tok = tokens[i + 1], tokens[i + 2]
            if vel_tok.startswith("Vel_") and dur_tok.startswith("Dur_"):
                start = bar_starts[max(bar, 0)] + pos * step
                end = start + int(dur_tok[4:]) * step
                notes.append((start, end, int(tok[6:]), int(vel_tok[4:])))
                i += 2
        i += 1
    return notes, tempos, bar_starts[: max(bar, 0) + 1]


def tokens_to_midi(
    tokens: list[str],
    out_path: str | Path | None,
    pos_per_beat: int,
    beats_per_bar: int | None = None,
    track_name: str = "PIANO",
) -> miditoolkit.MidiFile:
    notes, tempos, _ = notes_from_tokens(tokens, pos_per_beat, beats_per_bar)
    midi = miditoolkit.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, tempos[0][1] if tempos else DEFAULT_TEMPO))
    midi.tempo_changes = [miditoolkit.TempoChange(q, t) for t, q in tempos]
    # without it readers assume 4/4: a 6-beat song showed 48 bars on the listening page (T6 pair 12)
    tick, last = 0, None
    for beats in bar_lengths(tokens, pos_per_beat, beats_per_bar):
        if beats != last:
            midi.time_signature_changes.append(miditoolkit.TimeSignature(beats, 4, tick))
            last = beats
        tick += beats * TICKS_PER_BEAT
    inst = miditoolkit.Instrument(program=0, is_drum=False, name=track_name)
    inst.notes = [miditoolkit.Note(velocity=v, pitch=p, start=s, end=e) for s, e, p, v in notes]
    midi.instruments.append(inst)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        midi.dump(str(out_path))
    return midi
