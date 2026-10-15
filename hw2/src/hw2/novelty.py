"""How much of a generated piece is copied from the training songs (Exp 2).

- bar_copy: share of non-empty generated bars that equal some training bar note-for-note
  (Pos/Pitch/Dur; velocity and tempo ignored). bar_copy_any_key also allows any transposition.
- ngram_copy: share of generated token n-grams (default n=32, about 8 notes) found in the training songs.
- note8_copy: share of runs of 8 consecutive notes whose (position, interval to the run's first pitch) sequence
  occurs in the training songs; ignores velocity, duration and key, so it also catches loose or transposed copies.
"""

from hw2.remi import split_bars


def bar_key(bar: list[str], relative: bool = False):
    notes = []
    pos = 0
    for i, t in enumerate(bar):
        if t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and i + 2 < len(bar):
            notes.append((pos, int(t[6:]), bar[i + 2]))
    if not notes:
        return None
    if relative:
        low = min(p for _, p, _ in notes)
        notes = [(pos, p - low, d) for pos, p, d in notes]
    return tuple(sorted(notes))


def note_skeleton(tokens: list[str]) -> list[tuple[int, int]]:
    out, pos = [], 0
    for t in tokens:
        if t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_"):
            out.append((pos, int(t[6:])))
    return out


def note_runs(notes, n=8):
    return [hash(tuple((pos, p - notes[i][1]) for pos, p in notes[i : i + n])) for i in range(len(notes) - n + 1)]


def ngrams(tokens: list[str], n: int):
    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


class CopyIndex:
    def __init__(self, songs: list[dict], n: int = 32):
        self.n = n
        self.bars, self.bars_rel, self.grams, self.runs = set(), set(), set(), set()
        for s in songs:
            for bar in split_bars(s["tokens"]):
                key = bar_key(bar)
                if key:
                    self.bars.add(key)
                    self.bars_rel.add(bar_key(bar, relative=True))
            self.grams |= ngrams(s["tokens"], n)
            self.runs.update(note_runs(note_skeleton(s["tokens"])))

    def score(self, tokens: list[str], skip_bars: int = 0) -> dict[str, float]:
        bars = [b for b in split_bars(tokens)[skip_bars:] if bar_key(b)]
        gen = [t for b in split_bars(tokens)[skip_bars:] for t in b]
        grams = [hash(tuple(gen[i : i + self.n])) for i in range(len(gen) - self.n + 1)]
        runs = note_runs(note_skeleton(gen))
        return {
            "bar_copy": sum(bar_key(b) in self.bars for b in bars) / max(1, len(bars)),
            "bar_copy_any_key": sum(bar_key(b, True) in self.bars_rel for b in bars) / max(1, len(bars)),
            "ngram_copy": sum(g in self.grams for g in grams) / max(1, len(grams)),
            "note8_copy": sum(r in self.runs for r in runs) / max(1, len(runs)),
        }
