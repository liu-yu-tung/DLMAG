"""BONUS TASK `chords` variant: share of piano (and melody) note time on the tones of the chord label sounding at
each slot, read from the token sequence. Real: the bonus_chd training tokens and the 10 prompts' piano.mid;
generated: abl_chords on the prompts (t1 p0.95, free, seed 0).

uv run python scripts/bonus_label_follow.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch

from hw2.bonus import accompany, chord_bar_events, tokenize_prompt
from hw2.generate import SampleConfig, load_model
from hw2.plan import chord_tones
from hw2.plan_generate import note_events
from hw2.pop909 import TA, bar_events, read_beats
from hw2.remi import DATA_ROOT, HW2_ROOT, load_vocab

PPB = 16


def fit(chord_events, notes, end):
    on = tot = 0
    for pos, pitch, _, dur in notes:
        for t in range(pos, min(pos + max(dur, 1), end)):
            tok = next((c for p, c in reversed(chord_events) if p <= t), None)
            tones = chord_tones(tok) if tok else None
            if tones is not None:
                tot += 1
                on += pitch % 12 in tones
    return on, tot


def parse_bars(seq):
    bars, cur = [], None
    for t in seq:
        if t == "Bar":
            cur = {"chd": [], "mel": [], "acc": [], "mode": None}
            bars.append(cur)
        elif cur is not None and t in ("Chd", "Mel", "Acc"):
            cur["mode"] = t
        elif cur is not None and t not in ("<EOS>",):
            cur.setdefault("raw_" + str(cur["mode"]), []).append(t)
    out = []
    for b in bars:
        chd, pos = [], 0
        for t in b.get("raw_Chd", []):
            if t.startswith("Pos_"):
                pos = int(t[4:])
            elif t.startswith("Chord_"):
                chd.append((pos, t))
        out.append((chd, note_events(b.get("raw_Mel", [])), note_events(b.get("raw_Acc", []))))
    return out


def score(bars, end_of):
    m = np.zeros(2); p = np.zeros(2)
    for i, (chd, mel, acc) in enumerate(bars):
        m += fit(chd, mel, end_of(i)); p += fit(chd, acc, end_of(i))
    return m[0] / m[1], p[0] / p[1]


train = json.loads((HW2_ROOT / "data/tokens/bonus_chd/sequences.json").read_text())
rs = [score(parse_bars(s["tokens"]), lambda i: 8 * PPB) for s in train[:200]]
rows = [{"source": "training_tokens_200", "melody_on_labels": np.nanmedian([r[0] for r in rs]), "piano_on_labels": np.nanmedian([r[1] for r in rs])}]

model = load_model(str(HW2_ROOT / "checkpoints/abl_chords/step_7000"), "cuda")
tok2idx, idx2tok = load_vocab(HW2_ROOT / "data/tokens/bonus_chd/tok2idx.json")
variant = {"lookahead": False, "chords": True}
gen_r, real_r = [], []
for d in sorted((DATA_ROOT / "prompts/bonus").iterdir()):
    melody_bars, chord_bars, beat_times, down = tokenize_prompt(d / "melody.mid", d / "beat_midi.txt", PPB, d / "chord_midi.txt")
    bar_beats = [down[i + 1] - down[i] for i in range(len(down) - 1)]
    _, seq = accompany(model, tok2idx, idx2tok, melody_bars, chord_bars, variant, bar_beats, SampleConfig(1.0, 0, 0.95),
                       torch.Generator(device="cuda").manual_seed(0))
    gen_r.append(score(parse_bars(idx2tok[i] if isinstance(i, int) else i for i in seq), lambda i: bar_beats[i] * PPB))
    _, orders = read_beats(d / "beat_midi.txt")
    info = TA.get_beat_info(orders)
    first = info[down[0]][0]
    piano = bar_events(pretty_midi.PrettyMIDI(str(d / "piano.mid")).instruments[0].notes, beat_times, info, PPB)
    real_bars = [(parse_bars(["Bar", "Chd", *chord_bars[b]])[0][0], note_events(melody_bars[b]), piano.get(first + b, [])) for b in range(len(bar_beats))]
    real_r.append(score(real_bars, lambda i: bar_beats[i] * PPB))
rows += [{"source": "prompts_real_piano", "melody_on_labels": np.median([r[0] for r in real_r]), "piano_on_labels": np.median([r[1] for r in real_r])},
         {"source": "prompts_abl_chords", "melody_on_labels": np.median([r[0] for r in gen_r]), "piano_on_labels": np.median([r[1] for r in gen_r])}]
table = pd.DataFrame(rows)
table.to_csv(HW2_ROOT / "results" / "bonus_label_follow.csv", index=False)
print(table.round(3).to_string(index=False))
