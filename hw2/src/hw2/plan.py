"""Plan-fill pipeline: rule labels per bar, plan tokens (whole song) and fill blocks (per bar). Rules and token
layout in docs/plan_fill.md."""

from collections import Counter

import numpy as np
import pretty_midi

from hw2.pop909 import CHORD_QUALITIES, KEY_PENALTY, SCALES, TA, bar_events, chord_token, key_path, note_tokens, read_beats

PPB = 16
SECTION_BARS = 8
N_BINS = 8

STAT_NAMES = ("PDen", "PDur", "MDen", "MDur")
REPEAT_LAGS = (2, 4, 8, 16)
REPEAT_KINDS = ("exact", "var", "rhythm")


def vocab_tokens(repeat: bool = False) -> list[str]:
    toks = [f"Meter_{m}" for m in (4, 6)] + [f"Key_{k}" for k in range(12)] + ["Sec"]
    toks += [f"KeyShift_{k}" for k in range(-5, 7)]
    toks += ["Chord_N"] + [f"Chord_{r}_{q}" for r in range(12) for q in CHORD_QUALITIES]
    toks += [f"{s}_{b}" for s in STAT_NAMES for b in range(N_BINS)]
    toks += ["Mel", "Acc"]
    if repeat:
        toks += ["Rep_none"] + [f"Rep_{k}_{lag}" for k in REPEAT_KINDS for lag in REPEAT_LAGS]
    return toks


def repeat_kind(bar: list, earlier: list) -> str | None:
    """Thresholds from the POP909 melody survey in results/findings.md: exact 24.5% of melody bars, var 13.2%,
    rhythm 8.1%."""
    if not bar or not earlier:
        return None
    a, b = {(p, n) for p, n, *_ in bar}, {(p, n) for p, n, *_ in earlier}
    if len(a & b) / len(a | b) >= 0.8:
        return "exact"
    pa, pb = Counter(n for _, n, *_ in bar), Counter(n for _, n, *_ in earlier)
    ha, hb = np.bincount([n % 12 for _, n, *_ in bar], minlength=12), np.bincount([n % 12 for _, n, *_ in earlier], minlength=12)
    if sum((pa & pb).values()) / sum((pa | pb).values()) >= 0.6 and ha @ hb / np.linalg.norm(ha) / np.linalg.norm(hb) >= 0.85:
        return "var"
    ra, rb = {p for p, *_ in bar}, {p for p, *_ in earlier}
    if len(ra & rb) / len(ra | rb) >= 0.8 and len(a & b) / len(a | b) < 0.5:
        return "rhythm"
    return None


def repeat_label(melody_bars: list[list], i: int) -> str:
    """Strongest kind (exact > var > rhythm) over the lags; ties go to the shorter lag."""
    found = [(REPEAT_KINDS.index(k), lag) for lag in REPEAT_LAGS if i - lag >= 0
             for k in [repeat_kind(melody_bars[i], melody_bars[i - lag])] if k]
    if not found:
        return "Rep_none"
    k, lag = min(found)
    return f"Rep_{REPEAT_KINDS[k]}_{lag}"


def signed(k: int) -> int:
    k %= 12
    return k - 12 if k > 6 else k


def bar_spans(beat_times, info):
    """{bar: (start_sec, end_sec, beats, index of its first beat)} from the TA beat grid."""
    first = {}
    for i, (bar, _) in enumerate(info):
        first.setdefault(bar, i)
    bars = sorted(first)
    step = beat_times[-1] - beat_times[-2] if len(beat_times) > 1 else 0.5
    spans = {}
    for j, b in enumerate(bars):
        i0 = first[b]
        i1 = first[bars[j + 1]] if j + 1 < len(bars) else len(beat_times)
        end = beat_times[i1] if i1 < len(beat_times) else beat_times[-1] + step
        spans[b] = (beat_times[i0], end, i1 - i0, i0)
    return spans


def chord_at(chords, t):
    for a, e, tok in chords:
        if a <= t < e:
            return tok
    return "Chord_N"


def rel_chord(tok, tonic):
    if tok == "Chord_N":
        return tok
    _, r, q = tok.split("_")
    return f"Chord_{(int(r) - tonic) % 12}_{q}"


def song_raw(song_dir):
    """Rule-based labels and note events of one song, before binning. None if the song has no notes."""
    pm = pretty_midi.PrettyMIDI(str(song_dir / f"{song_dir.name}.mid"))
    beat_times, orders = read_beats(song_dir / "beat_midi.txt")
    info = TA.get_beat_info(orders)
    tracks = {i.name.strip().upper(): i.notes for i in pm.instruments}
    melody_notes = tracks.get("MELODY", [])
    piano_notes = tracks.get("PIANO", []) + tracks.get("BRIDGE", [])
    melody = bar_events(melody_notes, beat_times, info, PPB)
    piano = bar_events(piano_notes, beat_times, info, PPB)
    used = set(melody) | set(piano)
    if not used:
        return None
    spans = bar_spans(beat_times, info)
    bars = [b for b in range(min(used), max(used) + 1) if b in spans]

    hist = np.zeros((len(bars), 12))
    for n in melody_notes + piano_notes:
        for j, b in enumerate(bars):
            a, e = spans[b][0], spans[b][1]
            s, f = max(n.start, a), min(n.end, e)
            if f > s:
                hist[j, n.pitch % 12] += f - s
    tot = hist.sum(1, keepdims=True)
    share = np.divide(hist, tot, out=np.zeros_like(hist), where=tot > 0)
    local = key_path(np.where(tot > 0, 1 - share @ SCALES.T, 0.0), KEY_PENALTY)

    chords = []
    for line in (song_dir / "chord_midi.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            chords.append((float(parts[0]), float(parts[1]), chord_token(parts[2])))

    beats = [spans[b][2] for b in bars]
    meter = 6 if sum(x >= 5 for x in beats) >= len(beats) / 2 else 4
    bpm = 60 / np.median(np.diff(beat_times))
    out = {"song_name": song_dir.name, "key": local[0], "shift": signed(local[0]), "meter": meter,
           "tempo": int(min(210, max(30, round(bpm / 5) * 5))), "bars": []}
    for j, b in enumerate(bars):
        a, _, nb, i0 = spans[b]
        half = beat_times[min(i0 + max(1, nb // 2), len(beat_times) - 1)]
        m, p = melody.get(b, []), piano.get(b, [])
        out["bars"].append({
            "tonic": local[j], "chords": [chord_at(chords, a), chord_at(chords, half)],
            "stats": [len(p), np.mean([d for *_, d in p]) / PPB if p else 0.0, len(m), np.mean([d for *_, d in m]) / PPB if m else 0.0],
            "melody": m, "piano": p,
        })
    return out


def bin_edges(raws) -> dict[str, list[float]]:
    """Quantile edges (6 inner edges -> bins 1..7) of the nonzero values of each stat over the given songs."""
    edges = {}
    for i, name in enumerate(STAT_NAMES):
        vals = np.array([bar["stats"][i] for r in raws for bar in r["bars"] if bar["stats"][i] > 0])
        edges[name] = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS)[1:-1])).tolist()
    return edges


def stat_bin(value: float, edges: list[float]) -> int:
    return 0 if value <= 0 else 1 + int(np.searchsorted(edges, value, side="right"))


def transpose_events(events, shift):
    return [(pos, min(108, max(21, pitch - shift)), vel, dur) for pos, pitch, vel, dur in events]


def song_tokens(raw, edges, repeat: bool = False):
    """Plan tokens and fill blocks of one song (pitches transposed so the first key is C)."""
    melody_bars = [bar["melody"] for bar in raw["bars"]]
    header = [f"Tempo_{raw['tempo']}", f"Meter_{raw['meter']}", f"Key_{raw['key']}"]
    plan, blocks, prev = ["<BOS>"] + header, [], raw["key"]
    for j, bar in enumerate(raw["bars"]):
        marks = ["Sec"] if j % SECTION_BARS == 0 else []
        if bar["tonic"] != prev:
            marks.append(f"KeyShift_{signed(bar['tonic'] - raw['key'])}")
            prev = bar["tonic"]
        stats = [f"{n}_{stat_bin(v, edges[n])}" for n, v in zip(STAT_NAMES, bar["stats"])]
        rep = [repeat_label(melody_bars, j)] if repeat else []
        plan += marks + ["Bar"] + rep + [rel_chord(c, bar["tonic"]) for c in bar["chords"]] + stats
        blocks.append({
            "prefix": marks + ["Bar"] + rep + [rel_chord(c, raw["key"]) for c in bar["chords"]] + stats,
            "melody": note_tokens(transpose_events(bar["melody"], raw["shift"])),
            "piano": note_tokens(transpose_events(bar["piano"], raw["shift"])),
        })
    return plan + ["<EOS>"], {"song_name": raw["song_name"], "header": [f"Meter_{raw['meter']}"], "key": raw["key"],
                              "shift": raw["shift"], "tempo": raw["tempo"], "blocks": blocks}
