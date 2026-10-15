"""Sampling for the plan-then-fill pipeline (hw2.plan): a plan model writes the song's skeleton, a fill model writes
the notes bar by bar with the bar's plan forced.

Both samplers follow a token grammar, so every output parses: plan bars are [Sec] [KeyShift] Bar Chord Chord PDen
PDur MDen MDur; fill note blocks are (Pos Pitch Vel Dur)* with positions that never go back inside a block and
stay inside the bar.
"""

import re
from dataclasses import dataclass, field

import miditoolkit
import torch

from hw2.generate import SampleConfig, filter_logits
from hw2.plan import N_BINS, PPB, SECTION_BARS, STAT_NAMES, signed
from hw2.remi import TICKS_PER_BEAT


class Stepper:
    """Feeds tokens through a GPT-2 with a KV cache. Past the context it re-encodes from the first block start
    (Sec, KeyShift or Bar) in the last keep * ctx tokens, as training crops start at a block."""

    def __init__(self, model, tok2idx, gen, keep: float = 0.5):
        self.model, self.gen = model, gen
        self.device = next(model.parameters()).device
        self.ctx = model.config.n_positions
        self.keep = int(self.ctx * keep)
        self.starts = {i for t, i in tok2idx.items() if t in ("Bar", "Sec") or t.startswith("KeyShift_")}
        self.seq, self.past, self.n_seen, self.logits = [], None, 0, None
        self.nll = []

    def feed(self, ids):
        ids = list(ids)
        self.seq += ids
        if self.n_seen + len(ids) > self.ctx:
            tail = len(self.seq) - self.keep
            start = next((i for i in range(tail, len(self.seq)) if self.seq[i] in self.starts), tail)
            self.past, self.n_seen, ids = None, 0, self.seq[start:]
        x = torch.tensor([ids], device=self.device)
        pos = torch.arange(self.n_seen, self.n_seen + len(ids), device=self.device)[None]
        out = self.model(input_ids=x, position_ids=pos, past_key_values=self.past, use_cache=True)
        self.past, self.n_seen = out.past_key_values, self.n_seen + len(ids)
        self.logits = out.logits[0, -1].float()

    def sample(self, allowed: torch.Tensor, cfg: SampleConfig) -> int:
        masked = self.logits.masked_fill(~allowed, float("-inf"))[None]
        t = torch.multinomial(torch.softmax(filter_logits(masked, cfg), -1), 1, generator=self.gen).item()
        self.nll.append(-torch.log_softmax(self.logits, -1)[t].item())
        return t


class Masks:
    def __init__(self, tok2idx, device):
        self.tok2idx, self.n, self.device = tok2idx, len(tok2idx), device
        self.cache = {}

    def of(self, *prefixes, exact=()):
        key = (prefixes, tuple(exact))
        if key not in self.cache:
            m = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            for t, i in self.tok2idx.items():
                if t.startswith(prefixes) if prefixes else False:
                    m[i] = True
            for t in exact:
                m[self.tok2idx[t]] = True
            self.cache[key] = m
        return self.cache[key]

    def pos_range(self, lo, hi, extra=()):
        m = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        m[[self.tok2idx[f"Pos_{p}"] for p in range(lo, hi)]] = True
        for t in extra:
            m[self.tok2idx[t]] = True
        return m


@dataclass
class PlanBar:
    marks: list[str]
    chords: list[str]
    stats: list[str]
    key: int = 0
    rep: str = ""

    def tokens(self):
        return self.marks + ["Bar"] + ([self.rep] if self.rep else []) + self.chords + self.stats


@dataclass
class Plan:
    tempo: int
    meter: int
    key: int
    bars: list[PlanBar] = field(default_factory=list)
    nll: list[float] = field(default_factory=list)

    def tokens(self):
        return ["<BOS>", f"Tempo_{self.tempo}", f"Meter_{self.meter}", f"Key_{self.key}"] + [t for b in self.bars for t in b.tokens()] + ["<EOS>"]


@dataclass
class PlanRules:
    """Sampling-time limits on the plan. Defaults change nothing."""
    min_piano_bin: int = 0
    tempo_range: tuple[int, int] | None = None
    melody_by_bar: int = 0
    cadence: bool = False
    form: str = ""
    return_as: str = "copy"


@dataclass
class Form:
    copy_from: dict[int, int]
    intro: set[int]
    melody_start: int | None


def parse_form(form: str, n_bars: int) -> Form:
    """'I4A8B8A8E4': sections and their lengths in bars. I = intro without melody, E = free ending, a letter seen
    before repeats that section bar by bar (plan and melody; the fill model writes new piano)."""
    parts = re.findall(r"([A-Z])(\d+)", form)
    assert "".join(f"{x}{n}" for x, n in parts) == form, f"bad form {form!r}"
    assert sum(int(n) for _, n in parts) == n_bars, f"form {form!r} is not {n_bars} bars"
    first, copy_from, intro, start, b = {}, {}, set(), None, 0
    for letter, n in parts:
        n = int(n)
        if letter == "I":
            intro.update(range(b, b + n))
        elif letter in first and letter != "E":
            src, length = first[letter]
            assert n <= length, f"repeat of {letter} is longer than its first statement"
            copy_from.update({b + i: src + i for i in range(n)})
        else:
            first[letter] = (b, n)
            start = b if start is None and letter != "E" else start
        b += n
    return Form(copy_from, intro, start)


def tonic_frame(plan: Plan) -> tuple[str, str]:
    """(dominant, tonic) chord tokens: major (V, I) unless vi outnumbers I in the plan so far, then (III, vi)."""
    chords = [c for b in plan.bars for c in b.chords]
    if chords.count("Chord_9_min") > chords.count("Chord_0_maj"):
        return "Chord_4_maj", "Chord_9_min"
    return "Chord_7_maj", "Chord_0_maj"


def parse_plan(tokens: list[str]) -> Plan:
    """Plan tokens (with or without <BOS>/<EOS>) -> Plan; key of each bar relative to the first key."""
    head = {t.split("_")[0]: int(t.split("_")[1]) for t in tokens[:4] if t.startswith(("Tempo_", "Meter_", "Key_"))}
    plan, marks, key, cur = Plan(head["Tempo"], head["Meter"], head["Key"]), [], 0, None
    for t in tokens:
        if t == "Sec" or t.startswith("KeyShift_"):
            marks.append(t)
            if t.startswith("KeyShift_"):
                key = int(t.split("_")[1])
        elif t == "Bar":
            cur = PlanBar(marks, [], [], key)
            plan.bars.append(cur)
            marks = []
        elif cur is not None and t.startswith("Rep_"):
            cur.rep = t
        elif cur is not None and t.startswith("Chord_"):
            cur.chords.append(t)
        elif cur is not None and t.startswith(STAT_NAMES):
            cur.stats.append(t)
    return plan


@torch.no_grad()
def sample_plan(model, tok2idx, idx2tok, n_bars, cfg, gen, prompt: Plan | None = None, rules: PlanRules = PlanRules()) -> Plan:
    """Sample the header (unless a prompt plan is given) and bars until the plan has n_bars bars."""
    st, mk = Stepper(model, tok2idx, gen), Masks(tok2idx, next(model.parameters()).device)
    form = parse_form(rules.form, n_bars) if rules.form else Form({}, set(), None)
    has_rep = "Rep_none" in tok2idx
    if rules.tempo_range:
        lo, hi = rules.tempo_range
        tempo_ok = mk.of(exact=[t for t in tok2idx if t.startswith("Tempo_") and lo <= int(t[6:]) <= hi])
    if prompt is None:
        st.feed([tok2idx["<BOS>"]])
        head = []
        for pre in ("Tempo_", "Meter_", "Key_"):
            t = st.sample(tempo_ok if pre == "Tempo_" and rules.tempo_range else mk.of(pre), cfg)
            head.append(idx2tok[t])
            st.feed([t])
        plan = Plan(*(int(h.split("_")[1]) for h in head))
    else:
        plan = Plan(prompt.tempo, prompt.meter, prompt.key, list(prompt.bars))
        st.feed([tok2idx[t] for t in ["<BOS>", f"Tempo_{plan.tempo}", f"Meter_{plan.meter}", f"Key_{plan.key}"]])
        for bar in plan.bars:
            st.feed([tok2idx[t] for t in bar.tokens()])
    key = plan.bars[-1].key if plan.bars else 0
    while len(plan.bars) < n_bars:
        marks = ["Sec"] if len(plan.bars) % SECTION_BARS == 0 else []
        if marks:
            st.feed([tok2idx["Sec"]])
        n = len(plan.bars)
        src = plan.bars[form.copy_from[n]] if n in form.copy_from else None
        shifts = [f"KeyShift_{k}" for k in range(-5, 7) if k != key] if src is None else []
        t = idx2tok[st.sample(mk.of(exact=["Bar"] + shifts), cfg)]
        if t != "Bar":
            marks.append(t)
            key = int(t.split("_")[1])
            st.feed([tok2idx[t]])
        st.feed([tok2idx["Bar"]])
        bar = PlanBar(marks, [], [], key)
        if has_rep:
            if src is not None:
                lag = n - form.copy_from[n]
                rep = f"Rep_{'exact' if rules.return_as == 'copy' else rules.return_as}_{lag}"
                assert rep in tok2idx, f"no {rep} token: returns must be 2, 4, 8 or 16 bars after the section they repeat"
                allowed = mk.of(exact=[rep])
            elif n in form.intro:
                allowed = mk.of(exact=["Rep_none"])
            else:
                allowed = mk.of(exact=["Rep_none"] + [t for t in tok2idx if t.startswith("Rep_") and t != "Rep_none" and int(t.rsplit("_", 1)[1]) <= n])
            t = st.sample(allowed, cfg)
            bar.rep = idx2tok[t]
            st.feed([t])
        dominant, tonic = tonic_frame(plan)
        cadence = {(n_bars - 2, 1): dominant, (n_bars - 1, 0): tonic, (n_bars - 1, 1): tonic} if rules.cadence else {}
        melody_due = rules.melody_by_bar and n + 1 >= rules.melody_by_bar and all(b.stats[2] == "MDen_0" for b in plan.bars)
        for slot, pre in enumerate(("Chord_", "Chord_") + tuple(f"{s}_" for s in STAT_NAMES)):
            allowed = mk.of(pre)
            if (n, slot) in cadence:
                allowed = mk.of(exact=[cadence[n, slot]])
            elif src is not None:
                allowed = mk.of(exact=[(src.chords + src.stats)[slot]])
            elif pre == "PDen_" and rules.min_piano_bin:
                allowed = mk.of(exact=[f"PDen_{b}" for b in range(rules.min_piano_bin, N_BINS)])
            elif pre == "MDen_" and n in form.intro:
                allowed = mk.of(exact=["MDen_0"])
            elif pre == "MDen_" and (melody_due or n == form.melody_start):
                allowed = mk.of(exact=[f"MDen_{b}" for b in range(1, N_BINS)])
            t = st.sample(allowed, cfg)
            (bar.chords if pre == "Chord_" else bar.stats).append(idx2tok[t])
            st.feed([t])
        plan.bars.append(bar)
    plan.nll = st.nll
    return plan


def fill_prefix(bar: PlanBar, drop=()) -> list[str]:
    """Plan bar -> fill prefix: chords move from degrees above the local key to pitch classes of the C-based song."""
    chords = []
    for c in bar.chords:
        if c == "Chord_N":
            chords.append(c)
        else:
            _, d, q = c.split("_")
            chords.append(f"Chord_{(int(d) + bar.key) % 12}_{q}")
    groups = {"repeat": [bar.rep] if bar.rep else [], "chords": chords, "piano_stats": bar.stats[:2], "melody_stats": bar.stats[2:]}
    return bar.marks + ["Bar"] + [t for g in ("repeat", "chords", "piano_stats", "melody_stats") if g not in drop for t in groups[g]]


def note_events(tokens):
    out, cur = [], {}
    for t in tokens:
        kind, _, v = t.partition("_")
        if kind == "Pos":
            cur = {"pos": int(v)}
        elif kind == "Pitch" and "pos" in cur:
            cur["pitch"] = int(v)
        elif kind == "Vel" and "pitch" in cur:
            cur["vel"] = int(v)
        elif kind == "Dur" and "vel" in cur:
            out.append((cur["pos"], cur["pitch"], cur["vel"], int(v)))
            cur = {}
    return out


@torch.no_grad()
def sample_fill(model, tok2idx, idx2tok, plan: Plan, cfg, gen, melody_first=True, drop=(), given=None, max_notes=(32, 64),
                copy_melody: dict[int, int] | None = None):
    """Notes for every plan bar, in the C-based song (add signed(plan.key) to play), and the NLL of each sampled token.
    given[b] forces streams of bar b (prompts); copy_melody[b] = an earlier bar whose melody bar b repeats, moved to b's key."""
    st, mk = Stepper(model, tok2idx, gen, keep=0.8), Masks(tok2idx, next(model.parameters()).device)
    st.feed([tok2idx["<BOS>"], tok2idx[f"Meter_{plan.meter}"]])
    ends = mk.of("KeyShift_", exact=["Bar", "Sec", "<EOS>"])
    max_pos = min(plan.meter * PPB, 128)
    out = []
    for b, bar in enumerate(plan.bars):
        st.feed([tok2idx[t] for t in fill_prefix(bar, drop)])
        order = ("melody", "piano") if melody_first else ("piano", "melody")
        res = {}
        for i, stream in enumerate(order):
            st.feed([tok2idx["Mel" if stream == "melody" else "Acc"]])
            forced = (given or {}).get(b, {}).get(stream)
            if forced is None and stream == "melody" and b in (copy_melody or {}):
                a = copy_melody[b]
                k = signed(bar.key - plan.bars[a].key)
                forced = [t for pos, pitch, vel, dur in out[a]["melody"] if f"Pitch_{pitch + k}" in tok2idx
                          for t in (f"Pos_{pos}", f"Pitch_{pitch + k}", f"Vel_{vel}", f"Dur_{dur}")]
            if forced is not None:
                if forced:
                    st.feed([tok2idx[t] for t in forced])
                res[stream] = note_events(forced)
                continue
            end = mk.of(exact=["Acc" if stream == "melody" else "Mel"]) if i == 0 else ends
            toks, state, last = [], "pos", 0
            cap = max_notes[0] if stream == "melody" else max_notes[1]
            while True:
                if state == "pos":
                    allowed = mk.pos_range(last, max_pos) | end if len(toks) < 4 * cap else end
                else:
                    allowed = mk.of({"pitch": "Pitch_", "vel": "Vel_", "dur": "Dur_"}[state])
                t = st.sample(allowed, cfg)
                if end[t]:
                    break
                tok = idx2tok[t]
                toks.append(tok)
                st.feed([t])
                if state == "pos":
                    last, state = int(tok[4:]), "pitch"
                else:
                    state = {"pitch": "vel", "vel": "dur", "dur": "pos"}[state]
            res[stream] = note_events(toks)
        out.append(res)
    return out, st.nll


def to_outputs(plan: Plan, notes, path_mid=None):
    """(TA-REMI tokens of the merged piece, MIDI with MELODY and PIANO tracks) in the planned key and tempo."""
    shift = signed(plan.key)
    bar_ticks = plan.meter * TICKS_PER_BEAT
    step = TICKS_PER_BEAT // PPB
    tokens = ["<BOS>"]
    midi = miditoolkit.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    midi.tempo_changes = [miditoolkit.TempoChange(plan.tempo, 0)]
    tracks = {s: miditoolkit.Instrument(program=0, is_drum=False, name=n) for s, n in (("melody", "MELODY"), ("piano", "PIANO"))}
    for b, res in enumerate(notes):
        tokens.append("Bar")
        events = [] if b else [(0, -1, 0, 0, "tempo")]
        for stream in ("melody", "piano"):
            for pos, pitch, vel, dur in res[stream]:
                p = min(108, max(21, pitch + shift))
                events.append((pos, p, vel, dur, stream))
                start = b * bar_ticks + pos * step
                tracks[stream].notes.append(miditoolkit.Note(velocity=vel, pitch=p, start=start, end=start + dur * step))
        for pos, p, vel, dur, kind in sorted(events, key=lambda e: (e[0], e[4] != "tempo", e[1])):
            tokens += [f"Pos_{pos}"] + ([f"Tempo_{plan.tempo}"] if kind == "tempo" else [f"Pitch_{p}", f"Vel_{vel}", f"Dur_{dur}"])
    tokens.append("<EOS>")
    midi.instruments += [tracks["melody"], tracks["piano"]]
    if path_mid is not None:
        midi.dump(str(path_mid))
    return tokens, midi
