"""Bonus: generate a PIANO track for each 16-bar MELODY prompt.

Per bar, the tokens Bar [Chd chords] Mel <melody notes> Acc are forced (see hw2.bonus for the variants; the
variant is read from variant.json next to --tok2idx); piano notes are sampled until the model emits Bar.
With --grammar, sampling follows Pos -> Pitch -> Vel -> Dur, positions never go back in a bar and stay inside
the bar length given by the prompt's beat_midi.txt.
Output: <output_dir>/<song>.mid with tracks MELODY (the prompt, unchanged) and PIANO, plus .wav with --wav.

uv run python scripts/generate_bonus.py --ckpt checkpoints/abl_melody/step_7000 --output_dir outputs/bonus --wav
"""

import argparse
import json
import time
from pathlib import Path

import pretty_midi
import torch

from hw2.bonus import bar_prefix, piano_notes_to_seconds, prelude, tokenize_prompt
from hw2.generate import SampleConfig, filter_logits, load_model
from hw2.remi import DATA_ROOT, load_vocab
from hw2.render import midi_to_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prompt_dir", type=str, default=str(DATA_ROOT / "prompts" / "bonus"))
    p.add_argument("--tok2idx", type=str, default=str(DATA_ROOT.parent / "tokens" / "bonus_ppb16" / "tok2idx.json"))
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--grammar", action="store_true")
    p.add_argument("--max_notes_per_bar", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wav", action="store_true")
    return p.parse_args()


class Grammar:
    """Allowed next tokens for the accompaniment part of a bar."""

    def __init__(self, tok2idx, vocab_size, device):
        def ids(prefix):
            return [i for t, i in tok2idx.items() if t.startswith(prefix)]

        self.pos_id = {int(t[4:]): i for t, i in tok2idx.items() if t.startswith("Pos_")}
        self.masks = {}
        for name, allowed in {"pitch": ids("Pitch_"), "vel": ids("Vel_"), "dur": ids("Dur_")}.items():
            m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            m[allowed] = True
            self.masks[name] = m
        self.bar = tok2idx["Bar"]
        self.vocab_size, self.device = vocab_size, device

    def pos_or_bar(self, min_pos, max_pos):
        m = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        m[[self.pos_id[p] for p in range(min_pos, max_pos)]] = True
        m[self.bar] = True
        return m


@torch.no_grad()
def accompany(model, tok2idx, idx2tok, melody_bars, chord_bars, variant, bar_beats, cfg, args, gen):
    device = next(model.parameters()).device
    grammar = Grammar(tok2idx, len(idx2tok), device) if args.grammar else None
    free_ok = torch.zeros(len(idx2tok), dtype=torch.bool, device=device)
    for t, i in tok2idx.items():
        if t == "Bar" or t.startswith(("Pos_", "Pitch_", "Vel_", "Dur_")):
            free_ok[i] = True

    past, n_seen = None, 0
    seq = [tok2idx[t] for t in ["<BOS>"] + prelude(melody_bars, variant)]
    ctx = model.config.n_positions

    def feed(new_ids):
        """Run new_ids (already appended to seq) through the model. Past the context, re-encode from the
        first Bar in the last ctx//2 tokens, as training crops start at a Bar."""
        nonlocal past, n_seen
        if n_seen + len(new_ids) > ctx:
            tail = len(seq) - ctx // 2
            start = next((i for i in range(tail, len(seq)) if seq[i] == tok2idx["Bar"]), tail)
            past, n_seen, new_ids = None, 0, seq[start:]
        x = torch.tensor([new_ids], device=device)
        pos = torch.arange(n_seen, n_seen + len(new_ids), device=device)[None]
        out = model(input_ids=x, position_ids=pos, past_key_values=past, use_cache=True)
        past, n_seen = out.past_key_values, n_seen + len(new_ids)
        return out.logits[0, -1].float()

    logits = feed(seq)
    piano = []
    for b in range(len(melody_bars)):
        forced = [tok2idx[t] for t in bar_prefix(b, melody_bars, chord_bars, variant)]
        seq += forced
        logits = feed(forced)
        notes, state, last_pos, cur = [], "pos", 0, {}
        max_pos = bar_beats[b] * args.pos_per_beat
        for _ in range(4 * args.max_notes_per_bar + 1):
            if grammar is None:
                allowed = free_ok
            elif state == "pos":
                allowed = grammar.pos_or_bar(last_pos, max_pos) if len(notes) < args.max_notes_per_bar else grammar.pos_or_bar(0, 0)
            else:
                allowed = grammar.masks[state]
            masked = logits.masked_fill(~allowed, float("-inf"))[None]
            t = torch.multinomial(torch.softmax(filter_logits(masked, cfg), -1), 1, generator=gen).item()
            if t == tok2idx["Bar"]:
                break
            tok = idx2tok[t]
            seq.append(t)
            logits = feed([t])
            kind = tok.split("_")[0]
            if kind == "Pos":
                last_pos, cur, state = int(tok[4:]), {"pos": int(tok[4:])}, "pitch"
            elif kind == "Pitch" and "pos" in cur:
                cur["pitch"], state = int(tok[6:]), "vel"
            elif kind == "Vel" and "pitch" in cur:
                cur["vel"], state = int(tok[4:]), "dur"
            elif kind == "Dur" and "vel" in cur:
                notes.append((cur["pos"], cur["pitch"], cur["vel"], int(tok[4:])))
                cur, state = {}, "pos"
        piano.append(notes)
    return piano, seq


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    variant_file = Path(args.tok2idx).parent / "variant.json"
    variant = json.loads(variant_file.read_text()) if variant_file.exists() else {}
    model = load_model(args.ckpt)
    cfg = SampleConfig(args.temperature, args.top_k, args.top_p)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    info = []
    t0 = time.time()
    for d in sorted(p for p in Path(args.prompt_dir).iterdir() if p.is_dir()):
        chord_path = d / "chord_midi.txt" if variant.get("chords") else None
        melody_bars, chord_bars, beat_times, down = tokenize_prompt(d / "melody.mid", d / "beat_midi.txt", args.pos_per_beat, chord_path)
        bar_beats = [down[i + 1] - down[i] for i in range(len(down) - 1)]
        piano, seq = accompany(model, tok2idx, idx2tok, melody_bars, chord_bars, variant, bar_beats, cfg, args, gen)

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        melody = pretty_midi.PrettyMIDI(str(d / "melody.mid")).instruments[0]
        melody.name = "MELODY"
        piano_track = pretty_midi.Instrument(program=0, name="PIANO")
        piano_track.notes = piano_notes_to_seconds(piano, beat_times, down, args.pos_per_beat)
        midi.instruments += [melody, piano_track]
        midi.write(str(out / f"{d.name}.mid"))
        if args.wav:
            midi_to_wav(out / f"{d.name}.mid", out / f"{d.name}.wav")
        info.append({"song_name": d.name, "n_tokens": len(seq), "piano_notes": sum(map(len, piano)), "notes_per_bar": [len(x) for x in piano]})
        print(d.name, info[-1]["piano_notes"], "piano notes", flush=True)
    meta = {"args": vars(args), "variant": variant, "config": cfg.name + ("_grammar" if args.grammar else ""), "seconds": round(time.time() - t0, 1), "pieces": info}
    (out / "gen_info.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
