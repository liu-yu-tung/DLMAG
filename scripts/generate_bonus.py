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

from hw2 import code_version
from hw2.bonus import accompany, piano_notes_to_seconds, tokenize_prompt
from hw2.generate import SampleConfig, load_model
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
        piano, seq = accompany(model, tok2idx, idx2tok, melody_bars, chord_bars, variant, bar_beats, cfg, gen,
                               args.grammar, args.pos_per_beat, args.max_notes_per_bar)

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
    meta = {"code_version": code_version(), "args": vars(args), "variant": variant, "config": cfg.name + ("_grammar" if args.grammar else ""), "seconds": round(time.time() - t0, 1), "pieces": info}
    (out / "gen_info.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
