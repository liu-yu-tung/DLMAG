"""generate_plan_fill.py --piano_ckpt (add_piano) against generate_bonus.py (accompany on the prompt tokens): same BONUS
model, prompt melodies and seed must give the same piano, bar for bar, also with the melody moved to another key.

uv run python scripts/check_piano_hookup.py
"""

import argparse
from types import SimpleNamespace

import torch

from generate_plan_fill import add_piano
from hw2.bonus import accompany, tokenize_prompt
from hw2.generate import SampleConfig, load_model
from hw2.plan import signed
from hw2.plan_generate import note_events
from hw2.remi import DATA_ROOT, HW2_ROOT, load_vocab


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default=str(HW2_ROOT / "checkpoints" / "abl_melody" / "step_7000"))
    p.add_argument("--tok2idx", default=str(HW2_ROOT / "data" / "tokens" / "bonus_ppb16" / "tok2idx.json"))
    p.add_argument("--prompts", nargs="+", default=["001", "002", "003"])
    p.add_argument("--keys", nargs="+", type=int, default=[0, 5])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    model = load_model(args.ckpt, args.device)
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    cfg = SampleConfig(1.0, 0, 0.95)
    for name in args.prompts:
        d = DATA_ROOT / "prompts" / "bonus" / name
        melody_bars, _, _, down = tokenize_prompt(d / "melody.mid", d / "beat_midi.txt", 16)
        bar_beats = [down[i + 1] - down[i] for i in range(len(down) - 1)]
        assert len(set(bar_beats)) == 1, "add_piano takes one meter per piece"
        reference, _ = accompany(model, tok2idx, idx2tok, melody_bars, None, {}, bar_beats, cfg,
                                 torch.Generator(device=args.device).manual_seed(args.seed))
        for key in args.keys:
            shift = signed(key)
            notes = [{"melody": [(pos, q - shift, v, dur) for pos, q, v, dur in note_events(bar)]} for bar in melody_bars]
            add_piano(model, tok2idx, idx2tok, SimpleNamespace(key=key, meter=bar_beats[0]), notes, cfg,
                      torch.Generator(device=args.device).manual_seed(args.seed))
            hookup = [[(pos, q + shift, v, dur) for pos, q, v, dur in bar["piano"]] for bar in notes]
            same = sum(list(map(tuple, a)) == list(map(tuple, b)) for a, b in zip(reference, hookup))
            print(f"{name} key {key}: identical bars {same} of {len(reference)}, piano notes {sum(map(len, reference))} vs {sum(map(len, hookup))}")
            assert same == len(reference)


if __name__ == "__main__":
    main()
