"""Task a: generate pieces of n bars from scratch.

Writes to --output_dir: sequences.json (TA eval format), <name>.mid, <name>.wav (with --wav), gen_info.json.

uv run python scripts/generate_uncond.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir outputs/task_a --wav
"""

import argparse
import json
import time
from pathlib import Path

from hw2.generate import SampleConfig, generate, load_model
from hw2.remi import DATA_ROOT, load_vocab, tokens_to_midi
from hw2.render import midi_to_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tok2idx", type=str, default=str(DATA_ROOT.parent / "tokens" / "ppb16" / "tok2idx.json"))
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wav", action="store_true", help="also render WAV with FluidSynth")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    model = load_model(args.ckpt)
    cfg = SampleConfig(args.temperature, args.top_k, args.top_p)

    pieces, info = [], []
    t0 = time.time()
    for b0 in range(0, args.n, args.batch_size):
        n = min(args.batch_size, args.n - b0)
        prompts = [[tok2idx["<BOS>"], tok2idx["Bar"]]] * n
        ids, nll = generate(model, prompts, args.bars - 1, cfg, tok2idx, seed=args.seed + b0)
        for k, (seq, loss) in enumerate(zip(ids, nll)):
            name = f"{b0 + k:02d}"
            tokens = [idx2tok[i] for i in seq] + ["<EOS>"]
            pieces.append({"song_name": name, "tokens": tokens})
            info.append({"song_name": name, "n_tokens": len(tokens), "n_bars": tokens.count("Bar"), "nll": loss})
            tokens_to_midi(tokens, out / f"{name}.mid", args.pos_per_beat)
            if args.wav:
                midi_to_wav(out / f"{name}.mid", out / f"{name}.wav")
    (out / "sequences.json").write_text(json.dumps(pieces))
    meta = {"args": vars(args), "config": cfg.name, "seconds": round(time.time() - t0, 1), "pieces": info}
    (out / "gen_info.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(pieces)} pieces in {meta['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
