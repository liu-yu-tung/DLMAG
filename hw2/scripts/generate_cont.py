"""Task b: continue each 8-bar prompt by --bars bars (prompt kept in the output).

Writes to --output_dir/<config>/: <prompt>.mid, <prompt>.wav (with --wav), sequences.json, gen_info.json.
Several configs: repeat --config "temperature,top_k,top_p".

uv run python scripts/generate_cont.py --ckpt checkpoints/gpt2_ppb16/<checkpoint> --output_dir outputs/task_b \
    --config 1.0,0,0.95 --config 1.2,0,0.9 --wav
"""

import argparse
import json
import time
from pathlib import Path

from hw2 import code_version
from hw2.generate import SampleConfig, generate, load_model
from hw2.remi import DATA_ROOT, load_sequences, load_vocab, tokens_to_midi
from hw2.render import midi_to_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prompts", type=str, default=str(DATA_ROOT.parent / "tokens" / "ppb16" / "prompts_basic_b.json"))
    p.add_argument("--tok2idx", type=str, default=str(DATA_ROOT.parent / "tokens" / "ppb16" / "tok2idx.json"))
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--bars", type=int, default=24)
    p.add_argument("--config", action="append", default=None, help="temperature,top_k,top_p (repeatable)")
    p.add_argument("--samples", type=int, default=1, help="continuations per prompt and config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wav", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    model = load_model(args.ckpt)
    prompts = load_sequences(args.prompts)
    configs = [SampleConfig(float(t), int(k), float(p)) for t, k, p in (c.split(",") for c in (args.config or ["1.0,0,1.0"]))]

    for cfg in configs:
        out = Path(args.output_dir) / cfg.name
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(s["song_name"], j) for s in prompts for j in range(args.samples)]
        prompt_ids = {s["song_name"]: [tok2idx[t] for t in s["tokens"] if t != "<EOS>"] for s in prompts}
        t0 = time.time()
        ids, nll = generate(model, [prompt_ids[name] for name, _ in jobs], args.bars, cfg, tok2idx, seed=args.seed)
        pieces, info = [], []
        for (name, j), seq, loss in zip(jobs, ids, nll):
            fname = name if args.samples == 1 else f"{name}_{j}"
            tokens = [idx2tok[i] for i in seq] + ["<EOS>"]
            pieces.append({"song_name": fname, "tokens": tokens, "prompt_len": len(prompt_ids[name])})
            info.append({"song_name": fname, "n_tokens": len(tokens), "n_bars": tokens.count("Bar"), "nll": loss})
            tokens_to_midi(tokens, out / f"{fname}.mid", args.pos_per_beat)
            if args.wav:
                midi_to_wav(out / f"{fname}.mid", out / f"{fname}.wav")
        (out / "sequences.json").write_text(json.dumps(pieces))
        meta = {"code_version": code_version(), "args": vars(args), "config": cfg.name, "seconds": round(time.time() - t0, 1), "pieces": info}
        (out / "gen_info.json").write_text(json.dumps(meta, indent=1))
        print(f"{cfg.name}: {len(pieces)} pieces in {meta['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
