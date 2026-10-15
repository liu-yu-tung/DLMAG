"""Task a with the plan-then-fill pipeline (hw2.plan): the plan model writes n bars of plan, the fill model writes the
notes bar by bar, melody or accompaniment first.

Writes to --output_dir: sequences.json (TA eval format), plans.json (plan tokens, reusable with --plans_from),
<name>.mid (MELODY and PIANO tracks), <name>.wav (with --wav), gen_info.json.

uv run python scripts/generate_plan_fill.py --output_dir outputs/plan_fill/melody_first --order melody_first --wav
uv run python scripts/generate_plan_fill.py --output_dir outputs/plan_fill/piano_first --order piano_first \
    --plans_from outputs/plan_fill/melody_first/plans.json --wav
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hw2.generate import SampleConfig, load_model
from hw2.plan_generate import PlanRules, parse_form, parse_plan, sample_fill, sample_plan, to_outputs
from hw2.remi import DATA_ROOT, HW2_ROOT, load_vocab
from hw2.render import midi_to_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan_ckpt", type=str, default="checkpoints/plan_v1/best")
    p.add_argument("--fill_ckpt", type=str, default="checkpoints/fill_v1/best")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tok2idx", type=str, default=None, help="default: the token folder in runs/<fill run>/args.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--order", choices=("melody_first", "piano_first"), default="melody_first")
    p.add_argument("--drop", type=str, default="", help="attribute groups left out of the fill prefix: chords,piano_stats,melody_stats")
    p.add_argument("--plans_from", type=str, default=None, help="plans.json of an earlier run: fill the same plans")
    p.add_argument("--min_piano_bin", type=int, default=0, help="1 = no bar without piano notes (POP909: 0.4%% of bars)")
    p.add_argument("--tempo_range", type=str, default="", help="e.g. 60,125 (POP909 5th-95th percentile)")
    p.add_argument("--melody_by_bar", type=int, default=0, help="melody enters by this bar (0 = model decides; POP909 median entry is bar 8)")
    p.add_argument("--cadence", action="store_true", help="plan ends on V-I (or III-vi when vi outnumbers I)")
    p.add_argument("--form", type=str, default="", help="sections and bars, e.g. I4A8B8A8E4: I intro without melody, E free "
                   "ending, a repeated letter copies that section's plan and melody (piano is new); use with melody_first")
    p.add_argument("--return_as", choices=("copy", "exact", "var", "rhythm"), default="copy",
                   help="repeated form sections: copy the melody notes (rule), or ask a repeat-label model for that kind of repeat")
    p.add_argument("--plan_temperature", type=float, default=1.0)
    p.add_argument("--plan_top_p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wav", action="store_true", help="also render WAV with FluidSynth")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.tok2idx is None:
        run_args = HW2_ROOT / "runs" / Path(args.fill_ckpt).parent.name / "args.json"
        tokens_dir = json.loads(run_args.read_text())["tokens_dir"] if run_args.exists() else None
        args.tok2idx = str(Path(tokens_dir or DATA_ROOT.parent / "tokens" / "fill") / "tok2idx.json")
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    assert args.return_as == "copy" or "Rep_none" in tok2idx, "--return_as other than copy needs a repeat-label model"
    fill_model = load_model(args.fill_ckpt, args.device)
    plan_cfg = SampleConfig(args.plan_temperature, 0, args.plan_top_p)
    cfg = SampleConfig(args.temperature, args.top_k, args.top_p)
    drop = tuple(g for g in args.drop.split(",") if g)
    rules = PlanRules(args.min_piano_bin, tuple(int(x) for x in args.tempo_range.split(",")) if args.tempo_range else None,
                      args.melody_by_bar, args.cadence, args.form, args.return_as)
    copy_melody = parse_form(args.form, args.bars).copy_from if args.form and args.return_as == "copy" else None

    if args.plans_from:
        given = json.loads(Path(args.plans_from).read_text())[: args.n]
        plans = [parse_plan(p["tokens"]) for p in given]
    else:
        plan_model = load_model(args.plan_ckpt, args.device)
        plans = []
        for i in range(args.n):
            gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
            plans.append(sample_plan(plan_model, tok2idx, idx2tok, args.bars, plan_cfg, gen, rules=rules))
        del plan_model

    pieces, plan_rows, info = [], [], []
    t0 = time.time()
    for i, plan in enumerate(plans):
        name = f"{i:02d}"
        t1 = time.time()
        gen = torch.Generator(device=args.device).manual_seed(args.seed + 1000 + i)
        notes, nll = sample_fill(fill_model, tok2idx, idx2tok, plan, cfg, gen, melody_first=args.order == "melody_first", drop=drop,
                                 copy_melody=copy_melody)
        tokens, _ = to_outputs(plan, notes, out / f"{name}.mid")
        if args.wav:
            midi_to_wav(out / f"{name}.mid", out / f"{name}.wav")
        pieces.append({"song_name": name, "tokens": tokens})
        plan_rows.append({"song_name": name, "tokens": plan.tokens()})
        info.append({"song_name": name, "n_tokens": len(tokens), "n_bars": tokens.count("Bar"), "nll": float(np.mean(nll)),
                     "plan_nll": float(np.mean(plan.nll)) if plan.nll else None, "tempo": plan.tempo, "key": plan.key,
                     "n_melody": sum(len(b["melody"]) for b in notes), "n_piano": sum(len(b["piano"]) for b in notes),
                     "seconds": round(time.time() - t1, 1)})
        print(f"{name}: {info[-1]['n_melody']} melody / {info[-1]['n_piano']} piano notes, nll {info[-1]['nll']:.3f}, {info[-1]['seconds']}s", flush=True)

    (out / "sequences.json").write_text(json.dumps(pieces))
    (out / "plans.json").write_text(json.dumps(plan_rows))
    config = f"{args.order}_{cfg.name}" + (f"_drop-{'-'.join(drop)}" if drop else "")
    meta = {"args": vars(args), "config": config, "seconds": round(time.time() - t0, 1), "pieces": info}
    (out / "gen_info.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(pieces)} pieces in {meta['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
