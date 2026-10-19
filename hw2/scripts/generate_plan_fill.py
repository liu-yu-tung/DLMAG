"""TASK A, or TASK B with --prompt_dir, with the plan-then-fill pipeline (hw2.plan): the plan model writes the bar
labels, the fill model writes the notes bar by bar. Usage in the README."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from hw2 import code_version
from hw2.data import split_songs
from hw2.generate import SampleConfig, load_model
from hw2.plan import song_raw, song_tokens
from hw2.bonus import accompany
from hw2.plan_generate import PlanRules, parse_form, parse_plan, sample_fill, sample_plan, signed, to_outputs
from hw2.pop909 import note_tokens
from hw2.remi import DATA_ROOT, HW2_ROOT, load_sequences, load_vocab
from hw2.render import midi_to_wav


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan_ckpt", type=str, default="checkpoints/plan_v1/best")
    p.add_argument("--fill_ckpt", type=str, default="checkpoints/fill_v1/best")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tok2idx", type=str, default=None, help="default: the token folder in runs/<fill run>/args.json")
    p.add_argument("--n", type=int, default=20, help="pieces (TASK A)")
    p.add_argument("--bars", type=int, default=32, help="bars per piece, prompt included")
    p.add_argument("--skip_bars", type=int, default=0, help="TASK A: plan and fill bars + skip_bars (the form covers them all, "
                   "e.g. W8I4A8B8A8E4), then drop the first skip_bars")
    p.add_argument("--prompt_dir", type=str, default=None, help="TASK B: continue each folder in POP909 layout "
                   "(<name>/<name>.mid, beat_midi.txt, chord_midi.txt), e.g. data/POP909/prompts/basic-b")
    p.add_argument("--prompt_bars", type=int, default=8)
    p.add_argument("--order", choices=("melody_first", "piano_first"), default="melody_first")
    p.add_argument("--drop", type=str, default="", help="label groups left out of the fill prefix: repeat,chords,piano_stats,melody_stats")
    p.add_argument("--plans_from", type=str, default=None, help="plans.json of an earlier TASK A run: fill the same plans")
    p.add_argument("--real_plans", action="store_true", help="fill the real plans of the first --bars bars of --n songs "
                   "held out from the plan run (separates plan errors from fill errors)")
    p.add_argument("--min_piano_bin", type=int, default=0, help="1 = no bar without piano notes (POP909: 0.4%% of bars)")
    p.add_argument("--tempo_range", type=str, default="", help="e.g. 60,125 (POP909 5th-95th percentile)")
    p.add_argument("--melody_by_bar", type=int, default=0, help="melody enters by this bar (0 = model decides; POP909 median entry is bar 8)")
    p.add_argument("--cadence", action="store_true", help="plan ends on V-I (or III-vi when vi outnumbers I)")
    p.add_argument("--form", type=str, default="", help="sections and bars, e.g. I4A8B8A8E4: I intro without melody, E free "
                   "ending, a repeated letter copies that section's plan and melody (piano is new); use with melody_first")
    p.add_argument("--return_as", choices=("copy", "exact", "var", "rhythm"), default="copy",
                   help="repeated form sections: copy the melody notes (rule), or ask a repeat-label model for that kind of repeat")
    p.add_argument("--repeat_bias", type=float, default=0.0, help="logit bonus toward the source bar of melody bars with a "
                   "Rep label (repeat-label models; exact: all note parts, var: half, rhythm: onsets, velocities, durations)")
    p.add_argument("--plan_repeat_bias", type=float, default=0.0, help="logit bonus toward the source bar's chords and melody "
                   "bins when the plan model writes a Rep label (POP909 repeats keep them; exact: chords + melody bins, var: half, rhythm: melody bins)")
    p.add_argument("--plan_temperature", type=float, default=1.0)
    p.add_argument("--plan_top_p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--config", action="append", default=None, help="fill temperature,top_k,top_p (repeatable): one subfolder "
                   "per config, all filling the same plans; replaces --temperature/--top_k/--top_p")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--piano_ckpt", type=str, default=None, help="melody-only fill model (train.py --melody_only): the BONUS "
                   "model that writes the piano under the finished melody, e.g. checkpoints/abl_melody/step_7000")
    p.add_argument("--piano_tok2idx", type=str, default="data/tokens/bonus_ppb16/tok2idx.json")
    p.add_argument("--piano_top_p", type=float, default=0.95, help="BONUS TASK setting")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wav", action="store_true", help="also render WAV with FluidSynth")
    return p.parse_args()


def prompt_plans(prompt_dir: Path, prompt_bars: int, tok2idx_path: Path, tok2idx: dict) -> list[tuple]:
    """(name, plan of the prompt bars, prompt notes per bar) for each prompt folder, labelled by the training rules."""
    edges = json.loads((tok2idx_path.parent / "bins.json").read_text())
    out = []
    for d in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
        plan_tokens, fill = song_tokens(song_raw(d), edges, repeat="Rep_none" in tok2idx)
        plan = parse_plan(plan_tokens)
        assert len(plan.bars) >= prompt_bars, f"{d.name} has {len(plan.bars)} bars, fewer than --prompt_bars {prompt_bars}"
        plan.bars = plan.bars[:prompt_bars]
        given = {b: {s: fill["blocks"][b][s] for s in ("melody", "piano")} for b in range(prompt_bars)}
        out.append((d.name, plan, given))
    return out


def add_piano(model, tok2idx, idx2tok, plan, notes, cfg, gen):
    """Piano for the whole melody from the BONUS model, which reads pitches in the played key (as to_outputs plays them)."""
    shift = signed(plan.key)
    melody = [note_tokens([(pos, min(108, max(21, p + shift)), v, d) for pos, p, v, d in bar["melody"]]) for bar in notes]
    assert all(t in tok2idx for bar in melody for t in bar), "melody token missing from the BONUS vocabulary"
    piano, _ = accompany(model, tok2idx, idx2tok, melody, None, {}, [plan.meter] * len(notes), cfg, gen)
    for bar, events in zip(notes, piano):
        bar["piano"] = [(pos, p - shift, v, d) for pos, p, v, d in events]


def main():
    args = parse_args()
    assert sum(map(bool, (args.prompt_dir, args.plans_from, args.real_plans))) <= 1, "pick one plan source"
    assert not args.skip_bars or not (args.prompt_dir or args.plans_from or args.real_plans), "--skip_bars is for TASK A from scratch"
    total_bars = args.bars + args.skip_bars
    out = Path(args.output_dir)
    if args.tok2idx is None:
        run_args = HW2_ROOT / "runs" / Path(args.fill_ckpt).parent.name / "args.json"
        tokens_dir = json.loads(run_args.read_text())["tokens_dir"] if run_args.exists() else None
        args.tok2idx = str(Path(tokens_dir or DATA_ROOT.parent / "tokens" / "fill") / "tok2idx.json")
    tok2idx, idx2tok = load_vocab(args.tok2idx)
    assert args.return_as == "copy" or "Rep_none" in tok2idx, "--return_as other than copy needs a repeat-label model"
    assert not args.repeat_bias or "Rep_none" in tok2idx, "--repeat_bias needs a repeat-label model"
    plan_cfg = SampleConfig(args.plan_temperature, 0, args.plan_top_p)
    configs = ([SampleConfig(float(t), int(k), float(p)) for t, k, p in (c.split(",") for c in args.config)] if args.config
               else [SampleConfig(args.temperature, args.top_k, args.top_p)])
    fill_run = HW2_ROOT / "runs" / Path(args.fill_ckpt).parent.name / "args.json"
    fill_run = json.loads(fill_run.read_text()) if fill_run.exists() else {}
    melody_only = fill_run.get("melody_only", False)
    assert melody_only == bool(args.piano_ckpt), "a melody-only fill model needs --piano_ckpt, and only it takes one"
    drop = tuple(dict.fromkeys([g for g in args.drop.split(",") if g] + [g for g in fill_run.get("always_drop", "").split(",") if g]))
    rules = PlanRules(args.min_piano_bin, tuple(int(x) for x in args.tempo_range.split(",")) if args.tempo_range else None,
                      args.melody_by_bar, args.cadence, args.form, args.return_as, args.plan_repeat_bias)
    copy_melody = parse_form(args.form, total_bars).copy_from if args.form and args.return_as == "copy" else None

    if args.plans_from:
        jobs = [(p["song_name"], parse_plan(p["tokens"]), None) for p in json.loads(Path(args.plans_from).read_text())[: args.n]]
    elif args.real_plans:
        plan_run = json.loads((HW2_ROOT / "runs" / Path(args.plan_ckpt).parent.name / "args.json").read_text())
        _, held_out = split_songs(load_sequences(HW2_ROOT / plan_run["tokens_dir"] / "sequences.json"), plan_run["n_val"], plan_run["seed"])
        jobs = []
        for song in held_out[: args.n]:
            bar_starts = [i for i, t in enumerate(song["tokens"]) if t == "Bar"]
            assert len(bar_starts) > args.bars, f"{song['song_name']} is shorter than {args.bars} bars"
            jobs.append((song["song_name"], parse_plan(song["tokens"][: bar_starts[args.bars]]), None))
    else:
        prompts = (prompt_plans(Path(args.prompt_dir), args.prompt_bars, Path(args.tok2idx), tok2idx) if args.prompt_dir
                   else [(f"{i:02d}", None, None) for i in range(args.n)])
        plan_model = load_model(args.plan_ckpt, args.device)
        # a Rel_ fill vocabulary is the plan vocabulary plus tokens at the end; the plan model needs its own size
        plan_run = HW2_ROOT / "runs" / Path(args.plan_ckpt).parent.name / "args.json"
        plan_tokens = HW2_ROOT / json.loads(plan_run.read_text())["tokens_dir"] if plan_run.exists() else Path(args.tok2idx).parent
        plan_vocab = load_vocab(plan_tokens / "tok2idx.json")
        assert all(tok2idx.get(t) == i for t, i in plan_vocab[0].items()), "plan and fill vocabularies disagree"
        jobs = []
        for i, (name, prompt, given) in enumerate(prompts):
            gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
            jobs.append((name, sample_plan(plan_model, *plan_vocab, total_bars, plan_cfg, gen, prompt=prompt, rules=rules), given))
        del plan_model

    fill_model = load_model(args.fill_ckpt, args.device)
    if melody_only:
        piano_model, piano_vocab = load_model(args.piano_ckpt, args.device), load_vocab(args.piano_tok2idx)
    for cfg in configs:
        folder = out / cfg.name if args.config else out
        folder.mkdir(parents=True, exist_ok=True)
        pieces, plan_rows, info = [], [], []
        t0 = time.time()
        for i, (name, plan, given) in enumerate(jobs):
            t1 = time.time()
            gen = torch.Generator(device=args.device).manual_seed(args.seed + 1000 + i)
            notes, nll = sample_fill(fill_model, tok2idx, idx2tok, plan, cfg, gen, melody_first=args.order == "melody_first",
                                     drop=drop, given=given, copy_melody=copy_melody, repeat_bias=args.repeat_bias,
                                     melody_only=melody_only)
            if melody_only:
                add_piano(piano_model, *piano_vocab, plan, notes, SampleConfig(1.0, 0, args.piano_top_p), gen)
            if args.skip_bars:
                kept = plan.bars[args.skip_bars:]
                # a key change inside the dropped bars must stay on the first kept bar, or plans.json reads it in the first key
                if kept[0].key and not any(m.startswith("KeyShift_") for m in kept[0].marks):
                    kept[0] = dataclasses.replace(kept[0], marks=kept[0].marks + [f"KeyShift_{kept[0].key}"])
                plan, notes = dataclasses.replace(plan, bars=kept), notes[args.skip_bars:]
            tokens, _ = to_outputs(plan, notes, folder / f"{name}.mid")
            if args.wav:
                midi_to_wav(folder / f"{name}.mid", folder / f"{name}.wav")
            piece = {"song_name": name, "tokens": tokens}
            if given:
                piece["prompt_len"] = [j for j, t in enumerate(tokens) if t == "Bar"][args.prompt_bars]
            pieces.append(piece)
            plan_rows.append({"song_name": name, "tokens": plan.tokens()})
            info.append({"song_name": name, "n_tokens": len(tokens), "n_bars": tokens.count("Bar"), "nll": float(np.mean(nll)),
                         "plan_nll": float(np.mean(plan.nll)) if plan.nll else None, "tempo": plan.tempo, "key": plan.key,
                         "n_melody": sum(len(b["melody"]) for b in notes), "n_piano": sum(len(b["piano"]) for b in notes),
                         "seconds": round(time.time() - t1, 1)})
            print(f"{name}: {info[-1]['n_melody']} melody / {info[-1]['n_piano']} piano notes, nll {info[-1]['nll']:.3f}, {info[-1]['seconds']}s", flush=True)

        (folder / "sequences.json").write_text(json.dumps(pieces))
        (folder / "plans.json").write_text(json.dumps(plan_rows))
        config = f"{args.order}_{cfg.name}" + (f"_drop-{'-'.join(drop)}" if drop else "")
        meta = {"code_version": code_version(), "args": vars(args), "config": config, "seconds": round(time.time() - t0, 1), "pieces": info}
        (folder / "gen_info.json").write_text(json.dumps(meta, indent=1))
        print(f"{len(pieces)} pieces in {meta['seconds']}s -> {folder}")


if __name__ == "__main__":
    main()
