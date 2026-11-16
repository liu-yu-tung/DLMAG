import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import T5TokenizerFast

from text_to_music.prompts import PROMPT_STYLES, build_prompt, load_captions, load_tags
from text_to_music.sampler import MODEL_ID, Guidance, generate, load_pipeline

HW3 = Path(__file__).resolve().parents[2]


def code_version() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HW3, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."], cwd=HW3,
                           capture_output=True, text=True).stdout.strip()
    return commit + ("+dirty" if dirty else "")


def read_records(path: Path) -> dict[str, dict]:
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["id"]] = record
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one run of clips with Stable Audio Open (resumable).")
    parser.add_argument("--captions", type=Path, default=HW3 / "data/test.csv", help="csv with id,caption")
    parser.add_argument("--set", dest="set_name", default="test", help="prompt set name: test or dev")
    parser.add_argument("--run", required=True, help="run name, e.g. cfg7")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="caption")
    parser.add_argument("--tags", type=Path, help="jsonl of {id, tags: [..]} for tag prompt styles")
    parser.add_argument("--mode", choices=("cfg", "apg"), default="cfg")
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--apg-eta", type=float, default=0.0)
    parser.add_argument("--apg-momentum", type=float, default=0.0)
    parser.add_argument("--apg-norm-threshold", type=float, default=0.0)
    parser.add_argument("--apg-threshold-file", type=Path,
                        help="apg_threshold.json from analyze_runs.py; overrides --apg-norm-threshold with --apg-threshold-key")
    parser.add_argument("--apg-threshold-key", default="median", choices=("median", "twice_median"))
    parser.add_argument("--interval-sigma-low", type=float, default=0.0,
                        help="guide only where sigma > this (guidance interval, Kynkaanniemi et al. 2024)")
    parser.add_argument("--interval-sigma-high", type=float, default=float("inf"),
                        help="guide only where sigma <= this; outside the interval only the conditional pass runs")
    parser.add_argument("--rescale-phi", type=float, default=0.0,
                        help="CFG rescale mix (Lin et al. 2024); 0 is off, the paper uses 0.7; cfg mode only")
    parser.add_argument("--rescale-valid-frames", action="store_true",
                        help="CFG rescale takes its std over the latent frames that hold the clip, not the padded tail")
    parser.add_argument("--second-prompt-style", choices=PROMPT_STYLES,
                        help="second prompt for composed guidance, built from the same --tags file; needs --second-scale")
    parser.add_argument("--second-scale", type=float, default=0.0,
                        help="share of --scale given to the second prompt (composed guidance); cfg mode only")
    parser.add_argument("--extra-prompt", help="fixed text for an extra guidance term on top of plain CFG; needs --extra-prompt-scale")
    parser.add_argument("--extra-prompt-file", type=Path, help="csv with id,text: a per-caption text for the extra term")
    parser.add_argument("--extra-prompt-scale", type=float, default=0.0,
                        help="weight of the extra term; negative = steer away from --extra-prompt (negation), caption weight stays --scale")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed-index", type=int, default=0, help="seed = id + 1000 * seed_index")
    parser.add_argument("--ids", help="comma-separated subset of ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-quick-clap", action="store_true")
    parser.add_argument("--compile", choices=("auto", "on", "off"), default="auto",
                        help="torch.compile the transformer; auto follows results/bench_compile.json (off if missing)")
    parser.add_argument("--out-root", type=Path, default=HW3 / "outputs")
    parser.add_argument("--results-root", type=Path, default=HW3 / "results")
    return parser.parse_args()


def resolve_compile(choice: str) -> bool:
    if choice != "auto":
        return choice == "on"
    bench = HW3 / "results/bench_compile.json"
    return bench.exists() and bool(json.loads(bench.read_text())["use_compile"])


def build_prompts(args, ids: list[str], captions: dict[str, str]) -> dict[str, dict]:
    tags = load_tags(args.tags) if args.tags else None
    # The class model_index.json names; AutoTokenizer looks for a root config.json, which fails offline.
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")

    def measured(style: str, item_id: str) -> tuple[str, int, bool]:
        text = build_prompt(style, captions[item_id], tags.get(item_id) if tags else None)
        tokens = len(tokenizer(text).input_ids)
        return text, tokens, tokens > tokenizer.model_max_length

    extra_texts = {}
    if args.extra_prompt_file:
        with open(args.extra_prompt_file, newline="") as handle:
            extra_texts = {row["id"]: row["text"] for row in csv.DictReader(handle)}
    assert not args.extra_prompt_file or set(ids) <= set(extra_texts), "--extra-prompt-file must cover every caption"
    prompts = {}
    for item_id in ids:
        prompt, tokens, truncated = measured(args.prompt_style, item_id)
        prompts[item_id] = {"id": item_id, "prompt": prompt, "tokens": tokens, "truncated": truncated}
        if args.second_prompt_style:
            second, second_tokens, second_truncated = measured(args.second_prompt_style, item_id)
            prompts[item_id] |= {"second_prompt": second, "second_prompt_tokens": second_tokens,
                                 "second_prompt_truncated": second_truncated}
        extra = args.extra_prompt or (extra_texts[item_id] if args.extra_prompt_file else None)
        if extra:
            extra_tokens = len(tokenizer(extra).input_ids)
            prompts[item_id] |= {"second_prompt": extra, "second_prompt_tokens": extra_tokens,
                                 "second_prompt_truncated": extra_tokens > tokenizer.model_max_length}
    for label, token_key, truncated_key in (("", "tokens", "truncated"), ("second ", "second_prompt_tokens", "second_prompt_truncated")):
        if token_key in next(iter(prompts.values())):
            truncated = sum(p[truncated_key] for p in prompts.values())
            print(f"{len(prompts)} {label}prompts, max {max(p[token_key] for p in prompts.values())} tokens, "
                  f"{truncated} over the {tokenizer.model_max_length}-token limit (T5 cuts the tail)", flush=True)
    return prompts


def build_guidance(args) -> Guidance:
    if args.apg_threshold_file:
        args.apg_norm_threshold = json.loads(args.apg_threshold_file.read_text())["suggested_thresholds"][args.apg_threshold_key]
    has_extra = args.extra_prompt is not None or args.extra_prompt_file is not None
    assert not (args.extra_prompt and args.extra_prompt_file), "one extra prompt source"
    assert not (has_extra and args.second_prompt_style), "one second prompt only"
    assert args.second_prompt_style is None or args.second_scale > 0, "--second-prompt-style needs --second-scale"
    assert args.second_scale == 0 or args.second_prompt_style or has_extra, "--second-scale needs a second prompt"
    # an extra prompt is either added on top of the caption's weight (--extra-prompt-scale) or takes a share of it (--second-scale)
    assert not has_extra or (args.extra_prompt_scale != 0) != (args.second_scale > 0), \
        "an extra prompt needs exactly one of --extra-prompt-scale and --second-scale"
    assert has_extra or args.extra_prompt_scale == 0, "--extra-prompt-scale needs --extra-prompt or --extra-prompt-file"
    return Guidance(args.mode, args.scale, args.apg_eta, args.apg_momentum, args.apg_norm_threshold,
                    args.interval_sigma_low, args.interval_sigma_high, args.rescale_phi, args.second_scale, args.rescale_valid_frames,
                    args.extra_prompt_scale)


def build_settings(args, guidance: Guidance, compile_transformer: bool) -> dict:
    settings = {
        "captions": args.captions.name, "prompt_style": args.prompt_style, "tags": args.tags.name if args.tags else None,
        "guidance": guidance.settings(), "steps": args.steps, "seconds": args.seconds, "seed_index": args.seed_index,
        "compile": compile_transformer,
    }
    # Added only when set, so runs from before this option still resume.
    if args.second_prompt_style:
        settings["second_prompt_style"] = args.second_prompt_style
    if args.extra_prompt:
        settings["extra_prompt"] = args.extra_prompt
    if args.extra_prompt_file:
        settings["extra_prompt_file"] = args.extra_prompt_file.name
    return settings


def main():
    args = parse_args()
    captions = load_captions(args.captions)
    ids = args.ids.split(",") if args.ids else sorted(captions, key=int)
    if args.limit:
        ids = ids[: args.limit]
    guidance = build_guidance(args)
    # Prompts are built first so a missing tag list fails before the model load, not mid-run.
    prompts = build_prompts(args, ids, captions)

    audio_dir = args.out_root / args.set_name / args.run
    record_dir = args.results_root / f"{args.set_name}__{args.run}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    quick_path = record_dir / "quick.jsonl"
    info_path = record_dir / "gen_info.json"
    settings = build_settings(args, guidance, resolve_compile(args.compile))
    if info_path.exists():
        previous = json.loads(info_path.read_text())["settings"]
        if previous != settings:
            raise SystemExit(f"{record_dir} holds a run with other settings:\n{previous}\nvs\n{settings}")
    info = {"model": MODEL_ID, "code_version": code_version(), "settings": settings,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    done = read_records(quick_path)

    pipe = load_pipeline(compile_transformer=settings["compile"])
    quick_clap = None
    if not args.no_quick_clap:
        from text_to_music.quick_clap import QuickClap
        quick_clap = QuickClap()
    sampling_rate = pipe.vae.config.sampling_rate

    with open(quick_path, "a") as quick_file, open(audio_dir / "prompts.jsonl", "a") as prompt_file, \
            open(record_dir / "trajectory.jsonl", "a") as trajectory_file:
        for item_id in ids:
            if item_id in done and (audio_dir / f"{item_id}.wav").exists():
                continue
            seed = int(item_id) + 1000 * args.seed_index
            torch.cuda.synchronize()
            start = time.perf_counter()
            audio, item_info = generate(pipe, prompts[item_id]["prompt"], guidance, args.steps, args.seconds, seed,
                                       prompts[item_id].get("second_prompt"))
            torch.cuda.synchronize()
            seconds_taken = time.perf_counter() - start
            audio = audio.float()
            magnitude = audio.abs()
            # soundfile hard-clips beyond +-1 when writing PCM; clamp here so quick CLAP scores the same signal.
            levels = {"peak": round(float(magnitude.max()), 4), "clip_fraction": float((magnitude >= 1).float().mean()),
                      "rms": round(float(audio.pow(2).mean().sqrt()), 5)}
            audio = audio.clamp(-1, 1)
            sf.write(audio_dir / f"{item_id}.wav", audio.T.cpu().numpy(), sampling_rate)
            trajectory = item_info.pop("trajectory")
            record = {"id": item_id, "seed": seed, "seconds": round(seconds_taken, 3), "time": round(time.time(), 1),
                      "tokens": prompts[item_id]["tokens"], **levels, **item_info}
            if quick_clap is not None:
                record["clap_quick"] = quick_clap.score(audio, sampling_rate, captions[item_id])
            quick_file.write(json.dumps(record) + "\n")
            quick_file.flush()
            prompt_file.write(json.dumps(prompts[item_id]) + "\n")
            prompt_file.flush()
            trajectory_file.write(json.dumps({"id": item_id, **trajectory}) + "\n")
            trajectory_file.flush()
            print(json.dumps(record), flush=True)

    records = list(read_records(quick_path).values())
    assert records, "no clips generated"
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["n_items"] = len(records)
    info["mean_seconds_per_item"] = sum(r["seconds"] for r in records) / len(records)
    info["mean_clip_fraction"] = sum(r.get("clip_fraction", 0) for r in records) / len(records)
    if "clap_quick" in records[0]:
        info["mean_clap_quick"] = sum(r["clap_quick"] for r in records) / len(records)
    info_path.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
