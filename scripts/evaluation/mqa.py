#!/usr/bin/env python
"""Score a directory of clips (<id>.wav) with MQAScore (Music Flamingo, per-tag protocol).

Builds the per-tag input from a tags jsonl (output of
external/MQAScore/scripts/extract_tags.py), runs external/MQAScore/scripts/score.py
in the Music Flamingo venv with bf16 + partial CPU offload, and writes
results/<run>/mqa_summary.json and results/<run>/mqa_per_item.csv.
Re-running the same --run resumes (score.py skips finished items).

Usage notes: docs/mqa_notes.md
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
MQA_REPO = HW3 / "external" / "MQAScore"
DEFAULT_ATTRIBUTES = ["genre", "instrument", "mood_theme", "vocal", "context"]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name; outputs go to results/<run>/")
    p.add_argument("--audio-dir", required=True, help="directory with <id>.wav for every id in --tags")
    p.add_argument("--tags", required=True, help="tags jsonl: {id, tags: {attribute: [...]}} per line")
    p.add_argument("--python", default=str(MQA_REPO / ".venv-mf" / "bin" / "python"),
                   help="python of the Music Flamingo venv")
    p.add_argument("--gpu-memory-gib", type=float, default=13.0, help="GPU weight cap; the rest of the model is offloaded to CPU")
    p.add_argument("--cpu-memory-gib", type=float, default=40.0)
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES))
    p.add_argument("--limit", type=int, default=None, help="score only the first N items")
    p.add_argument("--summarize-only", action="store_true", help="skip scoring; rebuild summary and csv from the score file")
    p.add_argument("--allow-busy-gpu", action="store_true", help="skip the free-GPU-memory check")
    return p.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def free_gpu_gib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.split(",")
    used, total = (float(x) for x in out)
    return (total - used) / 1024


def build_items(tags_path, audio_dir, limit):
    items = []
    for row in read_jsonl(tags_path)[:limit]:
        audio = Path(audio_dir) / f"{row['id']}.wav"
        assert audio.exists(), f"missing audio {audio}"
        items.append({"id": str(row["id"]), "audio": str(audio.resolve()), "tags": row["tags"]})
    assert items, f"no items in {tags_path}"
    assert len({it["id"] for it in items}) == len(items), "duplicate ids"
    return items


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(score_path, run_dir, attributes, n_expected):
    rows = read_jsonl(score_path)
    per_item = pd.DataFrame([
        {"id": r["id"], "n_tags": r["n_tags"], "mqascore": r["mean_score"],
         **{f"{a}_mean": r["attribute_mean"].get(a) for a in attributes},
         **{f"{a}_n": len(r["per_attribute"].get(a, [])) for a in attributes}}
        for r in rows])
    per_item = per_item.sort_values("id", key=lambda s: s.map(lambda x: (len(x), x)))
    per_item.to_csv(run_dir / "mqa_per_item.csv", index=False)

    scored = per_item.dropna(subset=["mqascore"])
    summary = {
        "n_items_expected": n_expected,
        "n_items_scored": len(per_item),
        "n_items_with_tags": len(scored),
        "n_questions": int(per_item["n_tags"].sum()),
        "mqascore_mean": float(scored["mqascore"].mean()),
        "mqascore_ci95": bootstrap_ci(scored["mqascore"]),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "items"},
        "attribute_mean": {}, "attribute_n_items": {},
    }
    for a in attributes:
        present = per_item[f"{a}_mean"].dropna()
        summary["attribute_n_items"][a] = len(present)
        summary["attribute_mean"][a] = float(present.mean()) if len(present) else None
    (run_dir / "mqa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]
    run_dir = HW3 / "results" / args.run
    work_dir = HW3 / "outputs" / "mqa" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(args.tags, args.audio_dir, args.limit)
    score_path = work_dir / "score.jsonl"

    if not args.summarize_only:
        input_path = work_dir / "items.jsonl"
        input_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
        if not args.allow_busy_gpu:
            need = args.gpu_memory_gib + 2.5
            assert free_gpu_gib() >= need, f"GPU has {free_gpu_gib():.1f} GiB free, need about {need:.1f}; rerun when it is free"
        assert Path(args.python).exists(), f"no venv python at {args.python}"
        command = [args.python, "scripts/score.py", "--model", "mf", "--protocol", "pertag",
                   "--attributes", ",".join(attributes),
                   "--input", str(input_path), "--output", str(score_path),
                   "--cache-dir", str(HW3 / "outputs" / "mqa" / "audio_cache"),
                   "--gpu-memory-gib", str(args.gpu_memory_gib), "--cpu-memory-gib", str(args.cpu_memory_gib)]
        started = time.time()
        subprocess.run(command, cwd=MQA_REPO, check=True)
        (work_dir / "wall_seconds.txt").write_text(f"{time.time() - started:.1f}\n")

    assert score_path.exists(), f"no scores at {score_path}"
    summary = summarize(score_path, run_dir, attributes, len(items))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
