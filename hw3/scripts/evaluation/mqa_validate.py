#!/usr/bin/env python
"""Time the offloaded Music Flamingo and write per-item scores in score.py's output format.

Loads the model once with --gpu-memory-gib, then scores each --jobs pair (input_jsonl:output_jsonl)
and prints seconds per question. Compare outputs with external/MQAScore/tests/compare_reference.py.
Run with the Music Flamingo venv python and `-P` (keeps the script folder off sys.path). Usage notes: docs/mqa_notes.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

MQA_REPO = Path(__file__).resolve().parents[2] / "external" / "MQAScore"
sys.path.insert(0, str(MQA_REPO))

import torch  # noqa: E402

from mqascore import MQAScorer, load_backend  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-memory-gib", type=float, required=True)
    p.add_argument("--cpu-memory-gib", type=float, default=40.0)
    p.add_argument("--jobs", nargs="+", required=True, help="input_jsonl:output_jsonl pairs (score.py pertag input format)")
    p.add_argument("--cache-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    started = time.time()
    backend = load_backend("mf", gpu_memory_gib=args.gpu_memory_gib, cpu_memory_gib=args.cpu_memory_gib)
    device_map = backend.model.hf_device_map
    on_cpu = [name for name, device in device_map.items() if device == "cpu"]
    print(f"load {time.time() - started:.1f}s; {len(on_cpu)} of {len(device_map)} module groups on cpu; "
          f"gpu allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    scorer = MQAScorer(backend, cache_dir=args.cache_dir)
    for job in args.jobs:
        source, target = (Path(x) for x in job.split(":"))
        items = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
        target.unlink(missing_ok=True)
        n_questions, job_started = 0, time.time()
        for item in items:
            result = scorer.score_tags(item["audio"], item["tags"])
            n_questions += result["n_tags"]
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps({**item, **result}, ensure_ascii=False) + "\n")
        seconds = time.time() - job_started
        print(f"{target}: {len(items)} items, {n_questions} questions, {seconds:.1f}s, "
              f"{seconds / n_questions:.2f} s/question (audio loading included); "
              f"peak gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
