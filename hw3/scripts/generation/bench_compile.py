import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from text_to_music.prompts import load_captions
from text_to_music.sampler import Guidance, generate, load_pipeline

HW3 = Path(__file__).resolve().parents[2]
# Below this the compile time per run and the mixed-numerics risk are not worth it.
MIN_SPEEDUP = 1.15


def time_clips(pipe, prompts: list[str], seeds: list[int]) -> tuple[list[float], list[torch.Tensor]]:
    times, audios = [], []
    for prompt, seed in zip(prompts, seeds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        audio, _ = generate(pipe, prompt, Guidance("cfg", 7.0), 100, 10.0, seed)
        torch.cuda.synchronize()
        times.append(round(time.perf_counter() - start, 3))
        audios.append(audio.float().cpu())
        print(f"{times[-1]:.2f} s", flush=True)
    return times, audios


def main():
    parser = argparse.ArgumentParser(description="Time SAO generation eager vs torch.compile on dev captions; "
                                                 "writes results/bench_compile.json with use_compile for generate.py --compile auto.")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HW3 / "results/bench_compile.json")
    args = parser.parse_args()
    captions = load_captions(HW3 / "data/dev.csv")
    ids = sorted(captions, key=int)[: args.clips]
    prompts, seeds = [captions[i] for i in ids], [int(i) for i in ids]

    pipe = load_pipeline()
    eager_times, eager_audio = time_clips(pipe, prompts, seeds)
    pipe.transformer.compile()
    compiled_times, compiled_audio = time_clips(pipe, prompts, seeds)

    # The first clip of each mode carries warm-up (cuDNN autotune, or the compile itself).
    eager = statistics.median(eager_times[1:])
    compiled = statistics.median(compiled_times[1:])
    differences = [float((a - b).abs().max()) for a, b in zip(eager_audio, compiled_audio)]
    relative_rms = [float((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()) for a, b in zip(eager_audio, compiled_audio)]
    finite = all(bool(torch.isfinite(a).all()) for a in compiled_audio)
    speedup = eager / compiled
    result = {
        "ids": ids, "eager_seconds": eager_times, "compiled_seconds": compiled_times,
        "eager_median_seconds": eager, "compiled_median_seconds": compiled, "speedup": round(speedup, 3),
        "compile_overhead_seconds": round(compiled_times[0] - compiled, 1),
        "max_abs_difference": differences, "relative_rms_difference": relative_rms, "compiled_finite": finite,
        "rule": f"use_compile if speedup >= {MIN_SPEEDUP} and compiled audio is finite",
        "use_compile": bool(speedup >= MIN_SPEEDUP and finite),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
