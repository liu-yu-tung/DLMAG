import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HW3 = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recording-quality probe (diagnostic, never a submission): copy a run and degrade only the clips whose "
        "caption mentions recording quality (low-pass plus Gaussian noise at a fixed SNR). Re-scoring the copy shows "
        "whether a metric rewards matching recording conditions rather than music.")
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), default="dev")
    parser.add_argument("--run", default="rescale12")
    parser.add_argument("--name", default="rescale12_degraded")
    parser.add_argument("--cutoff-hz", type=float, default=4000.0)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def degrade(audio: np.ndarray, rate: int, cutoff_hz: float, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    filtered = sosfiltfilt(butter(4, cutoff_hz, fs=rate, output="sos"), audio, axis=0)
    signal_power = float(np.mean(filtered ** 2))
    noise = rng.standard_normal(filtered.shape) * np.sqrt(signal_power / 10 ** (snr_db / 10))
    return np.clip(filtered + noise, -1.0, 1.0)


def main():
    args = parse_args()
    classes = pd.read_csv(HW3 / f"results/analysis/caption_classes_{args.set_name}.csv", dtype={"id": str})
    flagged = set(classes.loc[classes["mentions_recording_quality"].astype(bool), "id"])
    source = HW3 / "outputs" / args.set_name / args.run
    target = HW3 / "outputs" / args.set_name / args.name
    target.mkdir(parents=True, exist_ok=True)
    clips = sorted(source.glob("*.wav"), key=lambda path: int(path.stem))
    assert len(clips) == 100, len(clips)
    for clip in clips:
        if clip.stem not in flagged:
            shutil.copyfile(clip, target / clip.name)
            continue
        audio, rate = sf.read(clip, dtype="float32")
        rng = np.random.default_rng([args.seed, int(clip.stem)])
        sf.write(target / clip.name, degrade(audio, rate, args.cutoff_hz, args.snr_db, rng), rate, subtype="PCM_16")
    record_dir = HW3 / "results" / f"{args.set_name}__{args.name}"
    record_dir.mkdir(parents=True, exist_ok=True)
    summary = {"set": args.set_name, "source_run": args.run, "cutoff_hz": args.cutoff_hz, "snr_db": args.snr_db,
               "seed": args.seed, "n_degraded": len(flagged), "degraded_ids": sorted(flagged, key=int),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (record_dir / "degrade.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(flagged)} of {len(clips)} clips degraded -> {target}")


if __name__ == "__main__":
    main()
