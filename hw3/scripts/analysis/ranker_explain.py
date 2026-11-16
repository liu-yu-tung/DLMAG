import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
sys.path.insert(0, str(HW3.parent / "hw1" / "src"))

RUNS = ["rescale12", "rescale12_seed1", "rescale12_seed2"]
FEATURES = ["rms_db", "centroid_hz", "percussive_share", "onset_rate_mix", "flatness", "share_above_12khz",
            "harmony_chroma_entropy_mean", "crest_factor_db"]
SCORES = {"clap8998": ("per_prompt.csv", "clap8998_cosine"), "probe": ("probe_rank.csv", "probe_score"),
          "mqa": ("mqa_per_item.csv", "mqascore"), "eval_clap": ("per_prompt.csv", "clap_cosine")}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Why do the best-of-3 rankers disagree? Within each caption (scores and features centred on the caption's "
        "mean over its 3 seeds, so caption difficulty drops out): the correlation between rankers' scores, and between each "
        "ranker's score and caption-free audio features (level, brightness, percussion, onset rate, noisiness, band above 12 kHz, "
        "chroma entropy, crest factor). Run with the hw1 venv and -P (needs the av-benchmark results only as csv). Writes "
        "results/analysis/ranker_explain.csv and ranker_explain_clips.csv.")
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def clip_features(path: str) -> dict:
    from analysis.dsp_extra import features as extra_features
    from hw1.features.handcrafted import SR, extract
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    hw1 = extract(librosa.resample(audio.T.mean(0), orig_sr=rate, target_sr=SR))
    extra = extra_features(audio.T, rate, None, "cpu", False)
    return {k: (extra | hw1)[k] for k in FEATURES}


def main():
    args = parse_args()
    out = HW3 / "results/analysis"
    clips = []
    for set_name in ("dev", "test"):
        for run in RUNS:
            frame = pd.DataFrame({"id": range(100)})
            for name, (file, column) in SCORES.items():
                frame[name] = pd.read_csv(HW3 / f"results/{set_name}__{run}/{file}").set_index("id")[column].reindex(frame["id"]).to_numpy()
            frame["set"], frame["run"] = set_name, run
            frame["path"] = [str(HW3 / f"outputs/{set_name}/{run}/{i}.wav") for i in frame["id"]]
            clips.append(frame)
    clips = pd.concat(clips, ignore_index=True)
    values = Parallel(n_jobs=args.jobs, verbose=2)(delayed(clip_features)(p) for p in clips["path"])
    clips = pd.concat([clips, pd.DataFrame(values)], axis=1).drop(columns="path")
    clips.to_csv(out / "ranker_explain_clips.csv", index=False)
    rows = []
    for set_name, part in clips.groupby("set"):
        part = part.dropna(subset=list(SCORES))
        part = part[part.groupby("id")["run"].transform("count") == len(RUNS)]
        centred = part[list(SCORES) + FEATURES] - part.groupby("id")[list(SCORES) + FEATURES].transform("mean")
        for score in SCORES:
            row = {"set": set_name, "score": score, "captions": part["id"].nunique()}
            for other in list(SCORES) + FEATURES:
                if other != score:
                    row[other] = float(centred[score].corr(centred[other], method="spearman"))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "ranker_explain.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
