import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
RESULTS = HW3 / "results"
RESAMPLES, SEED = 1000, 0
METRICS = {"clap_cosine": "CLAP", "kl_passt_softmax": "KL-PaSST", "mqascore": "MQAScore", "mqa_without_vocal": "MQAScore without vocal tags"}
OTHER_ATTRIBUTES = ["genre", "instrument", "mood_theme", "context"]


def table(set_name: str, run: str) -> pd.DataFrame:
    folder = RESULTS / f"{set_name}__{run}"
    scores = pd.read_csv(folder / "per_prompt.csv", dtype={"id": str}).set_index("id")
    mqa = pd.read_csv(folder / "mqa_per_item.csv", dtype={"id": str}).set_index("id")
    counts = mqa[[f"{a}_n" for a in OTHER_ATTRIBUTES]].to_numpy()
    means = mqa[[f"{a}_mean" for a in OTHER_ATTRIBUTES]].fillna(0).to_numpy()
    with np.errstate(invalid="ignore"):
        mqa["mqa_without_vocal"] = (means * counts).sum(axis=1) / counts.sum(axis=1)
    return scores.join(mqa[["mqascore", "mqa_without_vocal", "vocal_mean", "vocal_n"]])


def interval(values: np.ndarray) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    indices = np.random.default_rng(SEED).integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Split per-prompt metrics by whether the caption has vocal tags (MQAScore tag extraction), "
                                                 "with paired differences against a baseline run inside each group.")
    parser.add_argument("--set", dest="set_name", default="dev", choices=["dev", "test"])
    parser.add_argument("--runs", default="real,cfg7,rescale12,rescale12_best3_c8998")
    parser.add_argument("--baseline", default="cfg7")
    parser.add_argument("--out", type=Path, default=RESULTS / "analysis/vocal_split_dev.csv")
    args = parser.parse_args()
    baseline = table(args.set_name, args.baseline)
    groups = {"vocal": baseline.index[baseline["vocal_n"] > 0], "no_vocal": baseline.index[baseline["vocal_n"] == 0]}
    rows = []
    for run in args.runs.split(","):
        frame = table(args.set_name, run)
        for group, ids in groups.items():
            for column, label in METRICS.items():
                if column not in frame or frame.loc[ids, column].isna().all():
                    continue
                mean = frame.loc[ids, column].mean()
                difference = interval((frame.loc[ids, column] - baseline.loc[ids, column]).to_numpy()) if run != args.baseline else (0.0, 0.0, 0.0)
                rows.append({"run": run, "group": group, "prompts": len(ids), "metric": label, "mean": round(mean, 4),
                             "difference_vs_baseline": round(difference[0], 4), "low": round(difference[1], 4), "high": round(difference[2], 4)})
            vocal = frame.loc[ids, "vocal_mean"].dropna()
            if len(vocal):
                rows.append({"run": run, "group": group, "prompts": len(vocal), "metric": "MQAScore vocal tags", "mean": round(vocal.mean(), 4)})
    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    result.to_string(sys.stdout, index=False, na_rep="")
    print()


if __name__ == "__main__":
    main()
