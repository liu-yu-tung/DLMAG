import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
RUNS = ["rescale12", "rescale12_seed1", "rescale12_seed2"]
RANKERS = ("best3_c8998", "best3_probe", "best3_mqa")


def parse_args():
    parser = argparse.ArgumentParser(
        description="The three best-of-3 rankers side by side (CLAP 89.98 checkpoint, attribute probes, MQAScore): for each, the "
        "change vs seed 0 in CLAP, KL and MQAScore (paired bootstrap), and how often two rankers pick the same seed (chance 1/3). "
        "Uses each seed's per-item metrics, so every ranker is scored on every metric. Run with the av-benchmark venv. Writes "
        "results/analysis/ranker_compare.csv.")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for set_name in ("dev", "test"):
        metric = {}
        for run in RUNS:
            per_prompt = pd.read_csv(HW3 / f"results/{set_name}__{run}/per_prompt.csv").set_index("id")
            mqa = pd.read_csv(HW3 / f"results/{set_name}__{run}/mqa_per_item.csv").set_index("id")["mqascore"]
            metric[run] = pd.DataFrame({"clap": per_prompt["clap_cosine"], "kl": per_prompt["kl_passt_softmax"], "mqa": mqa})
        picks = {name: pd.read_csv(HW3 / f"results/{set_name}__rescale12_{name}/selection.csv").set_index("id")["chosen_run"]
                 for name in RANKERS}
        for name, pick in picks.items():
            row = {"set": set_name, "ranker": name}
            for m in ("clap", "kl", "mqa"):
                chosen = np.array([metric[pick[i]].loc[i, m] for i in pick.index])
                delta = chosen - metric["rescale12"].loc[pick.index, m].to_numpy()
                delta = delta[np.isfinite(delta)]
                row[f"{m}_n"] = len(delta)
                means = delta[rng.integers(0, len(delta), (args.resamples, len(delta)))].mean(1)
                row |= {m: float(delta.mean()), f"{m}_ci_low": float(np.quantile(means, 0.025)), f"{m}_ci_high": float(np.quantile(means, 0.975))}
            for other in RANKERS:
                row[f"same_pick_as_{other}"] = float((pick == picks[other].reindex(pick.index)).mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(HW3 / "results/analysis/ranker_compare.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
