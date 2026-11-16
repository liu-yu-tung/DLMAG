import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import fit_probes, labels  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe-ranked best-of-N (predictions docs/predictions.toml, 11-15 20:04): per caption, the word groups it asks "
        "for (scripts/analysis/caption_vocab.py lexicon) that passed the held-out real-dev check (results/analysis/probe_validity.csv, MERT "
        "AUC >= --min-auc); each seed's score = mean MERT probe log-odds over those groups. Captions with no valid group score 0 for "
        "every seed, so the first run (seed 0) is kept. Writes results/<set>__<run>/probe_rank.csv per candidate, runs "
        "scripts/selection/select_best_of_n.py, and assembles the selection's per_prompt.csv from the candidates' rows. Run with the "
        "av-benchmark venv.")
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--runs", default="rescale12,rescale12_seed1,rescale12_seed2")
    parser.add_argument("--name", default="rescale12_best3_probe")
    parser.add_argument("--min-auc", type=float, default=0.75)
    return parser.parse_args()


def valid_groups(min_auc: float) -> list:
    table = pd.read_csv(HW3 / "results/analysis/probe_validity.csv", header=[0, 1], index_col=[0, 1])
    auc = table[("mert", "real")].droplevel(1)
    return sorted(auc[auc >= min_auc].index)


def main():
    args = parse_args()
    runs = args.runs.split(",")
    groups = valid_groups(args.min_auc)
    captions = pd.read_csv(HW3 / f"data/{args.set_name}.csv")
    asked = labels(captions["caption"]).set_index(captions["id"].astype(int))[groups]
    probes = fit_probes("mert", groups)
    for run in runs:
        path = HW3 / f"outputs/cache/runs/{args.set_name}__{run}_mert.npy"
        ids = [int(i) for i in path.with_suffix(".ids.txt").read_text().split()]
        x = np.load(path)
        log_odds = pd.DataFrame({g: probes[g].decision_function(x) for g in groups}, index=ids)
        mask = asked.loc[ids].to_numpy()
        score = np.where(mask.any(1), (log_odds.to_numpy() * mask).sum(1) / np.maximum(mask.sum(1), 1), 0.0)
        frame = pd.DataFrame({"id": ids, "probe_score": score})
        if args.set_name == "dev":
            frame["groups_asked"] = mask.sum(1)
            frame["groups_heard"] = ((log_odds.to_numpy() > 0) & mask).sum(1)
        frame.to_csv(HW3 / f"results/{args.set_name}__{run}/probe_rank.csv", index=False)
    print(f"{len(groups)} valid groups; captions with at least one: {int(asked.any(axis=1).sum())} of {len(asked)}")
    subprocess.run([sys.executable, str(HW3 / "scripts/selection/select_best_of_n.py"), "--set", args.set_name, "--runs", args.runs,
                    "--name", args.name, "--score-file", "probe_rank.csv", "--score", "probe_score"], check=True)
    selection = pd.read_csv(HW3 / f"results/{args.set_name}__{args.name}/selection.csv", dtype={"id": str}).set_index("id")
    rows = {run: pd.read_csv(HW3 / f"results/{args.set_name}__{run}/per_prompt.csv", dtype={"id": str}).set_index("id") for run in runs}
    per_prompt = pd.DataFrame([rows[run].loc[i] for i, run in selection["chosen_run"].items()])
    per_prompt.index.name = "id"
    per_prompt.to_csv(HW3 / f"results/{args.set_name}__{args.name}/per_prompt.csv")
    base, seed_mean = rows[runs[0]].loc[per_prompt.index], sum(rows[r].loc[per_prompt.index] for r in runs) / len(runs)
    rng = np.random.default_rng(0)
    for metric in ("clap_cosine", "kl_passt_softmax"):
        for label, against in (("vs seed 0", base), ("vs seed mean", seed_mean)):
            diff = (per_prompt[metric] - against[metric]).to_numpy(dtype=float)
            boot = diff[rng.integers(0, len(diff), (1000, len(diff)))].mean(1)
            print(f"{metric} {label}: {diff.mean():+.4f} [{np.quantile(boot, 0.025):+.4f}, {np.quantile(boot, 0.975):+.4f}]")


if __name__ == "__main__":
    main()
