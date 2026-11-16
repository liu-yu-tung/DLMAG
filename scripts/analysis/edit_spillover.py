import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import fit_probes  # noqa: E402
from analysis.edit_effects import probe_scores  # noqa: E402
from selection.probe_rank import valid_groups  # noqa: E402

AXES = ["voice_male_to_female", "guitar_to_piano", "acoustic_to_electric", "soft_to_loud", "happy_to_sad", "fast_to_slow",
        "low_quality_to_studio", "insert_jazz", "insert_country"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Which other probes move when one word is swapped? Per axis (pooled registered + extended pairs) and probe group "
        "(groups that passed the held-out check, MERT AUC >= 0.75): mean edited-minus-original log-odds, the share of pairs moving "
        "that way, and the mean in units of the seed spread (same caption, seeds 1-2 vs 0). Run with the av-benchmark venv. Writes "
        "results/analysis/edit_spillover.csv.")
    parser.add_argument("--kind", default="mert", choices=("mert", "clap90"))
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    groups = valid_groups(0.75)
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    probes = fit_probes(args.kind, groups)
    base = probe_scores(probes, "rescale12", args.kind)
    seeds = [probe_scores(probes, f"rescale12_seed{k}", args.kind) for k in (1, 2)]
    rng = np.random.default_rng(0)
    rows = []
    for axis in AXES:
        entry = manifest[axis]
        parts = [probe_scores(probes, f"edit_{axis}{s}", args.kind) for s in ("", "_ext")]
        edited = pd.concat([p for p in parts if p is not None])
        ids = [i for i in entry["ids"] + entry.get("ext_ids", []) if i in edited.index]
        change = edited.loc[ids] - base.loc[ids]
        noise = pd.concat([s.loc[ids] - base.loc[ids] for s in seeds])
        for group in groups:
            delta = change[group].to_numpy()
            means = delta[rng.integers(0, len(delta), (args.resamples, len(delta)))].mean(1)
            rows.append({"axis": axis, "group": group, "role": "target" if group == entry["target"] else
                         "source" if group == entry["source"] else "other", "n": len(delta), "mean_change": float(delta.mean()),
                         "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
                         "share_up": float((delta > 0).mean()), "seed_sd": float(noise[group].std(ddof=1)),
                         "change_vs_seed_sd": float(delta.mean() / noise[group].std(ddof=1))})
    table = pd.DataFrame(rows)
    table.to_csv(HW3 / f"results/analysis/edit_spillover{'' if args.kind == 'mert' else '_' + args.kind}.csv", index=False)
    clear = table[(table["ci_low"] > 0) | (table["ci_high"] < 0)]
    pd.set_option("display.width", 200)
    for axis, part in clear.groupby("axis", sort=False):
        part = part.reindex(part["change_vs_seed_sd"].abs().sort_values(ascending=False).index)
        print(f"\n{axis} (n={part['n'].iloc[0]}): probes whose 95% interval excludes 0")
        print(part[["group", "role", "mean_change", "ci_low", "ci_high", "share_up", "change_vs_seed_sd"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
