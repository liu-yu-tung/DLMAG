import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import labels, probe_model, training_set  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Where do the word-swap bundles come from? For each axis and other probe group, compares the spillover "
        "(results/analysis/edit_spillover.csv, change in seed-spread units) with two associations in the MusicCaps training pool "
        "(no dev or test-retrieved clips): caption co-occurrence = P(group | target word) - P(group | source word) in human "
        "captions; audio co-occurrence = the same contrast on cross-validated probe scores of the real audio (in score SDs). "
        "Spearman across axis x group pairs, and a two-predictor linear fit. Run with the av-benchmark venv. Writes "
        "results/analysis/bundle_origin.csv and bundle_origin.json.")
    parser.add_argument("--kind", default="mert", choices=("mert", "clap90"))
    return parser.parse_args()


def main():
    args = parse_args()
    spill = pd.read_csv(HW3 / "results/analysis/edit_spillover.csv")
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    train, _ = training_set()
    ids = open(HW3 / f"outputs/cache/musiccaps_{args.kind}_ids.txt").read().split()
    features = np.load(HW3 / f"outputs/cache/musiccaps_{args.kind}.npy")
    index = {clip: i for i, clip in enumerate(ids)}
    train = train[train["ytid"].map(index).notna().to_numpy()].reset_index(drop=True)
    x = features[train["ytid"].map(index).astype(int).to_numpy()]
    y = labels(train["caption"])
    groups = sorted(spill["group"].unique())
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = pd.DataFrame({g: cross_val_predict(probe_model(), x, y[g].to_numpy(), cv=folds, method="decision_function")
                           for g in groups})
    scores = (scores - scores.mean()) / scores.std()
    rows = []
    for axis in spill["axis"].unique():
        source, target = manifest[axis]["source"], manifest[axis]["target"]
        has_target = y[target].to_numpy()
        if source:
            has_source = y[source].to_numpy()
            positive, negative = has_target & ~has_source, has_source & ~has_target
        else:
            positive, negative = has_target, ~has_target
        for g in groups:
            if g in (source, target):
                continue
            s = spill[(spill["axis"] == axis) & (spill["group"] == g)].iloc[0]
            rows.append({"axis": axis, "group": g, "spillover": s["change_vs_seed_sd"],
                         "spillover_clear": bool(s["ci_low"] > 0 or s["ci_high"] < 0),
                         "caption_cooccurrence": float(y[g][positive].mean() - y[g][negative].mean()),
                         "audio_cooccurrence": float(scores[g][positive].mean() - scores[g][negative].mean()),
                         "n_target_clips": int(positive.sum()), "n_source_clips": int(negative.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(HW3 / "results/analysis/bundle_origin.csv", index=False)
    summary = {}
    for column in ("caption_cooccurrence", "audio_cooccurrence"):
        rho, p = spearmanr(table["spillover"], table[column])
        summary[column] = {"spearman": float(rho), "p": float(p)}
    clear = table[table["spillover_clear"]]
    for column in ("caption_cooccurrence", "audio_cooccurrence"):
        summary[f"{column}_sign_agreement_on_clear"] = float((np.sign(clear["spillover"]) == np.sign(clear[column])).mean())
    design = np.column_stack([np.ones(len(table)), (table["caption_cooccurrence"] - table["caption_cooccurrence"].mean())
                              / table["caption_cooccurrence"].std(),
                              (table["audio_cooccurrence"] - table["audio_cooccurrence"].mean()) / table["audio_cooccurrence"].std()])
    coef, *_ = np.linalg.lstsq(design, table["spillover"].to_numpy(), rcond=None)
    fitted = design @ coef
    summary["linear_fit"] = {"caption_coef": float(coef[1]), "audio_coef": float(coef[2]),
                             "r2": float(1 - ((table["spillover"] - fitted) ** 2).sum() / ((table["spillover"] - table["spillover"].mean()) ** 2).sum())}
    summary["pairs"], summary["clear_pairs"] = len(table), len(clear)
    summary["inter_association_spearman"] = float(spearmanr(table["caption_cooccurrence"], table["audio_cooccurrence"])[0])
    (HW3 / "results/analysis/bundle_origin.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(clear.sort_values("spillover").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
