import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import fit_probes, labels  # noqa: E402

MIN_CLIPS = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe validity (plan P1.2-P1.3): AUC of each caption-word probe on clips it never saw, labels from the dev "
        "captions. 'real' = the 100 real dev clips (from the MusicCaps cache; dev is excluded from probe training); other runs = "
        "generated dev clips from outputs/cache/runs/dev__<run>_<kind>.npy, when present. Groups need at least 8 positive and 8 "
        "negative dev captions. Run with the av-benchmark venv. Writes results/analysis/probe_validity.csv.")
    parser.add_argument("--runs", default="real,rescale12,cfg7")
    parser.add_argument("--embeddings", default="mert,clap90")
    return parser.parse_args()


def real_dev(kind: str) -> pd.DataFrame:
    meta = pd.read_csv(HW3 / "data/dev_meta.csv")
    ids = open(HW3 / f"outputs/cache/musiccaps_{kind}_ids.txt").read().split()
    index = {clip: i for i, clip in enumerate(ids)}
    features = np.load(HW3 / f"outputs/cache/musiccaps_{kind}.npy")
    return pd.DataFrame(features[meta["ytid"].map(index).astype(int).to_numpy()], index=meta["id"].astype(int).to_numpy())


def run_embeddings(run: str, kind: str):
    if run == "real":
        return real_dev(kind)
    path = HW3 / f"outputs/cache/runs/dev__{run}_{kind}.npy"
    if not path.exists():
        return None
    return pd.DataFrame(np.load(path), index=[int(i) for i in path.with_suffix(".ids.txt").read_text().split()])


def main():
    args = parse_args()
    dev = pd.read_csv(HW3 / "data/dev.csv")
    y = labels(dev["caption"]).set_index(dev["id"].astype(int))
    groups = [g for g in y.columns if MIN_CLIPS <= y[g].sum() <= len(y) - MIN_CLIPS]
    rows = []
    for kind in args.embeddings.split(","):
        probes = fit_probes(kind, groups)
        for run in args.runs.split(","):
            x = run_embeddings(run, kind)
            if x is None:
                print(f"skip {run} {kind}: no embeddings yet")
                continue
            for group, model in probes.items():
                target = y.loc[x.index, group].to_numpy()
                rows.append({"embedding": kind, "run": run, "group": group, "positives": int(target.sum()), "clips": len(target),
                             "auc": roc_auc_score(target, model.decision_function(x.to_numpy()))})
    table = pd.DataFrame(rows).pivot_table(index=["group", "positives"], columns=["embedding", "run"], values="auc")
    out = HW3 / "results/analysis/probe_validity.csv"
    table.to_csv(out)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    print("->", out)


if __name__ == "__main__":
    main()
