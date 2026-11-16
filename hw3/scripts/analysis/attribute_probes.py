import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.caption_vocab import LEXICON, hits  # noqa: E402

MIN_POSITIVES = 80


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attribute probes: one logistic regression per caption-word group (scripts/analysis/caption_vocab.py), trained on cached "
        "MusicCaps audio embeddings with labels from the human captions. Training clips exclude every dev clip and every top-5 "
        "test-retrieved clip. Reports 5-fold AUC per group and embedding: which caption attributes an audio encoder can hear.")
    parser.add_argument("--embeddings", default="mert,clap90", help="cached names under outputs/cache/musiccaps_<name>.npy")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def training_set() -> tuple[pd.DataFrame, set]:
    meta = pd.read_csv(HW3 / "data/musiccaps_meta/musiccaps-public.csv")
    held_out = set(pd.read_csv(HW3 / "data/dev_meta.csv")["ytid"]) | set(pd.read_csv(HW3 / "results/retrieval_top5.csv")["youtube_id"])
    return meta[~meta["ytid"].isin(held_out)].reset_index(drop=True), held_out


def labels(captions: pd.Series) -> pd.DataFrame:
    lower = captions.str.lower()
    return pd.DataFrame({f"{category}: {group}": hits(lower, patterns).to_numpy() for category, group, _, patterns in LEXICON})


def probe_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))


def fit_probes(name: str, groups=None) -> dict:
    train, _ = training_set()
    y_all = labels(train["caption"])
    ids = open(HW3 / f"outputs/cache/musiccaps_{name}_ids.txt").read().split()
    features = np.load(HW3 / f"outputs/cache/musiccaps_{name}.npy")
    index = {clip: i for i, clip in enumerate(ids)}
    keep = train["ytid"].map(index).notna().to_numpy()
    x = features[train["ytid"][keep].map(index).astype(int).to_numpy()]
    probes = {}
    for group in groups or y_all.columns:
        y = y_all[group].to_numpy()[keep]
        if y.sum() >= MIN_POSITIVES:
            probes[group] = probe_model().fit(x, y)
    return probes


def main():
    args = parse_args()
    train, held_out = training_set()
    y_all = labels(train["caption"])
    rows = []
    for name in args.embeddings.split(","):
        ids = open(HW3 / f"outputs/cache/musiccaps_{name}_ids.txt").read().split()
        features = np.load(HW3 / f"outputs/cache/musiccaps_{name}.npy")
        index = {clip: i for i, clip in enumerate(ids)}
        keep = train["ytid"].map(index).notna().to_numpy()
        x = features[train["ytid"][keep].map(index).astype(int).to_numpy()]
        for group in y_all.columns:
            y = y_all[group].to_numpy()[keep]
            if y.sum() < MIN_POSITIVES:
                continue
            scores = np.zeros(len(y))
            for fold_train, fold_test in StratifiedKFold(args.folds, shuffle=True, random_state=0).split(x, y):
                model = probe_model()
                scores[fold_test] = model.fit(x[fold_train], y[fold_train]).predict_proba(x[fold_test])[:, 1]
            rows.append({"embedding": name, "group": group, "positives": int(y.sum()), "clips": len(y), "auc": roc_auc_score(y, scores)})
    table = pd.DataFrame(rows).pivot_table(index=["group", "positives"], columns="embedding", values="auc").reset_index()
    table = table.sort_values(table.columns[-1], ascending=False)
    out = HW3 / "results/analysis/attribute_probes_cv.csv"
    table.to_csv(out, index=False)
    print(f"training clips {len(train)} (held out {len(held_out)} dev and test-retrieved ids)")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
