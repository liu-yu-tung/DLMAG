import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import training_set  # noqa: E402

# one word (or short regex) per fine genre; grouping is learned from audio confusion, not set by hand
TERMS = {"rock": r"rock", "metal": r"metal|heavy metal", "punk": r"punk", "indie": r"indie", "pop": r"pop", "electronic": r"electronic",
         "techno": r"techno", "house": r"house", "trance": r"trance", "dubstep": r"dubstep", "edm": r"edm", "disco": r"disco",
         "hip hop": r"hip[- ]?hop", "rap": r"rap", "trap": r"trap", "r&b": r"r ?& ?b|rnb", "funk": r"funk|funky", "soul": r"soul",
         "gospel": r"gospel", "jazz": r"jazz|jazzy", "blues": r"blues|bluesy", "swing": r"swing", "folk": r"folk",
         "country": r"country", "bluegrass": r"bluegrass", "classical": r"classical", "orchestral": r"orchestral|orchestra",
         "opera": r"opera|operatic", "latin": r"latin", "reggae": r"reggae", "salsa": r"salsa", "flamenco": r"flamenco",
         "bossa nova": r"bossa nova", "indian": r"indian|bollywood", "african": r"african|afrobeat", "arabic": r"arabic|middle eastern",
         "ambient": r"ambient", "soundtrack": r"soundtrack|movie score|film score", "lo-fi": r"lo-?fi", "children": r"children'?s|nursery",
         "christmas": r"christmas", "lullaby": r"lullaby", "march": r"march(?:ing)? band|military march"}
MIN_CLIPS = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genre map learned from audio: MusicCaps clips (probe training pool) whose caption or tag list names exactly one "
        "fine genre word; a 5-fold multi-class probe on each embedding gives the confusion matrix; genres are ordered by average-"
        "linkage clustering on 1 - symmetric confusion. Also the CLAP text cosine between genre names, to compare what the text "
        "side assumes. Run with the av-benchmark venv. Writes results/analysis/genre_terms.csv, genre_confusion_<kind>.csv, "
        "genre_text_cosine.csv and results/analysis/figures/genre_confusion.png.")
    parser.add_argument("--embeddings", default="mert,clap90")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS)
    return parser.parse_args()


def single_genre(train: pd.DataFrame) -> pd.Series:
    text = (train["caption"] + " " + train["aspect_list"]).str.lower()
    hits = pd.DataFrame({g: text.str.contains(rf"\b(?:{p})\b", regex=True) for g, p in TERMS.items()})
    hits.loc[hits["hip hop"], "rap"] = False
    hits.loc[hits["bossa nova"], ["jazz"]] = False
    one = hits.sum(axis=1) == 1
    return hits[one].idxmax(axis=1), hits.sum()


def order_by_confusion(matrix: pd.DataFrame) -> list:
    m = matrix.to_numpy()
    symmetric = (m + m.T) / 2
    distance = 1 - symmetric / np.sqrt(np.outer(np.diag(m), np.diag(m))).clip(1e-9)
    distance = ((distance + distance.T) / 2).clip(0)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return [matrix.index[i] for i in leaves_list(tree)]


def main():
    args = parse_args()
    train, _ = training_set()
    genre, counts = single_genre(train)
    keep = genre.value_counts()
    keep = keep[keep >= args.min_clips].index
    genre = genre[genre.isin(keep)]
    out = HW3 / "results/analysis"
    pd.DataFrame({"mentions": counts, "single_genre_clips": genre.value_counts().reindex(counts.index).fillna(0).astype(int)}) \
        .sort_values("mentions", ascending=False).to_csv(out / "genre_terms.csv")
    print(f"{len(genre)} single-genre clips, {len(keep)} genres with at least {args.min_clips}:", dict(genre.value_counts()))
    matrices = {}
    for kind in args.embeddings.split(","):
        ids = open(HW3 / f"outputs/cache/musiccaps_{kind}_ids.txt").read().split()
        index = {clip: i for i, clip in enumerate(ids)}
        clips = train.loc[genre.index, "ytid"]
        present = clips.map(index).notna().to_numpy()
        x = np.load(HW3 / f"outputs/cache/musiccaps_{kind}.npy")[clips[present].map(index).astype(int).to_numpy()]
        y = genre.to_numpy()[present]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        predicted = cross_val_predict(model, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        labels = sorted(keep)
        matrix = pd.DataFrame(confusion_matrix(y, predicted, labels=labels, normalize="true"), index=labels, columns=labels)
        matrices[kind] = matrix
        print(kind, "balanced accuracy", round(float(np.diag(matrix).mean()), 3))
    order = order_by_confusion(sum(matrices.values()) / len(matrices))
    for kind, matrix in matrices.items():
        matrix.loc[order, order].to_csv(out / f"genre_confusion_{kind}.csv")
    import torch
    from retrieval.retrieval import load_model
    clap = load_model("cpu")
    with torch.inference_mode():
        text = clap.get_text_embedding([f"{g} music" for g in order], use_tensor=False)
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    pd.DataFrame(text @ text.T, index=order, columns=order).to_csv(out / "genre_text_cosine.csv")
    plot(matrices, order, pd.DataFrame(text @ text.T, index=order, columns=order), out / "figures/genre_confusion.png")


def plot(matrices: dict, order: list, text: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(f"audio probe confusion ({k}, 5-fold, row = true)", m.loc[order, order]) for k, m in matrices.items()]
    panels.append(("CLAP text cosine between genre names", text))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    for ax, (title, m) in zip(axes, panels):
        im = ax.imshow(m.to_numpy(), cmap="viridis", vmin=0, vmax=1 if "confusion" in title else None)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
        ax.set_yticks(range(len(order)), order, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("->", path)


if __name__ == "__main__":
    main()
