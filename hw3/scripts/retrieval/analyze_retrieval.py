"""Hubness of the CLAP test-caption retrieval, as analysis only; results/retrieval.json is not touched. See docs/analysis_state.txt."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from text_to_music.eval_common import CACHE_DIR, DATA_DIR, HW3_ROOT, RESULTS_DIR, TEST_CSV, load_retrieval

OUT_DIR = RESULTS_DIR / "analysis" / "retrieval"
PRIVATE_DIR = HW3_ROOT / "outputs" / "analysis_private"
META_CSV = DATA_DIR / "musiccaps_meta" / "musiccaps-public.csv"
PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00", "grey": "#777777"}
QUALITY_WORDS = ["low quality", "noisy", "recording", "mono"]
FIGURE_SIZE = (8, 4.5)
TOP_HUBS = 10
CSLS_K = 10
INVERTED_SOFTMAX_BETAS = (10, 20, 50)
OCCURRENCE_KS = (1, 5, 10)


def load_embeddings() -> tuple[np.ndarray, list[str], np.ndarray]:
    audio = np.load(CACHE_DIR / "musiccaps_clap90.npy")
    audio_ids = (CACHE_DIR / "musiccaps_clap90_ids.txt").read_text().split()
    text = np.load(CACHE_DIR / "test_text_clap90.npy")
    text_ids = (CACHE_DIR / "test_text_clap90_ids.txt").read_text().split()
    assert audio.shape[0] == len(audio_ids) and text.shape[0] == len(text_ids)
    assert text_ids == [str(i) for i in range(len(text_ids))]
    return audio, audio_ids, text


def normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def k_occurrence(scores: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return np.bincount(top.ravel(), minlength=scores.shape[1])


def hubness_table(scores: np.ndarray) -> pd.DataFrame:
    rows = []
    n_captions, n_clips = scores.shape
    for k in OCCURRENCE_KS:
        counts = k_occurrence(scores, k)
        order = np.sort(counts)[::-1]
        rows.append({
            "k": k,
            "captions": n_captions,
            "distinct_clips": int((counts > 0).sum()),
            "max_occurrence": int(order[0]),
            "skewness": float(skew(counts)),
            "top5_clip_occurrences": int(order[:5].sum()),
            "top5_clip_occurrences_share_of_slots": float(order[:5].sum() / (n_captions * k)),
            "top5_clips_captions_served": int(np.isin(np.argpartition(-scores, k - 1, axis=1)[:, :k], np.argsort(-counts, kind="stable")[:5]).any(axis=1).sum()),
            "uniform_expected_distinct": float(n_clips * (1 - (1 - 1 / n_clips) ** (n_captions * k))),
        })
    return pd.DataFrame(rows)


def hubness_figure(scores: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=True)
    for axis, k in zip(axes, OCCURRENCE_KS):
        counts = k_occurrence(scores, k)
        values, frequency = np.unique(counts, return_counts=True)
        axis.bar(values, frequency, color=PALETTE["blue"], width=0.8)
        axis.set_yscale("log")
        axis.set_xlabel(f"times a clip is in a top-{k}")
        axis.text(0.97, 0.95, f"max {counts.max()}\nskew {skew(counts):.1f}", transform=axis.transAxes, ha="right", va="top")
    axes[0].set_ylabel("number of clips (log scale)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_occurrence_hist.png", dpi=200)
    plt.close(fig)


def describe_hubs(scores, audio, audio_ids, text, meta) -> tuple[pd.DataFrame, dict]:
    counts_1, counts_5 = k_occurrence(scores, 1), k_occurrence(scores, 5)
    top5_sets = np.argsort(-scores, axis=1)[:, :5]
    top1 = scores.argmax(axis=1)
    mean_cosine = scores.mean(axis=0)
    to_audio_centroid = audio @ normalise(audio.mean(axis=0, keepdims=True))[0]
    to_text_centroid = audio @ normalise(text.mean(axis=0, keepdims=True))[0]
    rows = []
    for clip in np.argsort(-counts_1, kind="stable")[:TOP_HUBS]:
        record = meta.loc[audio_ids[clip]]
        rows.append({
            "youtube_id": audio_ids[clip],
            "n1": int(counts_1[clip]),
            "n5": int(counts_5[clip]),
            "musiccaps_caption": record["caption"],
            "aspect_list": record["aspect_list"],
            "top1_caption_ids": " ".join(str(i) for i in np.flatnonzero(top1 == clip)),
            "top5_caption_ids": " ".join(str(i) for i in np.flatnonzero((top5_sets == clip).any(axis=1))),
            "mean_cosine_to_100_captions": float(mean_cosine[clip]),
            "percentile_of_mean_cosine_among_clips": float((mean_cosine < mean_cosine[clip]).mean() * 100),
            "cosine_to_text_centroid": float(to_text_centroid[clip]),
            "percentile_of_text_centroid_cosine": float((to_text_centroid < to_text_centroid[clip]).mean() * 100),
            "cosine_to_audio_centroid": float(to_audio_centroid[clip]),
            "percentile_of_audio_centroid_cosine": float((to_audio_centroid < to_audio_centroid[clip]).mean() * 100),
        })
    average_clip = {
        "mean_cosine_to_100_captions": float(mean_cosine.mean()),
        "cosine_to_text_centroid": float(to_text_centroid.mean()),
        "cosine_to_audio_centroid": float(to_audio_centroid.mean()),
        "spearman_n10_vs_mean_cosine_to_captions": float(spearmanr(k_occurrence(scores, 10), mean_cosine)[0]),
        "spearman_n10_vs_cosine_to_audio_centroid": float(spearmanr(k_occurrence(scores, 10), to_audio_centroid)[0]),
        "spearman_n10_vs_cosine_to_text_centroid": float(spearmanr(k_occurrence(scores, 10), to_text_centroid)[0]),
    }
    return pd.DataFrame(rows), average_clip


def reduction_table(audio: np.ndarray, text: np.ndarray, plain_scores: np.ndarray) -> pd.DataFrame:
    centred = normalise(text - text.mean(axis=0)) @ normalise(audio - audio.mean(axis=0)).T
    clip_neighbourhood = np.sort(plain_scores, axis=0)[-CSLS_K:].mean(axis=0)
    caption_neighbourhood = np.sort(plain_scores, axis=1)[:, -CSLS_K:].mean(axis=1)
    csls = 2 * plain_scores - clip_neighbourhood[None, :] - caption_neighbourhood[:, None]

    methods = {"plain_cosine": plain_scores, "centering": centred, f"csls_k{CSLS_K}": csls}
    for beta in INVERTED_SOFTMAX_BETAS:
        logits = beta * plain_scores
        weights = np.exp(logits - logits.max(axis=0, keepdims=True))
        methods[f"inverted_softmax_beta{beta}"] = weights / weights.sum(axis=0, keepdims=True)

    plain_top1 = plain_scores.argmax(axis=1)
    rows = []
    for name, matrix in methods.items():
        top1 = matrix.argmax(axis=1)
        counts = np.bincount(top1, minlength=matrix.shape[1])
        rows.append({
            "method": name,
            "distinct_top1_clips": int((counts > 0).sum()),
            "max_n1": int(counts.max()),
            "top1_changed_vs_plain": int((top1 != plain_top1).sum()),
            "mean_plain_cosine_of_picks": float(plain_scores[np.arange(len(top1)), top1].mean()),
            "n1_skewness": float(skew(counts)),
        })
    return pd.DataFrame(rows)


def modality_gap(audio, text, scores, seed: int = 0) -> tuple[dict, dict[str, np.ndarray]]:
    audio_mean, text_mean = audio.mean(axis=0), text.mean(axis=0)
    rng = np.random.default_rng(seed)
    matched = scores.max(axis=1)
    first, second = rng.integers(0, audio.shape[0], 20000), rng.integers(0, audio.shape[0], 20000)
    audio_audio = (audio[first] * audio[second])[first != second].sum(axis=1)
    text_text = (text @ text.T)[np.triu_indices(text.shape[0], 1)]
    summary = {
        "centroid_distance": float(np.linalg.norm(audio_mean - text_mean)),
        "audio_centroid_norm": float(np.linalg.norm(audio_mean)),
        "text_centroid_norm": float(np.linalg.norm(text_mean)),
        "cosine_between_centroids": float(audio_mean @ text_mean / np.linalg.norm(audio_mean) / np.linalg.norm(text_mean)),
        "mean_cosine_matched_top1": float(matched.mean()),
        "mean_cosine_all_caption_clip_pairs": float(scores.mean()),
        "std_cosine_all_caption_clip_pairs": float(scores.std()),
        "mean_cosine_audio_audio_random_pairs": float(audio_audio.mean()),
        "mean_cosine_text_text_all_pairs": float(text_text.mean()),
        "matched_top1_z_over_all_pairs": float((matched.mean() - scores.mean()) / scores.std()),
    }
    return summary, {"matched_top1": matched, "all_caption_clip_pairs": scores.ravel()}


def gap_figure(distributions: dict[str, np.ndarray]) -> None:
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    bins = np.linspace(-0.2, 0.65, 70)
    axis.hist(distributions["all_caption_clip_pairs"], bins=bins, density=True, alpha=0.6, color=PALETTE["blue"], label="all caption-clip pairs (536,700)")
    axis.hist(distributions["matched_top1"], bins=bins, density=True, alpha=0.7, color=PALETTE["orange"], label="top-1 clip per caption (100)")
    axis.set_xlabel("cosine between caption embedding and audio embedding")
    axis.set_ylabel("density")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "modality_gap_cosines.png", dpi=200)
    plt.close(fig)
    pd.DataFrame({name: pd.Series(values).describe() for name, values in distributions.items()}).T.to_csv(OUT_DIR / "modality_gap_cosines.csv")


def caption_style(captions_by_set: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, openings = [], []
    for name, captions in captions_by_set.items():
        words = np.array([len(caption.split()) for caption in captions])
        row = {"set": name, "n": len(captions), "words_mean": words.mean(), "words_median": float(np.median(words)),
               "words_std": words.std(), "words_min": words.min(), "words_max": words.max()}
        lowered = [caption.lower() for caption in captions]
        for word in QUALITY_WORDS:
            row[f"share_with_{word.replace(' ', '_')}"] = float(np.mean([word in caption for caption in lowered]))
        row["share_with_any_quality_word"] = float(np.mean([any(word in caption for word in QUALITY_WORDS) for caption in lowered]))
        rows.append(row)
        counter = Counter(" ".join(caption.split()[:3]).lower() for caption in captions)
        for rank, (opening, count) in enumerate(counter.most_common(5), 1):
            openings.append({"set": name, "rank": rank, "first_three_words": opening, "count": count, "share": count / len(captions)})
    return pd.DataFrame(rows), pd.DataFrame(openings)


def style_figure(test_captions: list[str], dev_captions: list[str]) -> None:
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    bins = np.arange(0, 126, 5)
    axis.hist([len(c.split()) for c in dev_captions], bins=bins, alpha=0.6, color=PALETTE["orange"], label="dev (MusicCaps captions)")
    axis.hist([len(c.split()) for c in test_captions], bins=bins, alpha=0.6, color=PALETTE["blue"], label="test")
    axis.set_xlabel("words per caption")
    axis.set_ylabel("captions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "caption_word_counts.png", dpi=200)
    plt.close(fig)
    pd.DataFrame({"set": ["test"] * len(test_captions) + ["dev"] * len(dev_captions),
                  "words": [len(c.split()) for c in test_captions + dev_captions]}).to_csv(OUT_DIR / "caption_word_counts.csv", index=False)


def main() -> None:
    argparse.ArgumentParser(description="Hubness, hubness reduction, modality gap and caption style of the CLAP test retrieval; writes results/analysis/retrieval/.").parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, audio_ids, text = load_embeddings()
    for name, matrix in (("audio", audio), ("text", text)):
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1, atol=1e-4), f"{name} embeddings are not L2-normalised (norms {norms.min()}..{norms.max()})"
    meta = pd.read_csv(META_CSV).drop_duplicates("ytid").set_index("ytid")
    test_captions = pd.read_csv(TEST_CSV).sort_values("id")["caption"].tolist()
    dev_captions = pd.read_csv(DATA_DIR / "dev.csv").sort_values("id")["caption"].tolist()

    scores = text @ audio.T
    plain_top1 = scores.argmax(axis=1)
    submitted = load_retrieval()
    assert all(audio_ids[plain_top1[i]] == submitted[str(i)] for i in range(len(plain_top1))), "plain top-1 differs from results/retrieval.json"

    hub_counts = hubness_table(scores)
    hub_counts.to_csv(OUT_DIR / "k_occurrence_summary.csv", index=False)
    hubness_figure(scores)
    pd.DataFrame({"youtube_id": audio_ids, **{f"n{k}": k_occurrence(scores, k) for k in OCCURRENCE_KS}}).to_csv(OUT_DIR / "k_occurrence_per_clip.csv", index=False)

    hubs, average_clip = describe_hubs(scores, audio, audio_ids, text, meta)
    hubs.to_csv(OUT_DIR / "hub_clips.csv", index=False)
    (OUT_DIR / "hub_average_clip.json").write_text(json.dumps(average_clip, indent=1) + "\n")

    reduction = reduction_table(audio, text, scores)
    reduction.to_csv(OUT_DIR / "hubness_reduction.csv", index=False)

    gap, distributions = modality_gap(audio, text, scores)
    gap_figure(distributions)
    (OUT_DIR / "modality_gap.json").write_text(json.dumps(gap, indent=1) + "\n")

    style, openings = caption_style({"test": test_captions, "dev": dev_captions, "musiccaps_all": meta["caption"].tolist()})
    style.to_csv(OUT_DIR / "caption_style.csv", index=False)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    openings.to_csv(PRIVATE_DIR / "caption_openings.csv", index=False)
    style_figure(test_captions, dev_captions)
    exact_matches = {"test": len(set(test_captions) & set(meta["caption"])), "dev": len(set(dev_captions) & set(meta["caption"]))}

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
    print(hub_counts.to_string(index=False), "\n")
    print(hubs.drop(columns=["aspect_list"]).to_string(index=False), "\n")
    print(json.dumps(average_clip, indent=1), "\n")
    print(reduction.to_string(index=False), "\n")
    print(json.dumps(gap, indent=1), "\n")
    print(style.T.to_string(), "\n")
    print(openings.to_string(index=False))
    print("exact caption matches in MusicCaps:", exact_matches)


if __name__ == "__main__":
    main()
