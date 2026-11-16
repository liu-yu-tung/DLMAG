"""Sample-size bias of FD at n = 100: real MusicCaps subsets against real MusicCaps. See docs/analysis_state.txt."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from text_to_music.eval_common import CACHE_DIR, HW3_ROOT, RESULTS_DIR, FrechetReference

OUT_DIR = RESULTS_DIR / "analysis" / "fd_floor"
REFERENCE_CACHE = HW3_ROOT / "outputs" / "ref" / "musiccaps_cache"
SAMPLE_SIZES = (50, 100, 200, 500, 1000, 2000)
VARIANTS = ("subset_vs_full", "disjoint_subsets")
FIT_MIN_SIZES = (50, 200)
PALETTE = {"subset_vs_full": "#0072B2", "disjoint_subsets": "#D55E00"}
FIGURE_SIZE = (8, 4.5)


def load_features() -> dict[str, list[np.ndarray]]:
    """Per feature, one row block per clip (VGGish keeps its 10 frames per clip, as av-benchmark's FD does)."""
    pann = torch.load(REFERENCE_CACHE / "pann_features.pth", weights_only=True)
    passt = torch.load(REFERENCE_CACHE / "passt_features_embed.pth", weights_only=True)
    vgg = torch.load(REFERENCE_CACHE / "vggish_features.pth", weights_only=True)
    names = list(pann)
    assert set(names) == set(passt) == set(vgg)
    mert = np.load(CACHE_DIR / "musiccaps_mert.npy")
    mert_ids = (CACHE_DIR / "musiccaps_mert_ids.txt").read_text().split()
    mert_by_id = dict(zip(mert_ids, mert))
    assert set(names) == set(mert_by_id)
    return {
        "VGGish": [vgg[name].numpy().astype(np.float64) for name in names],
        "PANN": [pann[name]["2048"].numpy().astype(np.float64)[None] for name in names],
        "PaSST": [passt[name].numpy().astype(np.float64)[None] for name in names],
        "MERT": [mert_by_id[name].astype(np.float64)[None] for name in names],
    }


def disjoint_distance(first: np.ndarray, second: np.ndarray) -> float:
    """FD between two samples with fewer rows than dimensions, from singular values of the n x n cross-Gram matrix."""
    (rows_first, dimension), rows_second = first.shape, second.shape[0]
    assert rows_first <= dimension and rows_second <= dimension
    centered_first, centered_second = first - first.mean(axis=0), second - second.mean(axis=0)
    singular_values = np.linalg.svd(centered_first @ centered_second.T, compute_uv=False)
    trace_root = singular_values.sum() / np.sqrt((rows_first - 1) * (rows_second - 1))
    difference = first.mean(axis=0) - second.mean(axis=0)
    return float(
        difference @ difference + (centered_first**2).sum() / (rows_first - 1) + (centered_second**2).sum() / (rows_second - 1) - 2 * trace_root
    )


def pair_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[0] <= first.shape[1] and second.shape[0] <= second.shape[1]:
        return disjoint_distance(first, second)
    return FrechetReference(second).distance(first)


def draw_indices(rng: np.random.Generator, n_clips: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    permutation = rng.permutation(n_clips)
    return permutation[:size], permutation[size : 2 * size]


def fit_against_inverse_size(sizes: np.ndarray, values: np.ndarray) -> dict:
    coefficients, covariance = np.polyfit(1 / sizes, values, 1, cov=True)
    return {"slope": float(coefficients[0]), "intercept": float(coefficients[1]), "intercept_se": float(np.sqrt(covariance[1, 1]))}


def run_feature(name: str, clips: list[np.ndarray], n_draws: int, seed: int, time_budget_s: float) -> list[dict]:
    started = time.time()
    full = np.concatenate(clips)
    reference = FrechetReference(full)
    check = np.random.default_rng(seed)
    first, second = draw_indices(check, len(clips), 100)
    a, b = np.concatenate([clips[i] for i in first]), np.concatenate([clips[i] for i in second])
    if a.shape[0] <= a.shape[1]:
        assert np.isclose(disjoint_distance(a, b), FrechetReference(b).distance(a), rtol=1e-6), "singular-value form disagrees with FrechetReference"
    rows = []
    for size in SAMPLE_SIZES:
        for draw in range(n_draws):
            if time.time() - started > time_budget_s:
                print(f"{name}: time budget reached at n={size}, draw {draw}; remaining draws cut")
                return rows
            rng = np.random.default_rng([seed, size, draw])
            first, second = draw_indices(rng, len(clips), size)
            subset_a, subset_b = np.concatenate([clips[i] for i in first]), np.concatenate([clips[i] for i in second])
            rows.append({"feature": name, "variant": "subset_vs_full", "n": size, "draw": draw, "fd": reference.distance(subset_a)})
            rows.append({"feature": name, "variant": "disjoint_subsets", "n": size, "draw": draw, "fd": pair_distance(subset_a, subset_b)})
        print(f"{name}: n={size} done, {time.time() - started:.0f} s", flush=True)
    return rows


def figure(summary: pd.DataFrame, fits: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    for axis, feature in zip(axes.ravel(), features):
        for variant in VARIANTS:
            part = summary[(summary.feature == feature) & (summary.variant == variant)].sort_values("n")
            axis.errorbar(1 / part["n"], part["mean"], yerr=part["std"], marker="o", markersize=3, linewidth=1, capsize=2, color=PALETTE[variant], label=variant.replace("_", " "))
            fit = fits[(fits.feature == feature) & (fits.variant == variant) & (fits.fit_min_n == FIT_MIN_SIZES[0])].iloc[0]
            grid = np.linspace(0, 1 / part["n"].min(), 50)
            axis.plot(grid, fit["intercept"] + fit["slope"] * grid, linestyle="--", linewidth=0.8, color=PALETTE[variant])
        axis.set_xlim(left=0)
        axis.set_xlabel("1 / n (clips)", fontsize=8)
        axis.set_ylabel(f"FD {feature}", fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fd_vs_inverse_n.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FD of real MusicCaps subsets against real MusicCaps as a function of subset size; writes results/analysis/fd_floor/.")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-budget-minutes", type=float, default=20, help="cap on the whole run; draws beyond it are skipped and printed")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    started = time.time()
    rows = []
    for name, clips in features.items():
        remaining = args.time_budget_minutes * 60 - (time.time() - started)
        rows += run_feature(name, clips, args.draws, args.seed, remaining / (len(features) - list(features).index(name)))
    draws = pd.DataFrame(rows)
    draws.to_csv(OUT_DIR / "fd_draws.csv", index=False)
    summary = draws.groupby(["feature", "variant", "n"])["fd"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(OUT_DIR / "fd_vs_n.csv", index=False)

    fit_rows = []
    for (feature, variant), part in draws.groupby(["feature", "variant"]):
        for minimum in FIT_MIN_SIZES:
            used = part[part.n >= minimum]
            fit = fit_against_inverse_size(used["n"].to_numpy(dtype=float), used["fd"].to_numpy())
            at_100 = summary[(summary.feature == feature) & (summary.variant == variant) & (summary.n == 100)]["mean"]
            fit_rows.append({"feature": feature, "variant": variant, "fit_min_n": minimum, **fit,
                             "mean_fd_at_n100": float(at_100.iloc[0]) if len(at_100) else None,
                             "bias_at_n100_from_slope": fit["slope"] / 100})
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(OUT_DIR / "fd_fit.csv", index=False)
    figure(summary, fits, list(features))

    dev_real = json.loads((RESULTS_DIR / "dev__real" / "metrics.json").read_text()) if (RESULTS_DIR / "dev__real" / "metrics.json").exists() else {}
    cross_check = {key: dev_real[key]["value"] for key in ("FD-VGG_musiccaps", "FD-PANN_musiccaps", "FD-PASST_musiccaps", "fd_mert_musiccaps") if key in dev_real}
    (OUT_DIR / "dev_real_cross_check.json").write_text(json.dumps(cross_check, indent=1) + "\n")

    pd.set_option("display.width", 200)
    print(summary.pivot_table(index=["feature", "variant"], columns="n", values="mean").round(3).to_string())
    print(summary.pivot_table(index=["feature", "variant"], columns="n", values="std").round(3).to_string())
    print(fits.round(4).to_string(index=False))
    print("dev__real (100 real dev clips) against the full MusicCaps reference:", cross_check)
    print(f"total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
