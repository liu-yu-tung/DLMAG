"""Shared paths and helpers for the evaluation scripts (retrieval, evaluate, fd_mert); numpy only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

HW3_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = HW3_ROOT / "data"
MUSICCAPS_DIR = DATA_DIR / "musiccaps" / "audio_mp3"
TEST_CSV = DATA_DIR / "test.csv"
CACHE_DIR = HW3_ROOT / "outputs" / "cache"
RESULTS_DIR = HW3_ROOT / "results"
AV_BENCH_DIR = HW3_ROOT / "external" / "av-benchmark"
AV_BENCH_PYTHON = AV_BENCH_DIR / ".venv" / "bin" / "python"
CLAP_CHECKPOINT = AV_BENCH_DIR / "weights" / "music_audioset_epoch_15_esc_90.14.pt"
RETRIEVED_DIR = HW3_ROOT / "outputs" / "ref" / "test_retrieved"
RETRIEVAL_JSON = RESULTS_DIR / "retrieval.json"

SEED = 0
N_BOOTSTRAP = 1000


def code_version() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=HW3_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=HW3_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return head + ("+dirty" if dirty else "")


def load_retrieval() -> dict[str, str]:
    mapping = json.loads(RETRIEVAL_JSON.read_text())
    assert sorted(mapping, key=int) == [str(i) for i in range(100)], "retrieval.json must have keys 0..99"
    return mapping


def bootstrap_indices(n_items: int, n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_resamples, n_items))


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


class FrechetReference:
    """Fixed reference set; distance() matches av_bench.metrics.compute_fd (sqrtm) but uses eigenvalues,
    which is exact for PSD covariances and fast enough for 1000 bootstrap resamples at 2048 dims."""

    def __init__(self, features: np.ndarray):
        self.mean = features.mean(axis=0)
        self.covariance = np.cov(features, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        self.covariance_root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))) @ eigenvectors.T

    def distance(self, features: np.ndarray) -> float:
        n_samples, dimension = features.shape
        assert n_samples > 1
        mean = features.mean(axis=0)
        centered = features - mean
        if n_samples <= dimension:
            # nonzero eigenvalues of covariance @ reference equal those of this n x n matrix
            small = centered @ self.covariance @ centered.T / (n_samples - 1)
            eigenvalues = np.linalg.eigvalsh((small + small.T) / 2)
            covariance = None
        else:
            covariance = centered.T @ centered / (n_samples - 1)
            product = self.covariance_root @ covariance @ self.covariance_root
            eigenvalues = np.linalg.eigvalsh((product + product.T) / 2)
        trace_root = np.sqrt(np.clip(eigenvalues, 0, None)).sum()
        trace_covariance = np.trace(covariance) if covariance is not None else float((centered**2).sum() / (n_samples - 1))
        diff = mean - self.mean
        return float(diff @ diff + trace_covariance + np.trace(self.covariance) - 2 * trace_root)


def bootstrap_frechet(
    reference: FrechetReference, groups: list[np.ndarray], indices: np.ndarray
) -> tuple[float, list[float], float]:
    """groups[i] holds the feature rows of prompt i (one row, or several frames); resample prompts with replacement.
    FD is biased upward at small n and duplicates in a resample lower the effective n, so the point value can sit
    below the percentile interval; the bootstrap mean is returned to show that shift."""
    point = reference.distance(np.concatenate(groups))
    values = np.array([reference.distance(np.concatenate([groups[j] for j in row])) for row in indices])
    return point, percentile_interval(values), float(values.mean())
