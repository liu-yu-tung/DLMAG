"""Six extra per-run metrics (CPU, incremental). Run from hw3/ with ./.venv/bin/python scripts/evaluation/extra_metrics.py [--runs dev__cfg7,...] [--force]."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from text_to_music.eval_common import DATA_DIR, HW3_ROOT, RESULTS_DIR, FrechetReference, bootstrap_indices, percentile_interval

METRICS_VERSION = 2  # bump when a metric changes so cached runs are recomputed
OUT_DIR = RESULTS_DIR / "analysis"
CACHE_DIR = OUT_DIR / "extra_metrics_cache"
CSV_PATH = OUT_DIR / "extra_metrics.csv"
PRED_CACHE_DIR = HW3_ROOT / "outputs" / "pred_cache"
REF_DIR = HW3_ROOT / "outputs" / "ref"
MUSICCAPS_CACHE = REF_DIR / "musiccaps_cache"
SET_REFERENCE = {"dev": REF_DIR / "dev_real_cache", "test": REF_DIR / "test_retrieved_cache"}
SET_CAPTIONS = {"dev": DATA_DIR / "dev.csv", "test": DATA_DIR / "test.csv"}
DEV_META = DATA_DIR / "dev_meta.csv"
RADII_PATH = CACHE_DIR / "_musiccaps_radii.npz"

N_PROMPTS = 100
NEIGHBOURS = 5
PRED_FILES = ("clap_laion_audio.pth", "vggish_features.pth", "pann_features.pth", "passt_features_embed.pth", "passt_logits.pth")
FEATURE_KINDS = ("pann", "passt", "vggish")
# AudioSet indices (527-class order shared by PaSST and PANN), from the LAION-CLAP repo's audioset_class_labels_indices.json in the uv cache
VOCAL_CLASSES = {"Speech": 0, "Singing": 27, "Choir": 28, "Rapping": 36, "Vocal music": 254}
VOCAL_KEYWORDS = ("vocal", "sing", "singer", "voice", "choir", "rap", "lyrics", "male", "female", "chant")
# whole words with their usual endings, so "rap" does not hit "rapid" and "sing" does not hit "single"
VOCAL_PATTERN = re.compile(
    r"\b(vocals?|vocalists?|sing|sings|singer|singers|singing|voices?|choir|choirs|rap|raps|rapper|rappers|rapping|lyrics?|males?|females?|chant|chants|chanting)\b",
    re.IGNORECASE,
)
STFT_SIZE, STFT_HOP = 2048, 512
BASELINE_RUN = "cfg7"
SEED_LEVEL_RUNS = ("cfg7_seed1", "cfg7_seed2")


def split_name(name: str) -> tuple[str, str]:
    set_name, run = name.split("__", 1)
    assert set_name in SET_REFERENCE, name
    return set_name, run


def audio_dir(set_name: str, run: str) -> Path:
    return REF_DIR / "dev_real" if (set_name, run) == ("dev", "real") else HW3_ROOT / "outputs" / set_name / run


def load_pth(path: Path):
    return torch.load(path, weights_only=True, map_location="cpu")


def names() -> list[str]:
    return [str(i) for i in range(N_PROMPTS)]


def feature_matrix(cache: dict, kind: str, keys: list[str]) -> np.ndarray:
    if kind == "pann":
        rows = [cache[key]["2048"].numpy() for key in keys]
    elif kind == "vggish":
        rows = [cache[key].numpy().mean(axis=0) for key in keys]  # mean over a clip's frames
    else:
        rows = [cache[key].numpy() for key in keys]
    return np.stack(rows).astype(np.float64)


def feature_file(kind: str) -> str:
    return {"pann": "pann_features.pth", "passt": "passt_features_embed.pth", "vggish": "vggish_features.pth"}[kind]


def pairwise_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    squared = (first**2).sum(1)[:, None] + (second**2).sum(1)[None, :] - 2 * first @ second.T
    return np.sqrt(np.clip(squared, 0, None))


def neighbour_radii(points: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
    distances = pairwise_distance(points, points)
    return np.partition(distances, k, axis=1)[:, k]  # index 0 is the point itself


def density_coverage(real: np.ndarray, real_radii: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    inside = pairwise_distance(fake, real) <= real_radii[None, :]
    assert inside.shape == (len(fake), len(real))
    return float(inside.sum() / (NEIGHBOURS * len(fake))), float(inside.any(axis=0).mean())


def precision_recall(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    distances = pairwise_distance(fake, real)
    precision = (distances <= neighbour_radii(real)[None, :]).any(axis=1).mean()
    recall = (distances <= neighbour_radii(fake)[:, None]).any(axis=0).mean()
    return float(precision), float(recall)


def mtime(path: Path) -> int:
    return path.stat().st_mtime_ns


class References:
    """MusicCaps features and k-NN radii (radii cached on disk), plus each set's paired real clips; loaded on first use."""

    def __init__(self) -> None:
        self.musiccaps: dict[str, tuple[list[str], np.ndarray]] = {}
        self.radii: dict[str, np.ndarray] = {}
        self.paired: dict[tuple[str, str], np.ndarray] = {}
        self.text: dict[str, np.ndarray] = {}

    def load_musiccaps(self) -> None:
        if self.musiccaps:
            return
        for kind in FEATURE_KINDS:
            cache = load_pth(MUSICCAPS_CACHE / feature_file(kind))
            keys = sorted(cache)
            self.musiccaps[kind] = (keys, feature_matrix(cache, kind, keys))
        stamp = np.array([mtime(MUSICCAPS_CACHE / feature_file(kind)) for kind in FEATURE_KINDS] + [NEIGHBOURS])
        if RADII_PATH.exists():
            stored = np.load(RADII_PATH)
            if np.array_equal(stored["stamp"], stamp):
                self.radii = {kind: stored[kind] for kind in FEATURE_KINDS}
                return
        self.radii = {kind: neighbour_radii(self.musiccaps[kind][1]) for kind in FEATURE_KINDS}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(RADII_PATH, stamp=stamp, **self.radii)

    def musiccaps_for(self, kind: str, exclude_ytids: set[str]) -> tuple[np.ndarray, np.ndarray]:
        keys, matrix = self.musiccaps[kind]
        if not exclude_ytids:
            return matrix, self.radii[kind]
        keep = np.array([key not in exclude_ytids for key in keys])
        assert keep.sum() == len(keys) - len(exclude_ytids)
        return matrix[keep], neighbour_radii(matrix[keep])

    def paired_features(self, set_name: str, kind: str) -> np.ndarray:
        if (set_name, kind) not in self.paired:
            self.paired[(set_name, kind)] = feature_matrix(load_pth(SET_REFERENCE[set_name] / feature_file(kind)), kind, names())
        return self.paired[(set_name, kind)]

    def text_embeddings(self, set_name: str) -> np.ndarray:
        if set_name not in self.text:
            cache = load_pth(SET_REFERENCE[set_name] / "clap_laion_text.pth")
            self.text[set_name] = np.stack([cache[key].numpy() for key in names()]).astype(np.float64)
        return self.text[set_name]


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def clap_embeddings(set_name: str, run: str) -> np.ndarray:
    cache = load_pth(PRED_CACHE_DIR / f"{set_name}__{run}" / "clap_laion_audio.pth")
    matrix = np.stack([cache[key].numpy() for key in names()]).astype(np.float64)
    assert matrix.shape == (N_PROMPTS, 512)
    return matrix


def retrieval_metrics(audio: np.ndarray, text: np.ndarray) -> dict:
    similarity = unit_rows(audio) @ unit_rows(text).T
    assert similarity.shape == (N_PROMPTS, N_PROMPTS)
    own = np.diag(similarity)
    ranks = 1 + (similarity > own[:, None]).sum(axis=1)
    return {
        "retrieval_r1": float((ranks <= 1).mean()),
        "retrieval_r5": float((ranks <= 5).mean()),
        "retrieval_r10": float((ranks <= 10).mean()),
        "retrieval_median_rank": float(np.median(ranks)),
        "retrieval_mrr": float((1 / ranks).mean()),
    }


def distribution_metrics(references: References, set_name: str, run: str) -> dict:
    # density/coverage use 5367 reference vs 100 generated clips (coverage is capped low by the 100); precision/recall use 100 vs 100
    references.load_musiccaps()
    pred_cache = PRED_CACHE_DIR / f"{set_name}__{run}"
    exclude: set[str] = set()
    if (set_name, run) == ("dev", "real"):
        meta = pd.read_csv(DEV_META, dtype=str)
        exclude = set(meta["ytid"])  # the real clips are inside MusicCaps; leave them out so a clip is not its own neighbour
    out = {}
    for kind in FEATURE_KINDS:
        fake = feature_matrix(load_pth(pred_cache / feature_file(kind)), kind, names())
        real_all, radii = references.musiccaps_for(kind, exclude)
        density, coverage = density_coverage(real_all, radii, fake)
        precision, recall = precision_recall(references.paired_features(set_name, kind), fake)
        out.update({f"density_{kind}": density, f"coverage_{kind}": coverage, f"precision_{kind}": precision, f"recall_{kind}": recall})
    return out


def vocal_metrics(set_name: str, run: str) -> dict:
    logits = load_pth(PRED_CACHE_DIR / f"{set_name}__{run}" / "passt_logits.pth")
    matrix = np.stack([logits[key].numpy() for key in names()]).astype(np.float64)
    assert matrix.shape == (N_PROMPTS, 527)
    presence = (1 / (1 + np.exp(-matrix[:, list(VOCAL_CLASSES.values())]))).max(axis=1)
    captions = pd.read_csv(SET_CAPTIONS[set_name])
    assert list(captions["id"]) == list(range(N_PROMPTS)), "captions must be ids 0..99 in order"
    asks_vocals = np.array([bool(VOCAL_PATTERN.search(text)) for text in captions["caption"]])
    assert asks_vocals.any() and not asks_vocals.all()
    return {
        "vocal_presence_vocal_captions": float(presence[asks_vocals].mean()),
        "vocal_presence_other_captions": float(presence[~asks_vocals].mean()),
        "vocal_presence_all": float(presence.mean()),
        "n_vocal_captions": int(asks_vocals.sum()),
    }


def clip_loudness_brightness(path: Path) -> tuple[float, float, float]:
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    rms = float(np.sqrt((samples**2).mean()))  # RMS and crest over both channels; the centroid uses the mono mix
    assert rms > 0, path
    mono = samples.mean(axis=1)
    frames = np.lib.stride_tricks.sliding_window_view(mono, STFT_SIZE)[::STFT_HOP] * np.hanning(STFT_SIZE)
    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    energy = magnitude.sum(axis=1)
    voiced = energy > 1e-8
    frequencies = np.fft.rfftfreq(STFT_SIZE, 1 / rate)
    centroid = (magnitude[voiced] @ frequencies) / energy[voiced]
    return 20 * np.log10(rms), 20 * np.log10(np.abs(samples).max() / rms), float(centroid.mean())


def audio_metrics(set_name: str, run: str) -> dict:
    rows = np.array([clip_loudness_brightness(audio_dir(set_name, run) / f"{key}.wav") for key in names()])
    assert rows.shape == (N_PROMPTS, 3)
    return {"rms_dbfs_mean": float(rows[:, 0].mean()), "crest_db_mean": float(rows[:, 1].mean()), "centroid_hz_mean": float(rows[:, 2].mean())}


def clap_fd_metrics(set_name: str, run: str) -> dict:
    # no CLAP audio embeddings exist for the MusicCaps cache or the test retrieved clips, so FD-CLAP uses the dev paired real clips only
    if set_name != "dev":
        return {"fd_clap_paired": float("nan")}
    reference = FrechetReference(clap_embeddings("dev", "real"))
    return {"fd_clap_paired": reference.distance(clap_embeddings(set_name, run))}


def paired_change_metrics(set_name: str, run: str) -> dict:
    baseline = unit_rows(clap_embeddings(set_name, BASELINE_RUN))
    own = unit_rows(clap_embeddings(set_name, run))
    cosine = (baseline * own).sum(axis=1)
    return {"paired_change_mean": float(cosine.mean()), "paired_change_median": float(np.median(cosine))}


def win_rate_metrics(set_name: str, run: str) -> dict:
    # strict wins only; ties (e.g. the baseline against itself) count as not better
    base = pd.read_csv(RESULTS_DIR / f"{set_name}__{BASELINE_RUN}" / "per_prompt.csv").set_index("id").sort_index()
    own = pd.read_csv(RESULTS_DIR / f"{set_name}__{run}" / "per_prompt.csv").set_index("id").sort_index()
    assert list(base.index) == list(own.index) == list(range(N_PROMPTS))
    indices = bootstrap_indices(N_PROMPTS)
    out = {}
    for label, wins in (
        ("clap", (own["clap_cosine"] > base["clap_cosine"]).to_numpy(float)),
        ("kl_passt", (own["kl_passt_softmax"] < base["kl_passt_softmax"]).to_numpy(float)),
    ):
        low, high = percentile_interval(wins[indices].mean(axis=1))
        out.update({f"win_{label}": float(wins.mean()), f"win_{label}_ci_low": low, f"win_{label}_ci_high": high})
    return out


def input_stamps(set_name: str, run: str) -> dict[str, int] | str:
    """File modification times the result depends on, or a reason string when the run cannot be scored yet."""
    pred_cache = PRED_CACHE_DIR / f"{set_name}__{run}"
    paths = {f"pred/{file}": pred_cache / file for file in PRED_FILES}
    paths["per_prompt"] = RESULTS_DIR / f"{set_name}__{run}" / "per_prompt.csv"
    paths["metrics"] = RESULTS_DIR / f"{set_name}__{run}" / "metrics.json"
    paths["baseline_clap"] = PRED_CACHE_DIR / f"{set_name}__{BASELINE_RUN}" / "clap_laion_audio.pth"
    paths["baseline_per_prompt"] = RESULTS_DIR / f"{set_name}__{BASELINE_RUN}" / "per_prompt.csv"
    if set_name == "dev":
        paths["dev_real_clap"] = PRED_CACHE_DIR / "dev__real" / "clap_laion_audio.pth"
    paths["captions"] = SET_CAPTIONS[set_name]
    paths["text"] = SET_REFERENCE[set_name] / "clap_laion_text.pth"
    for kind in FEATURE_KINDS:
        paths[f"paired_{kind}"] = SET_REFERENCE[set_name] / feature_file(kind)
        paths[f"musiccaps_{kind}"] = MUSICCAPS_CACHE / feature_file(kind)
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        return f"eval not finished or inputs missing ({', '.join(missing[:3])})"
    wavs = [audio_dir(set_name, run) / f"{key}.wav" for key in names()]
    if not all(wav.exists() for wav in wavs):
        return "audio folder incomplete"
    stamps = {label: mtime(path) for label, path in paths.items()}
    stamps["wav_latest"] = max(mtime(wav) for wav in wavs)
    return stamps


def compute_run(references: References, set_name: str, run: str) -> dict:
    metrics: dict = {}
    metrics.update(retrieval_metrics(clap_embeddings(set_name, run), references.text_embeddings(set_name)))
    metrics.update(distribution_metrics(references, set_name, run))
    metrics.update(paired_change_metrics(set_name, run))
    metrics.update(clap_fd_metrics(set_name, run))
    metrics.update(vocal_metrics(set_name, run))
    metrics.update(audio_metrics(set_name, run))
    metrics.update(win_rate_metrics(set_name, run))
    return metrics


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    temporary.replace(path)


def discover_runs() -> list[str]:
    found = []
    for path in sorted(PRED_CACHE_DIR.iterdir()):
        if "__" in path.name and path.name.split("__", 1)[0] in SET_REFERENCE:
            found.append(path.name)
    return found


def build_csv() -> pd.DataFrame:
    rows = []
    for path in sorted(CACHE_DIR.glob("*__*.json")):
        entry = json.loads(path.read_text())
        rows.append({"set": entry["set"], "run": entry["run"], **entry["metrics"], "vocal_keywords": ";".join(VOCAL_KEYWORDS)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    levels = table[table["run"].isin(SEED_LEVEL_RUNS)].groupby("set")["paired_change_mean"].mean()
    table["paired_change_other_seed_mean"] = table["set"].map(levels)  # the "different seed" scale, mean of cfg7_seed1 and cfg7_seed2
    first = ["set", "run"]
    table = table[first + [column for column in table.columns if column not in first]]
    table.round(6).to_csv(CSV_PATH, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Add six extra metrics per run (retrieval rank, density/coverage, paired change, vocal presence, loudness, win rate).")
    parser.add_argument("--runs", default=None, help="comma-separated names such as dev__cfg7; default: every run with a pred_cache")
    parser.add_argument("--force", action="store_true", help="recompute even when the cache is current")
    args = parser.parse_args()

    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    requested = args.runs.split(",") if args.runs else discover_runs()
    references = References()
    computed = 0
    for name in requested:
        set_name, run = split_name(name)
        stamps = input_stamps(set_name, run)
        if isinstance(stamps, str):
            print(f"skip {name}: {stamps}")
            continue
        cache_path = CACHE_DIR / f"{name}.json"
        if not args.force and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get("version") == METRICS_VERSION and cached.get("inputs") == stamps:
                continue
        begin = time.time()
        metrics = compute_run(references, set_name, run)
        write_json_atomic(cache_path, {"version": METRICS_VERSION, "set": set_name, "run": run, "inputs": stamps, "metrics": metrics})
        computed += 1
        print(f"computed {name} in {time.time() - begin:.1f} s")
    table = build_csv()
    print(f"{computed} computed, {len(requested) - computed} cached or skipped, {len(table)} rows in {CSV_PATH.relative_to(HW3_ROOT)}, {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
