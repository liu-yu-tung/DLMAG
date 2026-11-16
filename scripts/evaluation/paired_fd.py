"""Paired FD-difference bootstrap (run minus baseline, same resampled prompt ids, fixed MusicCaps reference). Run with the av-benchmark venv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from text_to_music.eval_common import HW3_ROOT, RESULTS_DIR, FrechetReference, bootstrap_indices, percentile_interval

PRED_CACHE_DIR = HW3_ROOT / "outputs" / "pred_cache"
MUSICCAPS_CACHE = HW3_ROOT / "outputs" / "ref" / "musiccaps_cache"
FEATURE_FILES = {"VGG": "vggish_features.pth", "PANN": "pann_features.pth", "PASST": "passt_features_embed.pth"}
N_PROMPTS = 100


def load_features(cache_dir: Path) -> dict[str, dict]:
    return {kind: torch.load(cache_dir / file, weights_only=True, map_location="cpu") for kind, file in FEATURE_FILES.items()}


def feature_groups(cache: dict, kind: str, names: list[str]) -> list[np.ndarray]:
    # same grouping as scripts/evaluation/evaluate.py: VGGish keeps several frames per clip
    if kind == "PANN":
        return [cache["PANN"][name]["2048"].numpy()[None] for name in names]
    if kind == "VGG":
        return [cache["VGG"][name].numpy() for name in names]
    return [cache["PASST"][name].numpy()[None] for name in names]


def resampled_distances(reference: FrechetReference, groups: list[np.ndarray], indices: np.ndarray) -> tuple[float, np.ndarray]:
    point = reference.distance(np.concatenate(groups))
    values = np.array([reference.distance(np.concatenate([groups[j] for j in row])) for row in indices])
    return point, values


def runs_with_cache(set_name: str) -> list[str]:
    return sorted(p.name.split("__", 1)[1] for p in PRED_CACHE_DIR.glob(f"{set_name}__*") if (p / FEATURE_FILES["PANN"]).exists())


def stored_point(set_name: str, run: str, kind: str) -> float | None:
    path = RESULTS_DIR / f"{set_name}__{run}" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(f"FD-{kind}_musiccaps", {}).get("value")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap of FD(run) - FD(baseline) against all MusicCaps, same resampled prompt ids for both.")
    parser.add_argument("--set", dest="set_name", choices=("test", "dev"), default="dev")
    parser.add_argument("--baseline", required=True, help="run name without the set prefix, e.g. cfg7")
    parser.add_argument("--runs", default=None, help="comma-separated run names; default: every run of the set with a pred_cache")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = args.runs.split(",") if args.runs else runs_with_cache(args.set_name)
    names = [str(i) for i in range(N_PROMPTS)]
    indices = bootstrap_indices(N_PROMPTS)
    musiccaps = load_features(MUSICCAPS_CACHE)

    def run_groups(run: str) -> dict[str, list[np.ndarray]]:
        cache = load_features(PRED_CACHE_DIR / f"{args.set_name}__{run}")
        return {kind: feature_groups(cache, kind, names) for kind in FEATURE_FILES}

    references = {kind: FrechetReference(np.concatenate(feature_groups(musiccaps, kind, list(musiccaps[kind])))) for kind in FEATURE_FILES}
    baseline_groups = run_groups(args.baseline)
    baseline = {kind: resampled_distances(references[kind], baseline_groups[kind], indices) for kind in FEATURE_FILES}

    result: dict = {
        "set": args.set_name,
        "baseline": args.baseline,
        "reference": "all MusicCaps",
        "bootstrap": {"resamples": len(indices), "seed": 0, "unit": "prompt id", "paired": True, "interval": "percentile 95%"},
        "fd_mert": "skipped: no per-clip MERT embeddings are cached for the runs (fd_mert.py only caches the MusicCaps embeddings)",
        "runs": {},
    }
    for run in runs:
        groups = baseline_groups if run == args.baseline else run_groups(run)
        entry = {}
        for kind in FEATURE_FILES:
            point, values = baseline[kind] if run == args.baseline else resampled_distances(references[kind], groups[kind], indices)
            base_point, base_values = baseline[kind]
            assert values.shape == base_values.shape == (len(indices),)
            differences = values - base_values
            stored = stored_point(args.set_name, run, kind)
            entry[kind] = {
                "fd": point,
                "fd_baseline": base_point,
                "difference": point - base_point,
                "difference_ci95": percentile_interval(differences),
                "difference_bootstrap_mean": float(differences.mean()),
                "fd_metrics_json": stored,
            }
            if stored is not None:
                assert np.isclose(point, stored, rtol=1e-3), (run, kind, point, stored)
        result["runs"][run] = entry

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"Paired FD difference vs {args.set_name}__{args.baseline} (all MusicCaps; 1000 paired resamples; 95% percentile interval); FD-MERT skipped (no per-run MERT cache)\n")
    print("| run | " + " | ".join(f"{kind} FD | {kind} diff [95%]" for kind in FEATURE_FILES) + " |")
    print("|---|" + "---|---|" * len(FEATURE_FILES))
    for run, entry in result["runs"].items():
        cells = [
            f"{entry[kind]['fd']:.3f} | {entry[kind]['difference']:+.3f} [{entry[kind]['difference_ci95'][0]:+.3f}, {entry[kind]['difference_ci95'][1]:+.3f}]"
            for kind in FEATURE_FILES
        ]
        print(f"| {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
