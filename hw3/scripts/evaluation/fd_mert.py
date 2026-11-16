"""FD-MERT of a prediction folder vs the retrieved clips and vs all MusicCaps. Run with hw1's venv; see docs/eval_notes.md."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile
import torch
import torchaudio
from tqdm import tqdm

from text_to_music.eval_common import (
    CACHE_DIR,
    HW3_ROOT,
    MUSICCAPS_DIR,
    RESULTS_DIR,
    FrechetReference,
    bootstrap_frechet,
    bootstrap_indices,
    load_retrieval,
)

HW1_SOURCE = Path(__file__).resolve().parents[3] / "hw1" / "src"
sys.path.insert(0, str(HW1_SOURCE))
from hw1.features.mert import DEFAULT_MODEL, SAMPLE_RATE, load_mert, pooled_layers  # noqa: E402

CLIP_SAMPLES = 10 * SAMPLE_RATE
MUSICCAPS_CACHE = CACHE_DIR / "musiccaps_mert.npy"
MUSICCAPS_IDS = CACHE_DIR / "musiccaps_mert_ids.txt"


def read_clip(path: Path) -> np.ndarray:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = torch.from_numpy(audio.mean(axis=1))
    if sample_rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, SAMPLE_RATE)
    return mono[:CLIP_SAMPLES].numpy()


def embed(model: torch.nn.Module, paths: list[Path], batch_size: int) -> np.ndarray:
    """Time-mean of every hidden state (embedding output + each transformer layer), averaged over layers: (N, D)."""
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="mert"):
        clips = [read_clip(path) for path in paths[start : start + batch_size]]
        lengths = [len(clip) for clip in clips]
        batch = np.zeros((len(clips), max(lengths)), dtype=np.float32)
        for row, clip in enumerate(clips):
            batch[row, : len(clip)] = clip
        pooled = pooled_layers(model, batch, None if len(set(lengths)) == 1 else lengths)
        dimension = pooled.shape[2] // 2
        chunks.append(pooled[:, :, :dimension].mean(axis=1))
    return np.concatenate(chunks)


def musiccaps_embeddings(model, batch_size: int) -> tuple[np.ndarray, list[str]]:
    if MUSICCAPS_CACHE.exists() and MUSICCAPS_IDS.exists():
        return np.load(MUSICCAPS_CACHE), MUSICCAPS_IDS.read_text().split()
    paths = sorted(MUSICCAPS_DIR.glob("*.mp3"))
    assert len(paths) == 5367
    features = embed(model, paths, batch_size)
    ids = [path.stem for path in paths]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MUSICCAPS_CACHE, features)
    MUSICCAPS_IDS.write_text("\n".join(ids) + "\n")
    return features, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-MERT of <pred>/0.wav..99.wav; appends to results/<name>/metrics.json.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--name", default=None, help="run name (default: name of the pred folder)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_name = args.name or args.pred.resolve().name
    metrics_path = RESULTS_DIR / run_name / "metrics.json"
    assert metrics_path.exists(), f"{metrics_path} not found: run scripts/evaluation/evaluate.py first"

    model = load_mert(DEFAULT_MODEL, args.dtype, args.device)
    musiccaps, musiccaps_ids = musiccaps_embeddings(model, args.batch_size)
    row_of = {youtube_id: row for row, youtube_id in enumerate(musiccaps_ids)}
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("set", "test") == "dev":
        with open(HW3_ROOT / "data" / "dev_meta.csv", newline="") as handle:
            clips = {row["id"]: row["ytid"] for row in csv.DictReader(handle)}
    else:
        clips = load_retrieval()
    retrieved = musiccaps[[row_of[clips[str(i)]] for i in range(100)]]

    paths = [args.pred / f"{i}.wav" for i in range(100)]
    assert all(path.exists() for path in paths)
    predicted = embed(model, paths, args.batch_size)
    groups = [row[None] for row in predicted]
    indices = bootstrap_indices(len(groups))

    for key, reference in (("fd_mert_retrieved", retrieved), ("fd_mert_musiccaps", musiccaps)):
        point, interval, shifted = bootstrap_frechet(FrechetReference(reference), groups, indices)
        metrics[key] = {"value": point, "ci95": interval, "bootstrap_mean": shifted}
        print(f"{key:<20} {point:.4f}  {interval}")
    metrics["fd_mert_embedding"] = f"{DEFAULT_MODEL}, time-mean of hidden states, mean over all {model.config.num_hidden_layers + 1} hidden states, first 10 s at 24 kHz"
    metrics_path.write_text(json.dumps(metrics, indent=1) + "\n")


if __name__ == "__main__":
    main()
