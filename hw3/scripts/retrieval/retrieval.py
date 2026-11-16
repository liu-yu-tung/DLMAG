"""CLAP text-to-audio retrieval over MusicCaps. Run with the av-benchmark venv; see docs/eval_notes.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import laion_clap
import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from tqdm import tqdm

from text_to_music.eval_common import (
    CACHE_DIR,
    CLAP_CHECKPOINT,
    MUSICCAPS_DIR,
    RESULTS_DIR,
    RETRIEVAL_JSON,
    TEST_CSV,
)

SAMPLE_RATE = 48000
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
AUDIO_CACHE = CACHE_DIR / "musiccaps_clap90.npy"
AUDIO_IDS = CACHE_DIR / "musiccaps_clap90_ids.txt"
TEXT_CACHE = CACHE_DIR / "test_text_clap90.npy"
TEXT_IDS = CACHE_DIR / "test_text_clap90_ids.txt"


def load_model(device: str) -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    model.load_ckpt(str(CLAP_CHECKPOINT), verbose=False)
    return model.eval()


def read_clip(path) -> np.ndarray:
    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1))
    waveform = waveform - waveform.mean()
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = waveform[:CLIP_SAMPLES]
    return torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[0])).numpy()


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).astype(np.float32)


def embed_musiccaps(
    model: laion_clap.CLAP_Module, batch_size: int, limit: int | None = None
) -> tuple[np.ndarray, list[str]]:
    paths = sorted(MUSICCAPS_DIR.glob("*.mp3"))
    assert len(paths) == 5367, len(paths)
    paths = paths[:limit]
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="audio"):
        batch = np.stack([read_clip(path) for path in paths[start : start + batch_size]])
        with torch.inference_mode():
            chunks.append(model.get_audio_embedding_from_data(x=batch, use_tensor=False))
    return normalize(np.concatenate(chunks)), [path.stem for path in paths]


def embed_text(model: laion_clap.CLAP_Module, captions: list[str]) -> np.ndarray:
    with torch.inference_mode():
        return normalize(model.get_text_embedding(captions, use_tensor=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-1 CLAP retrieval of MusicCaps clips for the test captions.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu (cuda needs about 2 GB)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recompute", action="store_true", help="ignore cached embeddings")
    parser.add_argument("--limit", type=int, default=None, help="debug: only the first N clips (no outputs written)")
    args = parser.parse_args()

    test = pd.read_csv(TEST_CSV)
    assert test["id"].tolist() == list(range(100))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(args.device)
    started = time.time()

    if args.limit:
        embeddings, ids = embed_musiccaps(model, args.batch_size, args.limit)
        print(embeddings.shape, ids[:3], f"{time.time() - started:.1f}s")
        return

    if AUDIO_CACHE.exists() and AUDIO_IDS.exists() and not args.recompute:
        audio = np.load(AUDIO_CACHE)
        ids = AUDIO_IDS.read_text().split()
    else:
        audio, ids = embed_musiccaps(model, args.batch_size)
        np.save(AUDIO_CACHE, audio)
        AUDIO_IDS.write_text("\n".join(ids) + "\n")
    assert audio.shape == (len(ids), 512) and len(ids) == 5367

    text = embed_text(model, test["caption"].tolist())
    np.save(TEXT_CACHE, text)
    TEXT_IDS.write_text("\n".join(str(i) for i in test["id"]) + "\n")
    print(f"embedding time {time.time() - started:.0f}s")

    scores = text @ audio.T
    top5 = np.argsort(-scores, axis=1)[:, :5]

    retrieval = {str(i): ids[top5[i, 0]] for i in range(100)}
    RETRIEVAL_JSON.write_text(json.dumps(retrieval, indent=1) + "\n")
    rows = [(i, rank + 1, ids[j], float(scores[i, j])) for i in range(100) for rank, j in enumerate(top5[i])]
    pd.DataFrame(rows, columns=["id", "rank", "youtube_id", "score"]).to_csv(
        RESULTS_DIR / "retrieval_top5.csv", index=False
    )
    ceiling = pd.DataFrame(
        {"id": range(100), "youtube_id": [retrieval[str(i)] for i in range(100)], "clap_cosine": scores[np.arange(100), top5[:, 0]]}
    )
    ceiling.to_csv(RESULTS_DIR / "retrieval_ceiling.csv", index=False)
    print(f"top-1 score min {ceiling.clap_cosine.min():.4f} mean {ceiling.clap_cosine.mean():.4f} max {ceiling.clap_cosine.max():.4f}")
    print(f"distinct clips among top-1: {len(set(retrieval.values()))}")


if __name__ == "__main__":
    main()
