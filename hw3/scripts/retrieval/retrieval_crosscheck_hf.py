"""Cross-check retrieval with HW1's HF model laion/larger_clap_music. Run with hw1's venv; see docs/eval_notes.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from scipy import stats
from tqdm import tqdm

from text_to_music.eval_common import CACHE_DIR, MUSICCAPS_DIR, RESULTS_DIR, TEST_CSV, load_retrieval

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "hw1" / "src"))
from hw1.features.clap import TARGET_SR, embed_text, embed_waveform_batch, load_model  # noqa: E402

CLIP_SAMPLES = 10 * TARGET_SR
HF_CACHE = CACHE_DIR / "musiccaps_clap_hf.npy"
HF_IDS = CACHE_DIR / "musiccaps_clap_hf_ids.txt"
PINNED_CACHE = CACHE_DIR / "musiccaps_clap90.npy"
PINNED_IDS = CACHE_DIR / "musiccaps_clap90_ids.txt"
PINNED_TEXT = CACHE_DIR / "test_text_clap90.npy"


def read_clip(path: Path) -> np.ndarray:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = torch.from_numpy(audio.mean(axis=1))
    mono = torchaudio.functional.resample(mono, sample_rate, TARGET_SR)[:CLIP_SAMPLES]
    return torch.nn.functional.pad(mono, (0, CLIP_SAMPLES - mono.shape[0])).numpy()


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-1 agreement between laion/larger_clap_music (HF) and the 90.14 .pt retrieval.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model, processor = load_model(args.device)
    captions = pd.read_csv(TEST_CSV)["caption"].tolist()
    text = normalize(embed_text(model, processor, captions, args.device))

    paths = sorted(MUSICCAPS_DIR.glob("*.mp3"))
    ids = [path.stem for path in paths]
    if HF_CACHE.exists() and HF_IDS.exists():
        audio = np.load(HF_CACHE)
        assert HF_IDS.read_text().split() == ids
    else:
        chunks = []
        for start in tqdm(range(0, len(paths), args.batch_size), desc="hf audio"):
            batch = [read_clip(path) for path in paths[start : start + args.batch_size]]
            chunks.append(embed_waveform_batch(model, processor, batch, args.device))
        audio = normalize(np.concatenate(chunks)).astype(np.float32)
        np.save(HF_CACHE, audio)
        HF_IDS.write_text("\n".join(ids) + "\n")

    hf_scores = text @ audio.T
    pinned_scores = np.load(PINNED_TEXT) @ np.load(PINNED_CACHE).T
    assert PINNED_IDS.read_text().split() == ids

    hf_top1 = hf_scores.argmax(axis=1)
    pinned_top1 = pinned_scores.argmax(axis=1)
    hf_top5 = np.argsort(-hf_scores, axis=1)[:, :5]
    pinned_top5 = np.argsort(-pinned_scores, axis=1)[:, :5]
    retrieval = load_retrieval()
    assert [ids[j] for j in pinned_top1] == [retrieval[str(i)] for i in range(100)]

    rows = np.arange(100)
    result = {
        "top1_agreement": float((hf_top1 == pinned_top1).mean()),
        "pinned_top1_in_hf_top5": float(np.mean([pinned_top1[i] in hf_top5[i] for i in rows])),
        "mean_top5_overlap": float(np.mean([len(set(hf_top5[i]) & set(pinned_top5[i])) for i in rows])),
        "pearson_all_scores": float(stats.pearsonr(hf_scores.ravel(), pinned_scores.ravel()).statistic),
        "pearson_top1_scores": float(stats.pearsonr(hf_scores[rows, hf_top1], pinned_scores[rows, pinned_top1]).statistic),
        "mean_per_caption_spearman": float(np.mean([stats.spearmanr(hf_scores[i], pinned_scores[i]).statistic for i in rows])),
        "hf_top1_score_range": [float(hf_scores[rows, hf_top1].min()), float(hf_scores[rows, hf_top1].max())],
        "model": "laion/larger_clap_music (HF) vs music_audioset_epoch_15_esc_90.14.pt",
    }
    (RESULTS_DIR / "retrieval_crosscheck_hf.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
