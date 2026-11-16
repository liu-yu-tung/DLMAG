"""Evaluate a folder of generated clips (0.wav .. 99.wav) with av-benchmark. Run with the av-benchmark venv; see docs/eval_notes.md."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# keep the script folder off sys.path so its file names cannot shadow installed modules
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import soundfile
import torch
import torch.nn.functional as F

from text_to_music.eval_common import (
    AV_BENCH_DIR,
    AV_BENCH_PYTHON,
    CLAP_CHECKPOINT,
    HW3_ROOT,
    MUSICCAPS_DIR,
    RESULTS_DIR,
    RETRIEVED_DIR,
    TEST_CSV,
    FrechetReference,
    bootstrap_frechet,
    bootstrap_indices,
    code_version,
    load_retrieval,
    percentile_interval,
)

AUDIO_LENGTH_S = 10
REFERENCE_DIR = HW3_ROOT / "outputs" / "ref"
MUSICCAPS_CACHE = REFERENCE_DIR / "musiccaps_cache"
DEV_DIR = REFERENCE_DIR / "dev_real"
DEV_CSV = HW3_ROOT / "data" / "dev.csv"
DEV_META = HW3_ROOT / "data" / "dev_meta.csv"
N_PROMPTS = 100
FD_FEATURES = {"VGG": "vggish_features.pth", "PANN": "pann_features.pth", "PASST": "passt_features_embed.pth"}
FD_KEYS = {"VGG": "FD-VGG", "PANN": "FD-PANN", "PASST": "FD-PASST"}


def build_reference(directory: Path, clips: dict[str, str]) -> None:
    """MusicCaps clips as 0.wav..99.wav (native sample rate, 16-bit) so file names pair with the predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    for prompt_id, youtube_id in clips.items():
        target = directory / f"{prompt_id}.wav"
        if target.exists():
            continue
        data, sample_rate = soundfile.read(str(MUSICCAPS_DIR / f"{youtube_id}.mp3"), dtype="float32", always_2d=True)
        soundfile.write(target, data, sample_rate, subtype="PCM_16")
    assert len(list(directory.glob("*.wav"))) == N_PROMPTS


def check_prediction_folder(directory: Path) -> None:
    for prompt_id in range(N_PROMPTS):
        path = directory / f"{prompt_id}.wav"
        assert path.exists(), f"missing {path}"
        info = soundfile.info(path)
        assert info.duration >= AUDIO_LENGTH_S - 0.01, f"{path} is {info.duration:.2f} s, need at least {AUDIO_LENGTH_S} s"


def run_av_benchmark(*arguments: str) -> None:
    command = [str(AV_BENCH_PYTHON), *arguments]
    subprocess.run(command, cwd=AV_BENCH_DIR, check=True)


def ensure_reference_cache(cache_dir: Path, captions_csv: Path, text_csv: Path) -> None:
    captions = pd.read_csv(captions_csv)
    pd.DataFrame({"name": captions["id"], "caption": captions["caption"]}).to_csv(text_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    if not (cache_dir / "clap_laion_text.pth").exists():
        run_av_benchmark(
            "extract_text.py", "--text_csv", str(text_csv), "--output_cache_path", str(cache_dir),
            "--clap_ckpt", str(CLAP_CHECKPOINT), "--skip_ms_clap",
        )


def ensure_musiccaps_cache() -> None:
    if (MUSICCAPS_CACHE / "passt_features_embed.pth").exists():
        return
    sys.path.insert(0, str(AV_BENCH_DIR))
    from av_bench.extract import extract

    extract(
        audio_path=MUSICCAPS_DIR, output_path=MUSICCAPS_CACHE, audio_length=AUDIO_LENGTH_S, device="cuda",
        batch_size=1, num_workers=4, skip_video_related=True, skip_clap=True, skip_ms_clap=True,
        clap_ckpt_path=CLAP_CHECKPOINT,
    )


def load_cache(cache_dir: Path) -> dict[str, dict]:
    return {
        "VGG": torch.load(cache_dir / "vggish_features.pth", weights_only=True),
        "PANN": torch.load(cache_dir / "pann_features.pth", weights_only=True),
        "PASST": torch.load(cache_dir / "passt_features_embed.pth", weights_only=True),
        "PASST_logits": torch.load(cache_dir / "passt_logits.pth", weights_only=True),
    }


def feature_groups(cache: dict, kind: str, names: list[str]) -> list[np.ndarray]:
    if kind == "PANN":
        return [cache["PANN"][name]["2048"].numpy()[None] for name in names]
    if kind == "VGG":
        return [cache["VGG"][name].numpy() for name in names]
    return [cache["PASST"][name].numpy()[None] for name in names]


def per_clip_kl(ground_truth_logits: torch.Tensor, predicted_logits: torch.Tensor) -> np.ndarray:
    """KL(ground truth || prediction) of the softmax class distributions, one value per clip (same formula as av_bench kl)."""
    kl = F.kl_div(F.log_softmax(predicted_logits, dim=1), F.log_softmax(ground_truth_logits, dim=1), reduction="none", log_target=True)
    return kl.sum(dim=1).numpy()


def mean_with_interval(values: np.ndarray, indices: np.ndarray) -> dict:
    means = values[indices].mean(axis=1)
    return {"value": float(values.mean()), "ci95": percentile_interval(means)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run av-benchmark on <pred>/0.wav..99.wav against the retrieved MusicCaps clips.")
    parser.add_argument("--pred", type=Path, required=True, help="folder with 0.wav .. 99.wav, at least 10 s each")
    parser.add_argument("--set", dest="set_name", choices=("test", "dev"), default="test",
                        help="test: captions vs retrieved clips; dev: dev captions vs their own MusicCaps clips")
    parser.add_argument("--gt", type=Path, default=None, help="reference folder 0.wav..99.wav; default is built for the set")
    parser.add_argument("--name", default=None, help="run name; results go to results/<name>/ (default: name of the pred folder)")
    parser.add_argument("--batch-size", type=int, default=20, help="prediction batch size for feature extraction (GPU memory)")
    parser.add_argument("--recompute-reference", action="store_true", help="rebuild the reference feature caches")
    args = parser.parse_args()

    if args.set_name == "test":
        default_gt, clips, captions_csv = RETRIEVED_DIR, load_retrieval(), TEST_CSV
        gt_cache, text_csv = REFERENCE_DIR / "test_retrieved_cache", REFERENCE_DIR / "test_retrieved_text.csv"
    else:
        meta = pd.read_csv(DEV_META, dtype=str)
        default_gt, clips, captions_csv = DEV_DIR, dict(zip(meta["id"], meta["ytid"])), DEV_CSV
        gt_cache, text_csv = REFERENCE_DIR / "dev_real_cache", REFERENCE_DIR / "dev_real_text.csv"
    pred_dir, gt_dir = args.pred.resolve(), (args.gt or default_gt).resolve()
    run_name = args.name or pred_dir.name
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_cache = HW3_ROOT / "outputs" / "pred_cache" / run_name

    if gt_dir == default_gt.resolve():
        build_reference(gt_dir, clips)
    check_prediction_folder(pred_dir)
    if args.recompute_reference:
        for stale in gt_cache.glob("*.pth"):
            stale.unlink()
    ensure_reference_cache(gt_cache, captions_csv, text_csv)

    run_av_benchmark(
        "evaluate.py", "--gt_audio", str(gt_dir), "--gt_cache", str(gt_cache),
        "--pred_audio", str(pred_dir), "--pred_cache", str(pred_cache), "--audio_length", str(AUDIO_LENGTH_S),
        "--skip_video_related", "--skip_ms_clap", "--clap_ckpt", str(CLAP_CHECKPOINT),
        "--num_workers", "4", "--pred_batch_size", str(args.batch_size), "--recompute_pred_cache",
    )
    raw = json.loads((pred_cache / "output_metrics.json").read_text())

    ensure_musiccaps_cache()
    musiccaps_cache = load_cache(MUSICCAPS_CACHE)
    gt = load_cache(gt_cache)
    pred = load_cache(pred_cache)
    names = [str(i) for i in range(N_PROMPTS)]
    indices = bootstrap_indices(N_PROMPTS)

    metrics: dict = {
        "run_name": run_name,
        "code_version": code_version(),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "set": args.set_name,
        "audio_length_s": AUDIO_LENGTH_S,
        "clap_checkpoint": CLAP_CHECKPOINT.name,
        "bootstrap": {"resamples": len(indices), "seed": 0, "unit": "prompt id", "interval": "percentile 95%"},
    }

    text = torch.load(gt_cache / "clap_laion_text.pth", weights_only=True)
    audio = torch.load(pred_cache / "clap_laion_audio.pth", weights_only=True)
    clap = np.array([F.cosine_similarity(text[name], audio[name], dim=0).item() for name in names])
    kl_passt = per_clip_kl(
        torch.stack([gt["PASST_logits"][name] for name in names]), torch.stack([pred["PASST_logits"][name] for name in names])
    )
    kl_panns = per_clip_kl(
        torch.stack([gt["PANN"][name]["logits"] for name in names]), torch.stack([pred["PANN"][name]["logits"] for name in names])
    )
    for key, values in (("LAION-CLAP-Score", clap), ("KL-PASST-softmax", kl_passt), ("KL-PANNS-softmax", kl_panns)):
        assert np.isclose(values.mean(), raw[key], rtol=1e-4, atol=1e-5), (key, values.mean(), raw[key])
        metrics[key] = mean_with_interval(values, indices)
    for key in ("ISC-PANNS-mean", "ISC-PANNS-std", "ISC-PASST-mean", "ISC-PASST-std"):
        metrics[key] = {"value": raw[key], "ci95": None}

    for kind, key in FD_KEYS.items():
        pred_groups = feature_groups(pred, kind, names)
        retrieved_reference = FrechetReference(np.concatenate(feature_groups(gt, kind, names)))
        point, interval, shifted = bootstrap_frechet(retrieved_reference, pred_groups, indices)
        assert np.isclose(point, raw[key], rtol=1e-2, atol=1e-3), (key, point, raw[key])
        metrics[key] = {"value": raw[key], "ci95": interval, "bootstrap_mean": shifted, "value_fast_formula": point}

        musiccaps_names = list(musiccaps_cache[kind])
        musiccaps_reference = FrechetReference(np.concatenate(feature_groups(musiccaps_cache, kind, musiccaps_names)))
        point, interval, shifted = bootstrap_frechet(musiccaps_reference, pred_groups, indices)
        metrics[f"{key}_musiccaps"] = {"value": point, "ci95": interval, "bootstrap_mean": shifted}

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    pd.DataFrame(
        {"id": range(N_PROMPTS), "clap_cosine": clap, "kl_passt_softmax": kl_passt, "kl_panns_softmax": kl_panns}
    ).to_csv(run_dir / "per_prompt.csv", index=False)
    for key, entry in metrics.items():
        if isinstance(entry, dict) and "value" in entry:
            print(f"{key:<22} {entry['value']:.4f}  {entry['ci95']}")


if __name__ == "__main__":
    main()
