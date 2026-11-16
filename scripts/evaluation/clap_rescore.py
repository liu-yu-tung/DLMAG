"""Score generated clips with a chosen LAION-CLAP checkpoint; write a per_prompt.csv column.

Run with external/av-benchmark/.venv/bin/python. Preprocessing copies av_bench AudioDataset
(mono, DC removal, resample to 48 kHz, cut to 10 s) so the 90.14 checkpoint reproduces `clap_cosine`.
"""

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

import laion_clap
import pandas as pd
import soundfile
import torch
import torch.nn.functional as F
import torchaudio

HW3_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 48_000
AUDIO_LENGTH_S = 10
N_PROMPTS = 100


def load_clip(path: Path, resamplers: dict) -> torch.Tensor:
    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T).mean(dim=0)
    waveform = waveform - waveform.mean()
    if sample_rate != SAMPLE_RATE:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resamplers[sample_rate](waveform)
    return waveform[: SAMPLE_RATE * AUDIO_LENGTH_S].unsqueeze(0)


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    return ((x.contiguous().view(torch.int32) + 4096) & -8192).view(torch.float32)


@contextlib.contextmanager
def tf32_convolutions(enabled: bool):
    # Why: the 90.14 scores came from a GPU, where cuDNN runs the model's convolutions (incl. the STFT) in TF32; plain fp32 on CPU differs by up to 0.02. Never around resampling, which is a CPU conv1d in av-benchmark.
    originals = {name: getattr(F, name) for name in ("conv1d", "conv2d")}
    if enabled:
        for name, original in originals.items():
            setattr(F, name, lambda x, w, *a, _f=original, **k: _f(round_to_tf32(x), round_to_tf32(w), *a, **k))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(F, name, original)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="set_name", required=True, choices=["dev", "test"])
    parser.add_argument("--runs", required=True, help="comma-separated run names")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="score only ids 0..N-1 (check runs; other rows stay empty)")
    args = parser.parse_args()
    runs = [r for r in args.runs.split(",") if r]
    assert runs and args.ckpt.is_file(), args.ckpt
    count = N_PROMPTS if args.limit is None else args.limit
    assert 1 <= count <= N_PROMPTS

    captions = pd.read_csv(HW3_ROOT / "data" / f"{args.set_name}.csv")
    assert list(captions["id"]) == list(range(N_PROMPTS)), "caption ids must be 0..99"

    if args.device == "cuda":
        assert torch.backends.cudnn.allow_tf32, "scores must use the same TF32 convolutions as the evaluation"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=args.device).eval()
    model.load_ckpt(args.ckpt, verbose=False)
    model = model.to(args.device)

    with torch.inference_mode():
        text = torch.cat(
            [model.get_text_embedding([captions["caption"][i]], use_tensor=True) for i in range(count)]
        ).cpu()
        assert text.ndim == 2 and text.shape[0] == count, text.shape

        for run in runs:
            audio_dir = HW3_ROOT / "outputs" / args.set_name / run
            run_dir = HW3_ROOT / "results" / f"{args.set_name}__{run}"
            csv_path = run_dir / "per_prompt.csv"
            table = pd.read_csv(csv_path)
            assert list(table["id"]) == list(range(N_PROMPTS)), f"{csv_path}: ids must be 0..99 in order"

            resamplers: dict = {}
            scores = []
            for i in range(count):
                wav = load_clip(audio_dir / f"{i}.wav", resamplers).to(args.device)
                with tf32_convolutions(args.device == "cpu"):
                    audio = model.get_audio_embedding_from_data(wav, use_tensor=True).cpu()
                assert audio.shape == text[i : i + 1].shape, (audio.shape, text.shape)
                scores.append(F.cosine_similarity(text[i], audio[0], dim=0).item())
            assert len(scores) == count

            table[args.column] = pd.Series(scores, index=range(count), dtype="float64")
            write_csv_atomic(table, csv_path)
            summary = {
                "column": args.column,
                "mean": sum(scores) / len(scores),
                "checkpoint": args.ckpt.name,
                "n_clips": len(scores),
            }
            (run_dir / f"{args.column}.json").write_text(json.dumps(summary, indent=1) + "\n")
            print(f"{args.set_name}__{run}: {args.column} mean {summary['mean']:.5f} over {len(scores)} clips")


if __name__ == "__main__":
    main()
