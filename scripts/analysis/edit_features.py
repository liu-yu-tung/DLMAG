import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
sys.path.insert(0, str(HW3.parent / "hw1" / "src"))
from hw1.features.handcrafted import SR, extract  # noqa: E402
from evaluation.vocal_stems import SR_DEMUCS, separate_vocals  # noqa: E402
from analysis.dsp_extra import features as extra_features  # noqa: E402
from analysis.voice_pitch_check import pitch_features  # noqa: E402

PITCH_SR = 22050
VOICE_AXES = {"voice_male_to_female"}
# candidate detectors checked on real audio (scripts/analysis/dsp_extra.py, scripts/analysis/voice_pitch_check.py); drum stem only where tempo matters
DRUM_AXES = {"fast_to_slow"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="HW1 hand-crafted features (and vocal-stem pitch for the voice axis) for each word-swap pair: the original "
        "clip outputs/dev/<base>/<id>.wav and the edited clip outputs/dev/edit_<axis>/<id>.wav. Run with the hw1 venv and -P. "
        "Writes results/analysis/edit_features.csv (one row per axis, id, version).")
    parser.add_argument("--base", default="rescale12")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--extended", action="store_true", help="also the edit_<axis>_ext pairs (manifest ext_ids)")
    parser.add_argument("--axes", default=None, help="comma-separated subset; default all axes in the manifest")
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.T, rate


def vocal_pitch(separator, stereo: np.ndarray, rate: int, device: str) -> dict:
    assert rate == SR_DEMUCS, f"clips are {rate} Hz, separator wants {SR_DEMUCS}"
    vocals = separate_vocals(separator, torch.from_numpy(stereo).to(device)).mean(0).cpu().numpy()
    mono = librosa.resample(vocals, orig_sr=rate, target_sr=PITCH_SR)
    f0, voiced, _ = librosa.pyin(mono, fmin=65, fmax=1000, sr=PITCH_SR)
    pitched = f0[voiced & np.isfinite(f0)]
    rms = float(20 * np.log10(np.sqrt(np.mean(vocals ** 2)) + 1e-10))
    return {"vocal_rms_dbfs": rms, "vocal_voiced_fraction": float(voiced.mean()),
            "vocal_f0_median_hz": float(np.median(pitched)) if len(pitched) >= 20 else np.nan}


def main():
    args = parse_args()
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    axes = args.axes.split(",") if args.axes else [a for a in manifest if (HW3 / "outputs/dev" / f"edit_{a}").is_dir()]
    separator = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(args.device)
    rows, cache = [], {}
    for axis in axes:
        pairs = [(i, "registered", f"edit_{axis}") for i in manifest[axis]["ids"]]
        if args.extended:
            pairs += [(i, "extended", f"edit_{axis}_ext") for i in manifest[axis]["ext_ids"]]
        for clip_id, subset, edit_run in pairs:
            # seed1/seed2 = original caption on other seeds: the size of a change that needs no word change
            for version, folder in (("original", args.base), ("edited", edit_run), ("seed1", f"{args.base}_seed1"),
                                    ("seed2", f"{args.base}_seed2")):
                path = HW3 / "outputs/dev" / folder / f"{clip_id}.wav"
                key = (path, axis in VOICE_AXES, axis in DRUM_AXES)
                if key not in cache:
                    stereo, rate = load(path)
                    cache[key] = extract(librosa.resample(stereo.mean(0), orig_sr=rate, target_sr=SR))
                    cache[key].update(extra_features(stereo, rate, separator, args.device, axis in DRUM_AXES))
                    if axis in VOICE_AXES:
                        cache[key].update(vocal_pitch(separator, stereo, rate, args.device))
                        with torch.inference_mode():
                            vocals = separate_vocals(separator, torch.from_numpy(stereo).to(args.device)).mean(0).cpu().numpy()
                        cache[key].update({f"pitch_{k}": v for k, v in pitch_features(vocals, stereo.mean(0)).items()})
                rows.append({"axis": axis, "id": clip_id, "subset": subset, "version": version, **cache[key]})
        print(axis, "done", flush=True)
    out = HW3 / "results/analysis/edit_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out, len(rows), "rows")


if __name__ == "__main__":
    main()
