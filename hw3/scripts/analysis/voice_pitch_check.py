import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from evaluation.vocal_stems import SR_DEMUCS, separate_vocals  # noqa: E402

PITCH_SR, HOP = 22050, 512
MIN_FRAMES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice pitch from the vocal stem only, with bleed controls, on the female/male clips of scripts/analysis/dsp_validity.py: "
        "all voiced frames (as edit_features.py), frames where the stem is within 20 dB of its loudest frame, frames with pYIN "
        "voiced probability >= 0.5, the share of confident frames above 200 Hz, and the same with near-silent stems dropped "
        "(stem below mix by more than --min-stem-db). Run with the hw1 venv and -P. Writes results/analysis/voice_pitch_check.csv "
        "and voice_pitch_check_clips.csv.")
    parser.add_argument("--min-stem-db", type=float, default=-20.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def pitch_features(vocals: np.ndarray, mix: np.ndarray) -> dict:
    y = librosa.resample(vocals, orig_sr=SR_DEMUCS, target_sr=PITCH_SR)
    f0, voiced, probability = librosa.pyin(y, fmin=65, fmax=1000, sr=PITCH_SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][: len(f0)]
    loud = 20 * np.log10(rms + 1e-10) > 20 * np.log10(rms.max() + 1e-10) - 20
    sets = {"all_voiced": voiced, "energy_gated": voiced & loud, "confident": voiced & loud & (probability >= 0.5)}
    out = {"stem_vs_mix_db": float(10 * np.log10((vocals ** 2).mean() / ((mix ** 2).mean() + 1e-12) + 1e-12))}
    for name, mask in sets.items():
        values = f0[mask & np.isfinite(f0)]
        out[f"f0_median_{name}"] = float(np.median(values)) if len(values) >= MIN_FRAMES else np.nan
        out[f"frames_{name}"] = int(len(values))
    confident = f0[sets["confident"] & np.isfinite(f0)]
    out["share_above_200hz"] = float((confident > 200).mean()) if len(confident) >= MIN_FRAMES else np.nan
    return out


def main():
    args = parse_args()
    clips = pd.read_csv(HW3 / "results/analysis/dsp_validity_clips.csv")
    clips = clips[clips["check"] == "voice pitch"][["ytid", "positive"]]
    separator = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(args.device)
    rows, start = [], time.time()
    for row in clips.itertuples():
        audio, rate = sf.read(HW3 / f"data/musiccaps/audio_mp3/{row.ytid}.mp3", dtype="float32", always_2d=True)
        stereo = audio.T if rate == SR_DEMUCS else librosa.resample(audio.T, orig_sr=rate, target_sr=SR_DEMUCS)
        if stereo.shape[0] == 1:
            stereo = np.repeat(stereo, 2, axis=0)
        with torch.inference_mode():
            vocals = separate_vocals(separator, torch.from_numpy(np.ascontiguousarray(stereo)).to(args.device)).mean(0).cpu().numpy()
        rows.append({"ytid": row.ytid, "female": row.positive, **pitch_features(vocals, stereo.mean(0))})
    table = pd.DataFrame(rows)
    out = HW3 / "results/analysis"
    table.to_csv(out / "voice_pitch_check_clips.csv", index=False)
    summary = []
    for subset, part in (("all clips", table), ("audible stem", table[table["stem_vs_mix_db"] > args.min_stem_db])):
        for column in [c for c in table.columns if c.startswith("f0_median") or c == "share_above_200hz"]:
            keep = part[column].notna()
            values, target = part.loc[keep, column], part.loc[keep, "female"]
            summary.append({"subset": subset, "feature": column, "female": int(target.sum()), "male": int((1 - target).sum()),
                            "auc": roc_auc_score(target, values) if target.nunique() == 2 else np.nan,
                            "median_female": float(values[target == 1].median()), "median_male": float(values[target == 0].median())})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "voice_pitch_check.csv", index=False)
    print(summary.round(3).to_string(index=False), f"\n{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
