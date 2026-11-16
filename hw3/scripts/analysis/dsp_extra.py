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

HOP = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Candidate detectors beyond HW1's features, on the same real MusicCaps clips as scripts/analysis/dsp_validity.py "
        "(read from dsp_validity_clips.csv): onset rate on the mix, the percussive part (HPSS) and the Demucs drum stem, "
        "prior-free tempo; native-rate band energy above 12 and 16 kHz, flatness, clipping, stereo width, noise floor; crest "
        "factor, percussive share, onset strength, centroid. Run with the hw1 venv and -P. Writes results/analysis/dsp_extra.csv "
        "(AUC per check and feature) and dsp_extra_clips.csv.")
    parser.add_argument("--drum-clips", type=int, default=60, help="per group, for the Demucs drum stem (slow on CPU)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def onset_rate(y: np.ndarray, sr: int) -> float:
    return len(librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")) / (len(y) / sr)


def free_tempo(y: np.ndarray, sr: int) -> float:
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    return float(librosa.feature.tempo(onset_envelope=envelope, sr=sr, hop_length=HOP, prior=None)[0])


def band_share(power: np.ndarray, freqs: np.ndarray, low: float) -> float:
    return float(power[freqs >= low].sum() / (power.sum() + 1e-12))


def features(stereo: np.ndarray, rate: int, separator, device: str, drums: bool) -> dict:
    mono = stereo.mean(0)
    y = librosa.resample(mono, orig_sr=rate, target_sr=22050)
    harmonic, percussive = librosa.effects.hpss(y)
    spectrum = np.abs(librosa.stft(mono, n_fft=4096, hop_length=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=rate, n_fft=4096)
    frame_rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=1024)[0]
    mid, side = stereo.mean(0), (stereo[0] - stereo[-1]) / 2
    out = {"onset_rate_mix": onset_rate(y, 22050), "onset_rate_percussive": onset_rate(percussive, 22050),
           "tempo_no_prior": free_tempo(percussive, 22050),
           "onset_strength_mean": float(librosa.onset.onset_strength(y=y, sr=22050).mean()),
           "percussive_share": float((percussive ** 2).sum() / ((y ** 2).sum() + 1e-12)),
           "crest_factor_db": float(20 * np.log10(np.abs(mono).max() / (np.sqrt((mono ** 2).mean()) + 1e-12) + 1e-12)),
           "rms_db": float(20 * np.log10(np.sqrt((mono ** 2).mean()) + 1e-12)),
           "centroid_hz": float(librosa.feature.spectral_centroid(y=y, sr=22050).mean()),
           "share_above_12khz": band_share(spectrum, freqs, 12000), "share_above_16khz": band_share(spectrum, freqs, 16000),
           "flatness": float(librosa.feature.spectral_flatness(y=y).mean()),
           "clipped_fraction": float((np.abs(mono) > 0.99 * np.abs(mono).max()).mean()),
           "stereo_side_db": float(10 * np.log10((side ** 2).mean() / ((mid ** 2).mean() + 1e-12) + 1e-12)),
           "noise_floor_db": float(20 * np.log10(np.percentile(frame_rms, 5) + 1e-12) - 20 * np.log10(np.median(frame_rms) + 1e-12)),
           "native_rate": rate}
    if drums:
        if rate != SR_DEMUCS:
            stereo = librosa.resample(stereo, orig_sr=rate, target_sr=SR_DEMUCS)
        if stereo.shape[0] == 1:
            stereo = np.repeat(stereo, 2, axis=0)
        mix = torch.from_numpy(np.ascontiguousarray(stereo)).to(device)
        drum = separate_vocals(separator, mix, source="drums").mean(0).cpu().numpy()
        drum = librosa.resample(drum, orig_sr=SR_DEMUCS, target_sr=22050)
        out.update({"onset_rate_drums": onset_rate(drum, 22050), "tempo_drums_no_prior": free_tempo(drum, 22050),
                    "drum_share_db": float(10 * np.log10((drum ** 2).mean() / ((y ** 2).mean() + 1e-12) + 1e-12))})
    return out


def main():
    args = parse_args()
    clips = pd.read_csv(HW3 / "results/analysis/dsp_validity_clips.csv")[["check", "ytid", "positive"]]
    clips = clips[clips["check"] != "voice pitch"]
    separator = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(args.device)
    rows, start = [], time.time()
    for check, part in clips.groupby("check"):
        counts = {0: 0, 1: 0}
        for row in part.itertuples():
            drums = check == "tempo" and counts[row.positive] < args.drum_clips
            counts[row.positive] += drums
            audio, rate = sf.read(HW3 / f"data/musiccaps/audio_mp3/{row.ytid}.mp3", dtype="float32", always_2d=True)
            rows.append({"check": check, "ytid": row.ytid, "positive": row.positive, **features(audio.T, rate, separator, args.device, drums)})
        print(check, "done", f"{time.time() - start:.0f} s", flush=True)
    table = pd.DataFrame(rows)
    out = HW3 / "results/analysis"
    table.to_csv(out / "dsp_extra_clips.csv", index=False)
    summary = []
    for check, part in table.groupby("check"):
        for column in part.columns.drop(["check", "ytid", "positive", "native_rate"]):
            values = part[column].to_numpy(dtype=float)
            keep = np.isfinite(values)
            if keep.sum() > 20 and part["positive"][keep].nunique() == 2 and np.ptp(values[keep]) > 0:
                auc = roc_auc_score(part["positive"][keep], values[keep])
                summary.append({"check": check, "feature": column, "clips": int(keep.sum()), "auc": auc, "separation": abs(auc - 0.5)})
    summary = pd.DataFrame(summary).sort_values(["check", "separation"], ascending=[True, False])
    summary.to_csv(out / "dsp_extra.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
