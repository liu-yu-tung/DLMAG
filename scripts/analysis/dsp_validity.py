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
sys.path.insert(0, str(HW3.parent / "hw1" / "src"))
from analysis.attribute_probes import labels, training_set  # noqa: E402
from analysis.edit_features import vocal_pitch  # noqa: E402
from hw1.features.handcrafted import SR, extract  # noqa: E402
from evaluation.vocal_stems import SR_DEMUCS  # noqa: E402

# (name, positive group, negative group, feature): AUC > 0.5 means the feature is higher for the positive group
CHECKS = [("tempo", "tempo: fast", "tempo: slow", "rhythm_tempo_bpm"),
          ("loudness", "dynamics: loud, intense", "dynamics: soft, calm", "energy_rms_db_mean"),
          ("quality", "recording: high quality", "recording: low quality", "energy_band_6000_12000hz_ratio"),
          ("voice pitch", "vocals: female voice", "vocals: male voice", "vocal_f0_median_hz")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DSP detector validity (plan P1.4): do the HW1 hand-crafted features used by the word-swap test track the "
        "caption words on real MusicCaps audio? For each pair of word groups, clips with one group and not the other (probe "
        "training pool: no dev or test-retrieved clips), up to --per-group each; AUC of the single feature. Run with the hw1 "
        "venv and -P. Writes results/analysis/dsp_validity.csv and dsp_validity_clips.csv.")
    parser.add_argument("--per-group", type=int, default=150)
    parser.add_argument("--pitch-per-group", type=int, default=60, help="vocal pitch needs source separation, so fewer clips")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def load(ytid: str):
    audio, rate = sf.read(HW3 / f"data/musiccaps/audio_mp3/{ytid}.mp3", dtype="float32", always_2d=True)
    return audio.T, rate


def main():
    args = parse_args()
    train, _ = training_set()
    y = labels(train["caption"])
    separator = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(args.device)
    rows, summary = [], []
    start = time.time()
    for name, positive, negative, feature in CHECKS:
        cap = args.pitch_per_group if feature.startswith("vocal") else args.per_group
        values, target = [], []
        for group, other, label in ((positive, negative, 1), (negative, positive, 0)):
            pool = train[(y[group] & ~y[other]).to_numpy()]
            pool = pool[pool["ytid"].map(lambda t: (HW3 / f"data/musiccaps/audio_mp3/{t}.mp3").exists())]
            for ytid in pool.sample(min(cap, len(pool)), random_state=0)["ytid"]:
                stereo, rate = load(ytid)
                if feature.startswith("vocal"):
                    if rate != SR_DEMUCS:
                        stereo, rate = librosa.resample(stereo, orig_sr=rate, target_sr=SR_DEMUCS), SR_DEMUCS
                    if stereo.shape[0] == 1:
                        stereo = np.repeat(stereo, 2, axis=0)
                    value = vocal_pitch(separator, np.ascontiguousarray(stereo), rate, args.device)[feature]
                    every = {feature: value}
                else:
                    every = extract(librosa.resample(stereo.mean(0), orig_sr=rate, target_sr=SR))
                    value = every[feature]
                values.append(value)
                target.append(label)
                rows.append({"check": name, "ytid": ytid, "positive": label, **every})
        values, target = np.array(values), np.array(target)
        keep = np.isfinite(values)
        summary.append({"check": name, "feature": feature, "positive_group": positive, "negative_group": negative,
                        "n_positive": int(target[keep].sum()), "n_negative": int((1 - target[keep]).sum()),
                        "auc": roc_auc_score(target[keep], values[keep]),
                        "median_positive": float(np.median(values[keep][target[keep] == 1])),
                        "median_negative": float(np.median(values[keep][target[keep] == 0]))})
        print(summary[-1], f"{time.time() - start:.0f} s", flush=True)
    out = HW3 / "results/analysis"
    pd.DataFrame(summary).to_csv(out / "dsp_validity.csv", index=False)
    clips = pd.DataFrame(rows)
    clips.to_csv(out / "dsp_validity_clips.csv", index=False)
    # every HW1 feature per check, so an axis whose named feature fails can use the best one chosen on real audio only
    every = []
    for name, part in clips.groupby("check"):
        for column in part.columns.drop(["check", "ytid", "positive"]):
            values = part[column].to_numpy(dtype=float)
            keep = np.isfinite(values)
            if keep.sum() > 20 and len(set(part["positive"][keep])) == 2 and np.ptp(values[keep]) > 0:
                auc = roc_auc_score(part["positive"][keep], values[keep])
                every.append({"check": name, "feature": column, "auc": auc, "separation": abs(auc - 0.5)})
    pd.DataFrame(every).sort_values(["check", "separation"], ascending=[True, False]).to_csv(out / "dsp_validity_all.csv", index=False)


if __name__ == "__main__":
    main()
