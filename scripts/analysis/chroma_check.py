import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HW3 = Path(__file__).resolve().parents[2]
SR, HOP, EPS = 22050, 512, 1e-10
# percussion and noise also spread energy over all 12 pitch classes, so they are held as controls (side-agent note 23:16)
CONTROLS = ["percussive_share", "flatness", "onset_strength_mean"]
CHROMA = ["harmony_chroma_entropy_mean", "harmonic_frame_entropy", "harmonic_profile_entropy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Does chroma entropy measure harmony or percussion? On the real clips of dsp_validity.py (tempo, loudness) "
        "and dsp_detectors.py (content pairs), chroma entropy of the HPSS harmonic part only: per frame (as HW1) and of the "
        "clip-average pitch-class profile (how many of the 12 notes the clip uses). AUC of each, and the cross-validated AUC "
        "gained over a percussion/noise control model (percussive share, flatness, onset strength). Also writes the same "
        "features for every word-swap clip (results/analysis/edit_chroma.csv) for edit_effects.py. Run with the hw1 venv.")
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def harmonic_entropy(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    y = librosa.resample(audio.mean(1), orig_sr=rate, target_sr=SR)
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP)
    frames = chroma / (chroma.sum(axis=0, keepdims=True) + EPS)
    profile = chroma.mean(axis=1) / (chroma.mean(axis=1).sum() + EPS)
    return {"harmonic_frame_entropy": float(-(frames * np.log(frames + EPS)).sum(axis=0).mean()),
            "harmonic_profile_entropy": float(-(profile * np.log(profile + EPS)).sum())}


def gain_over_controls(part: pd.DataFrame, feature: str) -> tuple[float, float]:
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    part = part.dropna(subset=CONTROLS + [feature])
    target = part["positive"].to_numpy()
    scores = [roc_auc_score(target, cross_val_predict(model, part[cols].to_numpy(float), target, cv=folds,
                                                      method="decision_function")) for cols in (CONTROLS, CONTROLS + [feature])]
    return scores[0], scores[1]


def main():
    args = parse_args()
    out = HW3 / "results/analysis"
    old = pd.read_csv(out / "dsp_validity_clips.csv").merge(
        pd.read_csv(out / "dsp_extra_clips.csv").drop(columns="positive"), on=["check", "ytid"])
    old = old[old["check"].isin(["tempo", "loudness"])].assign(sample="A")
    new = pd.read_csv(out / "dsp_detectors_clips.csv")
    clips = pd.concat([old, new[new["sample"] == "A"]], ignore_index=True)
    manifest = json.loads((out / "edits_manifest.json").read_text())
    edit_rows = []
    for axis, entry in manifest.items():
        for subset, suffix, ids in (("registered", "", entry["ids"]), ("extended", "_ext", entry.get("ext_ids", []))):
            for clip_id in ids:
                for version, folder in (("original", "rescale12"), ("edited", f"edit_{axis}{suffix}"),
                                        ("seed1", "rescale12_seed1"), ("seed2", "rescale12_seed2")):
                    path = HW3 / "outputs/dev" / folder / f"{clip_id}.wav"
                    if path.exists():
                        edit_rows.append({"axis": axis, "id": clip_id, "subset": subset, "version": version, "path": path})
    paths = sorted({str(HW3 / f"data/musiccaps/audio_mp3/{t}.mp3") for t in clips["ytid"]} | {str(r["path"]) for r in edit_rows})
    start = time.time()
    values = dict(zip(paths, Parallel(n_jobs=args.jobs, verbose=2)(delayed(harmonic_entropy)(Path(p)) for p in paths)))
    print(f"{len(paths)} files, {time.time() - start:.0f} s", flush=True)
    clips = clips.join(pd.DataFrame([values[str(HW3 / f"data/musiccaps/audio_mp3/{t}.mp3")] for t in clips["ytid"]], index=clips.index))
    rows = []
    for check, part in clips.groupby("check"):
        for feature in CHROMA:
            keep = part[feature].notna()
            control, both = gain_over_controls(part, feature)
            rows.append({"check": check, "feature": feature, "n": int(keep.sum()),
                         "auc": roc_auc_score(part["positive"][keep], part[feature][keep]),
                         "corr_with_percussive_share": float(part[feature].corr(part["percussive_share"])),
                         "auc_controls_cv": control, "auc_controls_plus_feature_cv": both, "gain": both - control})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "chroma_check.csv", index=False)
    edits = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} | values[str(r["path"])] for r in edit_rows])
    edits.to_csv(out / "edit_chroma.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
