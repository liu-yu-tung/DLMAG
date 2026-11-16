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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
sys.path.insert(0, str(HW3.parent / "hw1" / "src"))
from analysis.attribute_probes import labels, training_set  # noqa: E402
from analysis.dsp_extra import features as extra_features  # noqa: E402
from hw1.features.handcrafted import SR, extract  # noqa: E402

CHROMA = "harmony_chroma_entropy_mean"
# check: (positive group, negative group); the first four reuse the clips of scripts/analysis/dsp_validity.py and dsp_extra.py
CHECKS = {"tempo": ("tempo: fast", "tempo: slow"), "loudness": ("dynamics: loud, intense", "dynamics: soft, calm"),
          "quality": ("recording: high quality", "recording: low quality"),
          "jazz vs country": ("genre: jazz, blues", "genre: folk, country"),
          "happy vs sad": ("mood: happy, bright", "mood: sad, emotional"),
          "electric vs acoustic": ("timbre: electric, distorted", "timbre: acoustic"),
          "piano vs guitar": ("instrument: piano, keys", "instrument: guitar")}
REUSED = ("tempo", "loudness", "quality")
# chosen 11-15 23:15 from dsp_validity_all.csv and dsp_extra.csv (sample A) before any edit score; no chroma or genre-like
# features, and no stereo width (generated clips are near-mono, side -73 dB, so it would read every clip as low quality)
SINGLE = {"tempo": "onset_rate_mix", "loudness": "rms_db", "quality": "energy_band_400_1000hz_ratio"}
INTERPRETABLE = {"tempo": ["onset_rate_mix", "onset_rate_percussive", "onset_strength_mean", "percussive_share"],
                 "loudness": ["rms_db", "energy_rms_db_mean", "crest_factor_db", "onset_strength_mean", "centroid_hz"],
                 "quality": ["energy_band_400_1000hz_ratio", "energy_band_60_150hz_ratio", "share_above_12khz", "noise_floor_db",
                             "flatness"]}
SKIP = {"check", "ytid", "positive", "native_rate", "onset_rate_drums", "tempo_drums_no_prior", "drum_share_db", "stereo_side_db"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-feature detectors on real MusicCaps audio (probe training pool), and chroma entropy as a harmony "
        "detector. Per check, three readers: the single chosen feature, a logistic regression on a few interpretable features, and "
        "one on every HW1 + extra feature (genre-like features included, for comparison). Sample A = the clips of "
        "dsp_validity.py (5-fold cross-validated AUC); sample B = fresh clips from the same pool, not in A, scored by the readers "
        "fitted on A (the honest number, free of feature selection). Four content checks (jazz/country, happy/sad, "
        "electric/acoustic, piano/guitar) get new clips, cross-validated only. Run with the hw1 venv and -P. Writes "
        "results/analysis/dsp_detectors.csv, dsp_detectors.json (fitted weights, applied by edit_effects.py) and "
        "dsp_detectors_clips.csv.")
    parser.add_argument("--fresh-per-group", type=int, default=100)
    parser.add_argument("--content-per-group", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def clip_features(ytid: str) -> dict:
    audio, rate = sf.read(HW3 / f"data/musiccaps/audio_mp3/{ytid}.mp3", dtype="float32", always_2d=True)
    stereo = audio.T
    return {**extract(librosa.resample(stereo.mean(0), orig_sr=rate, target_sr=SR)),
            **extra_features(stereo, rate, None, "cpu", False)}


def reader(columns: list):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    return model, columns


def cv_auc(model, x: np.ndarray, y: np.ndarray) -> float:
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    return roc_auc_score(y, cross_val_predict(model, x, y, cv=folds, method="decision_function"))


def export(model, columns: list) -> dict:
    imputer, scaler, regression = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    return {"features": columns, "fill": imputer.statistics_.tolist(), "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": regression.coef_[0].tolist(), "intercept": float(regression.intercept_[0])}


def draw(train, y, positive, negative, per_group, exclude, seed):
    picked = []
    for group, other, label in ((positive, negative, 1), (negative, positive, 0)):
        pool = train[(y[group] & ~y[other]).to_numpy() & ~train["ytid"].isin(exclude).to_numpy()]
        pool = pool[pool["ytid"].map(lambda t: (HW3 / f"data/musiccaps/audio_mp3/{t}.mp3").exists())]
        picked += [(t, label) for t in pool.sample(min(per_group, len(pool)), random_state=seed)["ytid"]]
    return picked


def main():
    args = parse_args()
    out = HW3 / "results/analysis"
    train, _ = training_set()
    y = labels(train["caption"])
    old = pd.read_csv(out / "dsp_validity_clips.csv").merge(
        pd.read_csv(out / "dsp_extra_clips.csv").drop(columns="positive"), on=["check", "ytid"])
    jobs = []
    for check, (positive, negative) in CHECKS.items():
        if check in REUSED:
            used = set(old.loc[old["check"] == check, "ytid"])
            jobs += [(check, "B", t, label) for t, label in draw(train, y, positive, negative, args.fresh_per_group, used, 1)]
        else:
            jobs += [(check, "A", t, label) for t, label in draw(train, y, positive, negative, args.content_per_group, set(), 0)]
    print(f"{len(jobs)} new clips, {args.jobs} workers", flush=True)
    start = time.time()
    values = Parallel(n_jobs=args.jobs, verbose=5)(delayed(clip_features)(t) for _, _, t, _ in jobs)
    print(f"features {time.time() - start:.0f} s", flush=True)
    new = pd.DataFrame([{"check": c, "sample": s, "ytid": t, "positive": label, **v} for (c, s, t, label), v in zip(jobs, values)])
    new.to_csv(out / "dsp_detectors_clips.csv", index=False)
    old = old.assign(sample="A")
    table = pd.concat([old[old["check"].isin(REUSED)], new], ignore_index=True)
    every = [c for c in new.columns if c not in SKIP | {"sample"} and pd.api.types.is_numeric_dtype(new[c])]
    rows, weights = [], {}
    for check in CHECKS:
        a = table[(table["check"] == check) & (table["sample"] == "A")]
        b = table[(table["check"] == check) & (table["sample"] == "B")]
        readers = {"chroma entropy": [CHROMA], "all features": every}
        if check in SINGLE:
            readers = {"single": [SINGLE[check]], "interpretable": INTERPRETABLE[check], **readers}
        for name, columns in readers.items():
            model, _ = reader(columns)
            target = a["positive"].to_numpy()
            row = {"check": check, "reader": name, "features": len(columns), "n_positive_a": int(target.sum()),
                   "n_negative_a": int((1 - target).sum()), "auc_a_cv": cv_auc(model, a[columns].to_numpy(float), target)}
            if len(columns) == 1:
                # pooled out-of-fold scores from separate fits are biased low on weak single features; no fit is needed here
                keep = a[columns[0]].notna().to_numpy()
                row["auc_a_raw"] = roc_auc_score(target[keep], a[columns[0]].to_numpy(float)[keep])
            model.fit(a[columns].to_numpy(float), target)
            weights[f"{check}|{name}"] = {"positive_group": CHECKS[check][0], "negative_group": CHECKS[check][1],
                                          **export(model, columns)}
            if len(b) and b["positive"].nunique() == 2 and min(b["positive"].sum(), (1 - b["positive"]).sum()) >= 20:
                row.update({"n_positive_b": int(b["positive"].sum()), "n_negative_b": int((1 - b["positive"]).sum()),
                            "auc_b_fresh": roc_auc_score(b["positive"], model.decision_function(b[columns].to_numpy(float)))})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dsp_detectors.csv", index=False)
    (out / "dsp_detectors.json").write_text(json.dumps(weights, indent=1) + "\n")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
