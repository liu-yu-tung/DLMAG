import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import fit_probes, labels, probe_model, training_set  # noqa: E402
from analysis.edit_effects import probe_scores  # noqa: E402

# side-agent note 11-16 00:31 (user: estimate, not remove): the drums probe learned from captions where "drums" sits next to "loud", so it may score loudness;
# caption-free features (HPSS percussive share, percussive onset rate) estimate how much of "loud brings drums" is the probe's
DRUMS = "instrument: drums, percussion"
LEVEL, PERCUSSION = "rms_db", ["percussive_share", "onset_rate_percussive"]
AXES = ["soft_to_loud", "happy_to_sad", "voice_male_to_female"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="How much does the drums probe react to level rather than drums? 1) On real MusicCaps clips (dsp_extra_clips.csv, "
        "training pool), regress the cross-validated drums probe score on level (RMS dB) and caption-free percussion features (HPSS "
        "percussive share, percussive onset rate), all standardized. 2) On the word-swap pairs, the observed drums-probe change vs "
        "the part predicted from the level change alone, and the change of the caption-free percussion features, in seed-spread "
        "units. Run with the av-benchmark venv. Writes results/analysis/probe_leakage.json.")
    return parser.parse_args()


def zscore(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return (frame - reference.mean()) / reference.std()


def main():
    parse_args()
    out = HW3 / "results/analysis"
    train, _ = training_set()
    ids = open(HW3 / "outputs/cache/musiccaps_mert_ids.txt").read().split()
    features = np.load(HW3 / "outputs/cache/musiccaps_mert.npy")
    index = {clip: i for i, clip in enumerate(ids)}
    train = train[train["ytid"].map(index).notna().to_numpy()].reset_index(drop=True)
    x = features[train["ytid"].map(index).astype(int).to_numpy()]
    y = labels(train["caption"])[DRUMS].to_numpy()
    score = pd.Series(cross_val_predict(probe_model(), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                        method="decision_function"), index=train["ytid"])
    real = pd.read_csv(out / "dsp_extra_clips.csv").drop_duplicates("ytid")
    real = real[real["ytid"].isin(score.index)]
    real["drums_score"] = score.loc[real["ytid"]].to_numpy()
    columns = [LEVEL] + PERCUSSION
    design = zscore(real[columns], real[columns])
    target = (real["drums_score"] - real["drums_score"].mean()) / real["drums_score"].std()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(design)), design.to_numpy()]), target.to_numpy(), rcond=None)
    alone, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(design)), design[[LEVEL]].to_numpy()]), target.to_numpy(), rcond=None)
    summary = {"real_clips": len(real), "drums_score_on_level_alone": float(alone[1]),
               "drums_score_on_level_given_percussion": float(coef[1]),
               "drums_score_on_percussion": {c: float(b) for c, b in zip(PERCUSSION, coef[2:])},
               "corr_level_percussive_share": float(real[LEVEL].corr(real["percussive_share"])), "axes": {}}
    probes = fit_probes("mert", [DRUMS])
    base = probe_scores(probes, "rescale12", "mert")
    seeds = [probe_scores(probes, f"rescale12_seed{k}", "mert") for k in (1, 2)]
    manifest = json.loads((out / "edits_manifest.json").read_text())
    edit_features = pd.read_csv(out / "edit_features.csv")
    real_sd = real["drums_score"].std()
    for axis in AXES:
        edited = pd.concat([p for p in (probe_scores(probes, f"edit_{axis}{s}", "mert") for s in ("", "_ext")) if p is not None])
        pair_ids = [i for i in manifest[axis]["ids"] + manifest[axis].get("ext_ids", []) if i in edited.index]
        table = edit_features[edit_features["axis"] == axis].drop_duplicates(["id", "version"]).set_index(["id", "version"])
        def change(column, version):
            values = table[column].unstack("version")
            return (values[version] - values["original"]).loc[pair_ids]
        drums_change = (edited.loc[pair_ids, DRUMS] - base.loc[pair_ids, DRUMS])
        drums_noise = pd.concat([s.loc[pair_ids, DRUMS] - base.loc[pair_ids, DRUMS] for s in seeds]).std(ddof=1)
        level_change_z = change(LEVEL, "edited") / real[LEVEL].std()
        predicted = alone[1] * level_change_z * real_sd
        entry = {"pairs": len(pair_ids), "drums_probe_change_sd": float(drums_change.mean() / drums_noise),
                 "drums_change_predicted_from_level_alone_sd": float(predicted.mean() / drums_noise),
                 "drums_change_predicted_from_level_given_percussion_sd": float((coef[1] * level_change_z * real_sd).mean() / drums_noise),
                 "level_change_db": float(change(LEVEL, "edited").mean())}
        for column in PERCUSSION:
            noise = pd.concat([change(column, v) for v in ("seed1", "seed2")]).std(ddof=1)
            delta = change(column, "edited")
            entry[f"{column}_change_sd"] = float(delta.mean() / noise)
            entry[f"{column}_share_up"] = float((delta > 0).mean())
        summary["axes"][axis] = entry
    (out / "probe_leakage.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
