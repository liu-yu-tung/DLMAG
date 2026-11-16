import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.attribute_probes import fit_probes  # noqa: E402

CACHE = HW3 / "outputs/cache/runs"
OUT = HW3 / "results/analysis"
EMBEDDINGS = ("mert", "clap90")
# axis: (feature, sign); voice uses the vocal-stem pitch from scripts/analysis/edit_features.py
DSP = {"soft_to_loud": ("energy_rms_db_mean", 1), "fast_to_slow": ("rhythm_tempo_bpm", -1),
       "low_quality_to_studio": ("energy_band_6000_12000hz_ratio", 1), "voice_male_to_female": ("vocal_f0_median_hz", 1)}
# axis: (scripts/analysis/dsp_validity.py check, +1 if the edit target is that check's positive group); used for the "dsp best" reading
DSP_CHECK = {"fast_to_slow": ("tempo", -1), "low_quality_to_studio": ("quality", 1), "soft_to_loud": ("loudness", 1),
             "voice_male_to_female": ("voice pitch", 1)}
# energy-gated vocal-stem pitch passed on real audio (voice_pitch_check.csv, AUC 0.74); addendum 4
STEM = {"voice_male_to_female": ("pitch_f0_median_energy_gated", 1)}
JAZZ, COUNTRY = "insert_jazz", "insert_country"
PAIRED = "insert_jazz_vs_country"
# axis: (scripts/analysis/dsp_detectors.py check, +1 if the edit target is that check's positive group); readers fitted on real audio only
READER_CHECK = {"fast_to_slow": ("tempo", -1), "soft_to_loud": ("loudness", 1), "low_quality_to_studio": ("quality", 1),
                "happy_to_sad": ("happy vs sad", -1), "acoustic_to_electric": ("electric vs acoustic", 1),
                "guitar_to_piano": ("piano vs guitar", 1), JAZZ: ("jazz vs country", 1), COUNTRY: ("jazz vs country", -1),
                PAIRED: ("jazz vs country", 1)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the word-swap test: per axis and detector, how many edited-minus-original pairs moved the expected way "
        "(binomial p vs 6 of 12, Holm over 8 axes), the mean change (sign-flip p, bootstrap interval), the change as a multiple "
        "of seed noise (original caption, seeds 1-2 vs 0), and probe specificity. Checks docs/edit_predictions.toml on the 12 "
        "pre-registered pairs; 'pooled' adds the extended pairs when their embeddings exist. Run with the av-benchmark venv. "
        "Writes results/analysis/edit_effects.csv, edit_pairs.csv and edit_effects.json.")
    parser.add_argument("--base", default="rescale12")
    parser.add_argument("--flips", type=int, default=10000)
    parser.add_argument("--resamples", type=int, default=1000)
    return parser.parse_args()


def load_run(run: str, kind: str):
    path = CACHE / f"dev__{run}_{kind}.npy"
    if not path.exists():
        return None
    ids = [int(i) for i in path.with_suffix(".ids.txt").read_text().split()]
    return pd.DataFrame(np.load(path), index=ids)


def probe_scores(probes: dict, run: str, kind: str):
    x = load_run(run, kind)
    if x is None:
        return None
    return pd.DataFrame({group: model.decision_function(x.to_numpy()) for group, model in probes.items()}, index=x.index)


def apply_reader(weights: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[weights["features"]].to_numpy(float)
    x = np.where(np.isfinite(x), x, np.array(weights["fill"]))
    return ((x - np.array(weights["mean"])) / np.array(weights["scale"])) @ np.array(weights["coef"]) + weights["intercept"]


def version_table(features: pd.DataFrame, axis: str, feature: str, weights) -> pd.DataFrame:
    if axis == PAIRED:
        jazz, country = (version_table(features, a, feature, weights) for a in (JAZZ, COUNTRY))
        return jazz.assign(before=country["edited"].reindex(jazz.index))
    part = features[features["axis"] == axis].drop_duplicates(["id", "version"]).set_index(["id", "version"])
    values = pd.Series(apply_reader(weights, part) if weights else part[feature].to_numpy(float), index=part.index)
    table = values.unstack("version")
    return table.assign(before=table["original"])


def summarize(delta: np.ndarray, noise: np.ndarray, args) -> dict:
    n = len(delta)
    rng = np.random.default_rng(0)
    moved = int((delta > 0).sum())
    flips = rng.choice([-1.0, 1.0], size=(args.flips, n))
    means = delta[rng.integers(0, n, size=(args.resamples, n))].mean(1)
    noise_sd = float(np.std(noise, ddof=1)) if len(noise) > 1 else np.nan
    return {"n": n, "moved": moved, "p_binomial": float(binom.sf(moved - 1, n, 0.5)), "mean": float(delta.mean()),
            "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "p_signflip": float((((flips * delta).mean(1) >= delta.mean()).sum() + 1) / (args.flips + 1)), "noise_sd": noise_sd,
            "effect_vs_noise": float(delta.mean() / noise_sd) if noise_sd else np.nan,
            "edit_abs_vs_seed_abs": float(np.abs(delta).mean() / np.abs(noise).mean()) if len(noise) else np.nan}


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adjusted = np.maximum.accumulate([min(1.0, value * (len(p) - rank)) for rank, value in enumerate(p[order])])
    return pd.Series(adjusted, index=order).reindex(p.index)


def main():
    args = parse_args()
    manifest = json.loads((OUT / "edits_manifest.json").read_text())
    subsets = {"registered": {axis: entry["ids"] for axis, entry in manifest.items()},
               "pooled": {axis: entry["ids"] + entry.get("ext_ids", []) for axis, entry in manifest.items()}}
    category = lambda group: group.split(":")[0]  # noqa: E731
    rows, pair_rows = [], []

    def record(axis, detector, subset, ids, delta, noise, extra=None):
        keep = np.isfinite(delta)
        rows.append({"axis": axis, "detector": detector, "subset": subset, **summarize(delta[keep], noise[np.isfinite(noise)], args),
                     **(extra or {})})
        pair_rows.extend({"axis": axis, "detector": detector, "subset": subset, "id": i, "delta": d} for i, d in zip(ids, delta))

    for kind in EMBEDDINGS:
        probes = fit_probes(kind)
        base = probe_scores(probes, args.base, kind)
        seeds = [probe_scores(probes, f"{args.base}_seed{k}", kind) for k in (1, 2)]
        edits = {}
        for axis in manifest:
            parts = [probe_scores(probes, f"edit_{axis}{suffix}", kind) for suffix in ("", "_ext")]
            edits[axis] = pd.concat([p for p in parts if p is not None]) if any(p is not None for p in parts) else None
        swap_axes = [a for a in manifest if manifest[a]["source"]]
        targets = {a: (manifest[a]["source"], manifest[a]["target"]) for a in swap_axes}
        jazz, country = manifest[JAZZ]["target"], manifest[COUNTRY]["target"]
        targets.update({JAZZ: (country, jazz), COUNTRY: (jazz, country), PAIRED: (country, jazz)})
        for axis, (source, target) in targets.items():
            unrelated = [g for g in probes if category(g) not in {category(source), category(target)}]
            for subset, axis_ids in subsets.items():
                ids = axis_ids[JAZZ if axis == PAIRED else axis]
                if axis == PAIRED:
                    after, before = edits[JAZZ], edits[COUNTRY]
                else:
                    after, before = edits[axis], base
                if after is None or not set(ids) <= set(after.index):
                    continue
                if subset == "pooled" and len(ids) == len(subsets["registered"][JAZZ if axis == PAIRED else axis]):
                    continue
                change = after.loc[ids] - before.loc[ids]
                delta = (change[target] - change[source]).to_numpy()
                noise = np.concatenate([((s.loc[ids, target] - s.loc[ids, source]) - (base.loc[ids, target] - base.loc[ids, source])).to_numpy()
                                        for s in seeds if s is not None and set(ids) <= set(s.index)])
                unrelated_seed = float(np.mean([(s.loc[ids, unrelated] - base.loc[ids, unrelated]).abs().mean().mean()
                                                for s in seeds if s is not None and set(ids) <= set(s.index)]))
                record(axis, f"probe_{kind}", subset, ids, delta, noise,
                       {"target_rise": float(change[target].mean()), "source_drop": float(-change[source].mean()),
                        "unrelated_mean_abs": float(change[unrelated].abs().mean().mean()),
                        # same statistic on both sides (side-agent note 23:30): mean |change| under the edit over mean |change| between
                        # seeds of the original caption, for the target-minus-source probe and for the unrelated probes
                        "unrelated_seed_abs": unrelated_seed, "unrelated_edit_vs_seed": float(change[unrelated].abs().mean().mean()) / unrelated_seed})

    features_path = OUT / "edit_features.csv"
    if features_path.exists():
        features = pd.read_csv(features_path)
        if (OUT / "edit_chroma.csv").exists():
            features = features.merge(pd.read_csv(OUT / "edit_chroma.csv"), on=["axis", "id", "subset", "version"], how="left")
        readings = [(axis, feature, sign, "dsp", None) for axis, (feature, sign) in DSP.items()]
        readings += [(axis, feature, sign, "dsp stem", None) for axis, (feature, sign) in STEM.items() if feature in features.columns]
        validity_path = OUT / "dsp_validity_all.csv"
        if validity_path.exists():
            validity = pd.read_csv(validity_path)
            for axis, (check, direction) in DSP_CHECK.items():
                best = validity[(validity["check"] == check) & validity["feature"].isin(features.columns)].iloc[0]
                readings.append((axis, best["feature"], direction * (1 if best["auc"] > 0.5 else -1), "dsp best", None))
        detectors_path = OUT / "dsp_detectors.json"
        if detectors_path.exists():
            detectors = json.loads(detectors_path.read_text())
            for axis, (check, direction) in READER_CHECK.items():
                for key, weights in detectors.items():
                    if key.split("|")[0] == check and set(weights["features"]) <= set(features.columns):
                        readings.append((axis, key.split("|")[1], direction, "reader", weights))
        chroma_path = OUT / "chroma_check.csv"
        if chroma_path.exists():
            chroma = pd.read_csv(chroma_path)
            for axis, (check, direction) in READER_CHECK.items():
                for c in chroma[chroma["check"] == check].itertuples():
                    if c.feature in features.columns:
                        readings.append((axis, c.feature, direction * (1 if c.auc > 0.5 else -1), "chroma", None))
        for axis, feature, sign, label, weights in readings:
            table = version_table(features, axis, feature, weights)
            for subset, axis_ids in subsets.items():
                key_axis = JAZZ if axis == PAIRED else axis
                ids = [i for i in axis_ids[key_axis] if i in table.index]
                if subset == "pooled" and len(ids) == len(subsets["registered"][key_axis]):
                    continue
                t = table.loc[ids]
                delta = sign * (t["edited"] - t["before"]).to_numpy()
                noise = sign * np.concatenate([(t[k] - t["original"]).to_numpy() for k in ("seed1", "seed2")])
                record(axis, f"{label}: {feature}", subset, ids, delta, noise)

    table = pd.DataFrame(rows)
    predictions = tomllib.loads((HW3 / "docs/edit_predictions.toml").read_text())["prediction"]
    primary = {p["axis"]: ("probe_mert" if p["detector"] == "probe" else p["detector"]) for p in predictions}
    is_primary = table.apply(lambda r: r["subset"] == "registered" and primary.get(r["axis"]) == r["detector"], axis=1)
    table["p_binomial_holm"] = np.nan
    table.loc[is_primary, "p_binomial_holm"] = holm(table.loc[is_primary, "p_binomial"])
    verdicts = []
    for p in predictions:
        hit = table[is_primary & (table["axis"] == p["axis"])]
        if hit.empty:
            verdicts.append({"axis": p["axis"], "detector": p["detector"], "predicted": [p["low"], p["high"]], "result": None})
            continue
        r = hit.iloc[0]
        verdicts.append({"axis": p["axis"], "detector": p["detector"], "predicted": [p["low"], p["high"]], "moved": int(r["moved"]),
                         "n": int(r["n"]), "met": bool(p["low"] <= r["moved"] <= p["high"]), "p_binomial": r["p_binomial"],
                         "p_binomial_holm": r["p_binomial_holm"], "p_signflip": r["p_signflip"], "effect_vs_noise": r["effect_vs_noise"]})
    second = []
    for s in tomllib.loads((HW3 / "docs/edit_predictions.toml").read_text()).get("second_reading", []):
        hit = table[(table["subset"] == "registered") & (table["axis"] == s["axis"]) & (table["detector"] == s["detector"])]
        if hit.empty:
            second.append({"axis": s["axis"], "detector": s["detector"], "predicted": [s["low"], s["high"]], "result": None})
            continue
        r = hit.iloc[0]
        second.append({"axis": s["axis"], "detector": s["detector"], "predicted": [s["low"], s["high"]], "moved": int(r["moved"]),
                       "n": int(r["n"]), "met": bool(s["low"] <= r["moved"] <= s["high"]), "p_binomial": r["p_binomial"],
                       "p_signflip": r["p_signflip"], "effect_vs_noise": r["effect_vs_noise"]})
    out = OUT
    table.to_csv(out / "edit_effects.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "edit_pairs.csv", index=False)
    (out / "edit_effects.json").write_text(json.dumps({"verdicts": verdicts, "second_readings": second}, indent=1, default=float) + "\n")
    pd.set_option("display.width", 220)
    print(table.drop(columns=["ci_low", "ci_high"]).round(3).to_string(index=False))
    for v in verdicts + second:
        print(v)


if __name__ == "__main__":
    main()
