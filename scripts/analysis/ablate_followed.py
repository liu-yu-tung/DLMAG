import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from generation.clean_captions import groups_in  # noqa: E402

# positive control for caption cleaning (user 11-16 00:39): words the word-swap test showed the model follows
# (results/analysis/edit_effects.csv: guitar to piano 25/28, voice 21/26, acoustic to electric 14/17, MERT pooled)
FOLLOWED = {"instrument: guitar", "instrument: piano, keys", "vocals: male voice", "vocals: female voice", "timbre: acoustic",
            "timbre: electric, distorted"}
COUNT = 29


def parse_args():
    parser = argparse.ArgumentParser(
        description="Positive control for caption cleaning: drop the first sentence that names a followed group (guitar, piano, male "
        "or female voice, acoustic, electric) from 29 dev captions (the cleaning count; random_state 0), regenerate on seed 0. "
        "build: data/clean/dev_ablate.csv and results/analysis/ablate_dev.json; fill: copies the unchanged clips; score: on the "
        "changed ids, CLAP, KL and MQAScore against rescale12 (original captions), and the probe of each removed group in "
        "seed-spread units, next to the same numbers for caption cleaning. Run with the av-benchmark venv.")
    parser.add_argument("stage", choices=("build", "fill", "score"))
    parser.add_argument("--name", default="rescale12_ablate")
    return parser.parse_args()


def ablate(caption: str):
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    for k, sentence in enumerate(sentences):
        rest = " ".join(sentences[:k] + sentences[k + 1:])
        removed = (groups_in(sentence) & FOLLOWED) - groups_in(rest)
        if removed and len(sentences) > 1:
            return rest, sorted(removed)
    return None, []


def paired(a: pd.Series, b: pd.Series, ids, rng):
    d = (a.loc[ids] - b.loc[ids]).to_numpy(float)
    means = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "mean_abs": float(np.abs(d).mean())}


def main():
    args = parse_args()
    captions = pd.read_csv(HW3 / "data/dev.csv")
    record = HW3 / "results/analysis/ablate_dev.json"
    if args.stage == "build":
        rows = []
        for item_id, caption in zip(captions["id"], captions["caption"]):
            new, removed = ablate(caption)
            if new:
                rows.append({"id": int(item_id), "caption": new, "removed": removed})
        chosen = pd.DataFrame(rows).sample(COUNT, random_state=0).sort_values("id")
        out = HW3 / "data/clean"
        chosen[["id", "caption"]].to_csv(out / "dev_ablate.csv", index=False)
        record.write_text(json.dumps({"eligible": len(rows), "changed": COUNT, "ids": chosen["id"].tolist(),
                                      "removed": {str(i): r for i, r in zip(chosen["id"], chosen["removed"])}}, indent=1) + "\n")
        print(f"{len(rows)} eligible, {COUNT} chosen")
        return
    info = json.loads(record.read_text())
    changed = [int(i) for i in info["ids"]]
    if args.stage == "fill":
        target = HW3 / f"outputs/dev/{args.name}"
        for item_id in captions["id"]:
            if int(item_id) not in changed:
                shutil.copyfile(HW3 / f"outputs/dev/rescale12/{item_id}.wav", target / f"{item_id}.wav")
        assert len(list(target.glob("*.wav"))) == len(captions)
        return
    from analysis.attribute_probes import fit_probes
    from analysis.edit_effects import probe_scores
    rng = np.random.default_rng(0)
    clean_ids = [int(i) for i in json.loads((HW3 / "results/analysis/clean_dev.json").read_text())["ids"]]
    summary = {}
    for label, run, ids in (("ablate followed words", args.name, changed), ("clean recording words", "rescale12_clean", clean_ids)):
        entry = {"n": len(ids)}
        for file, column in (("per_prompt.csv", "clap_cosine"), ("per_prompt.csv", "kl_passt_softmax"), ("mqa_per_item.csv", "mqascore")):
            new, old = (HW3 / f"results/dev__{run}/{file}", HW3 / f"results/dev__rescale12/{file}")
            if new.exists():
                entry[column] = paired(pd.read_csv(new).set_index("id")[column], pd.read_csv(old).set_index("id")[column], ids, rng)
        summary[label] = entry
    seed_clap = [pd.read_csv(HW3 / f"results/dev__rescale12_seed{k}/per_prompt.csv").set_index("id")["clap_cosine"] for k in (1, 2)]
    base_clap = pd.read_csv(HW3 / "results/dev__rescale12/per_prompt.csv").set_index("id")["clap_cosine"]
    summary["seed change mean_abs clap on ablated ids"] = float(np.mean([np.abs(s.loc[changed] - base_clap.loc[changed]).mean() for s in seed_clap]))
    groups = sorted(FOLLOWED)
    probes = fit_probes("mert", groups)
    base = probe_scores(probes, "rescale12", "mert")
    seeds = [probe_scores(probes, f"rescale12_seed{k}", "mert") for k in (1, 2)]
    ablated = probe_scores(probes, args.name, "mert")
    drops = []
    for item_id in changed:
        for group in info["removed"][str(item_id)]:
            noise = pd.concat([s[group] - base[group] for s in seeds]).std(ddof=1)
            drops.append(float((ablated.loc[item_id, group] - base.loc[item_id, group]) / noise))
    drops = np.array(drops)
    means = drops[rng.integers(0, len(drops), (2000, len(drops)))].mean(1)
    summary["removed group probe change (seed SD)"] = {"n": len(drops), "mean": float(drops.mean()),
                                                       "ci_low": float(np.quantile(means, 0.025)),
                                                       "ci_high": float(np.quantile(means, 0.975)), "share_down": float((drops < 0).mean())}
    (HW3 / "results/analysis/ablate_dev_scores.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
