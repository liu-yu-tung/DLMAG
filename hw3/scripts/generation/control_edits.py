import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))

# side-agent note 11-15 23:47: any prompt change on the same seed may re-roll the clip (Prompt-to-Prompt), so a word swap's
# change must be compared with a swap that carries no musical meaning; both words are in caption_vocab.GENERIC
SYNONYM = ("song", "track")
NEUTRAL = " The clip is ten seconds long."
CONTROLS = ("control_song_to_track", "control_neutral_insert")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Meaning-free edit controls for the word-swap test. build: control_song_to_track (24 dev captions with the word "
        "'song', first one swapped to 'track') and control_neutral_insert (the 24 genre-insert captions plus a sentence with no "
        "musical content) -> data/edits/<control>.csv and results/analysis/edits_control_manifest.json. score: per control and per "
        "real axis (pooled pairs), the mean |change| of each valid probe group under the edit over the same between seeds, and the "
        "number of groups whose signed mean change has a 95% interval excluding 0. Run with the av-benchmark venv. Writes "
        "results/analysis/edit_controls.csv.")
    parser.add_argument("stage", choices=("build", "score"))
    parser.add_argument("--kind", default="mert", choices=("mert", "clap90"))
    return parser.parse_args()


def build():
    dev = pd.read_csv(HW3 / "data/dev.csv")
    has = dev[dev["caption"].str.contains(rf"\b{SYNONYM[0]}\b", case=False)]
    chosen = has.sample(min(24, len(has)), random_state=0).sort_values("id")
    swapped = chosen.assign(caption=chosen["caption"].map(
        lambda c: re.sub(rf"\b{SYNONYM[0]}\b", SYNONYM[1], c, count=1, flags=re.I)))
    assert (swapped["caption"] != chosen["caption"]).all()
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    insert_ids = manifest["insert_jazz"]["ids"] + manifest["insert_jazz"]["ext_ids"]
    inserted = dev.set_index("id").loc[insert_ids].reset_index()
    inserted["caption"] = inserted["caption"].str.rstrip() + NEUTRAL
    out = HW3 / "data/edits"
    swapped[["id", "caption"]].to_csv(out / f"{CONTROLS[0]}.csv", index=False)
    inserted[["id", "caption"]].to_csv(out / f"{CONTROLS[1]}.csv", index=False)
    record = {CONTROLS[0]: {"ids": [int(i) for i in swapped["id"]], "swap": list(SYNONYM)},
              CONTROLS[1]: {"ids": [int(i) for i in inserted["id"]], "sentence": NEUTRAL.strip()}}
    (HW3 / "results/analysis/edits_control_manifest.json").write_text(json.dumps(record, indent=1) + "\n")
    print({k: len(v["ids"]) for k, v in record.items()})


def score(kind: str):
    from analysis.attribute_probes import fit_probes
    from analysis.edit_effects import probe_scores
    from selection.probe_rank import valid_groups
    groups = valid_groups(0.75)
    probes = fit_probes(kind, groups)
    base = probe_scores(probes, "rescale12", kind)
    seeds = [probe_scores(probes, f"rescale12_seed{k}", kind) for k in (1, 2)]
    controls = json.loads((HW3 / "results/analysis/edits_control_manifest.json").read_text())
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    runs = {name: ([f"edit_{name}"], entry["ids"]) for name, entry in controls.items()}
    runs.update({axis: ([f"edit_{axis}", f"edit_{axis}_ext"], e["ids"] + e.get("ext_ids", []))
                 for axis, e in manifest.items() if (HW3 / f"outputs/dev/edit_{axis}").is_dir()})
    rng = np.random.default_rng(0)
    rows = []
    for name, (folders, ids) in runs.items():
        parts = [probe_scores(probes, f, kind) for f in folders]
        edited = pd.concat([p for p in parts if p is not None])
        ids = [i for i in ids if i in edited.index]
        change = edited.loc[ids] - base.loc[ids]
        noise = pd.concat([s.loc[ids] - base.loc[ids] for s in seeds])
        clear = 0
        for group in groups:
            delta = change[group].to_numpy()
            means = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
            clear += int(np.quantile(means, 0.025) > 0 or np.quantile(means, 0.975) < 0)
        rows.append({"edit": name, "control": name in controls, "kind": kind, "n": len(ids),
                     "profile_change_vs_seed": float(change.abs().mean().mean() / noise.abs().mean().mean()),
                     "groups_clear_of_zero": clear, "groups": len(groups)})
    table = pd.DataFrame(rows)
    path = HW3 / "results/analysis/edit_controls.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["kind"] != kind]
    pd.concat([old, table]).to_csv(path, index=False)
    print(table.round(3).to_string(index=False))


def main():
    args = parse_args()
    build() if args.stage == "build" else score(args.kind)


if __name__ == "__main__":
    main()
