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
from analysis.attribute_probes import fit_probes, labels  # noqa: E402
from analysis.caption_vocab import LEXICON  # noqa: E402
from selection.probe_rank import valid_groups  # noqa: E402

RUNS = ("rescale12", "rescale12_seed1", "rescale12_seed2")
FIXED = {"recording: live, crowd": "It is clearly a live recording.", "vocals: no vocals": "It is instrumental, with no vocals."}
TEMPLATE = {"instrument": "The {} can be clearly heard.", "vocals": "The {} can be clearly heard.", "other sounds": "The {} can be clearly heard.",
            "mood": "The music is clearly {}.", "dynamics": "The music is clearly {}.", "timbre": "The sound is clearly {}.",
            "genre": "It is clearly {} music."}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe-in-the-loop round 2 (plan, user 11-15 20:08). build: for each caption, the asked word groups (probe_rank.py "
        "valid groups) that the round-1 pick (rescale12_best3_probe) does not show (MERT probe log-odds < 0); appends one emphasis "
        "sentence per missing group, using the caption's own matched words, and writes data/loop/<set>_s<k>.csv per picked seed "
        "index. select: keeps the regenerated clip only if its probe score (mean log-odds over all asked valid groups, as round 1) "
        "rises, and assembles outputs/<set>/rescale12_loop2. Scoring uses the original captions. Run with the av-benchmark venv.")
    parser.add_argument("stage", choices=("build", "select"))
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--round1", default="rescale12_best3_probe")
    parser.add_argument("--name", default="rescale12_loop2")
    parser.add_argument("--min-auc", type=float, default=0.75)
    return parser.parse_args()


def emphasis(group: str, caption: str) -> str:
    if group in FIXED:
        return FIXED[group]
    patterns = next(p for c, g, _, p in LEXICON if f"{c}: {g}" == group)
    match = re.search(r"\b(?:" + "|".join(patterns) + r")\b", caption.lower())
    return TEMPLATE[group.split(":")[0]].format(match.group(0))


def log_odds(probes: dict, groups: list, set_name: str, run: str) -> pd.DataFrame:
    path = HW3 / f"outputs/cache/runs/{set_name}__{run}_mert.npy"
    ids = [int(i) for i in path.with_suffix(".ids.txt").read_text().split()]
    x = np.load(path)
    return pd.DataFrame({g: probes[g].decision_function(x) for g in groups}, index=ids)


def score(odds: pd.DataFrame, asked: pd.DataFrame) -> pd.Series:
    mask = asked.loc[odds.index].to_numpy()
    values = np.where(mask.any(1), (odds.to_numpy() * mask).sum(1) / np.maximum(mask.sum(1), 1), 0.0)
    return pd.Series(values, index=odds.index)


def main():
    args = parse_args()
    groups = valid_groups(args.min_auc)
    captions = pd.read_csv(HW3 / f"data/{args.set_name}.csv")
    captions["id"] = captions["id"].astype(int)
    captions = captions.set_index("id")
    asked = labels(captions["caption"]).set_index(captions.index)[groups]
    probes = fit_probes("mert", groups)
    chosen = pd.read_csv(HW3 / f"results/{args.set_name}__{args.round1}/selection.csv").set_index("id")["chosen_run"]
    seed_index = {run: k for k, run in enumerate(RUNS)}
    record = HW3 / f"results/analysis/loop_{args.set_name}.json"
    if args.stage == "build":
        odds = {run: log_odds(probes, groups, args.set_name, run) for run in RUNS}
        out = HW3 / "data/loop"
        out.mkdir(parents=True, exist_ok=True)
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="tokenizer")
        rows, missing_counts, cut_counts = [], {}, 0
        for item_id, run in chosen.items():
            missing = [g for g in groups if asked.loc[item_id, g] and odds[run].loc[item_id, g] < 0]
            text = captions.loc[item_id, "caption"].strip()
            kept = []
            for g in missing:
                # T5 drops tokens past its limit silently, so an emphasis sentence is added only if it fits (side-agent note 00:42)
                if len(tokenizer(" ".join([text] + [emphasis(k, text) for k in kept + [g]])).input_ids) <= tokenizer.model_max_length:
                    kept.append(g)
            cut_counts += len(missing) - len(kept)
            if kept:
                rows.append({"id": item_id, "seed_index": seed_index[run],
                             "caption": " ".join([text] + [emphasis(g, text) for g in kept])})
                for g in kept:
                    missing_counts[g] = missing_counts.get(g, 0) + 1
        table = pd.DataFrame(rows)
        for k in range(len(RUNS)):
            part = table[table["seed_index"] == k][["id", "caption"]]
            if len(part):
                part.to_csv(out / f"{args.set_name}_s{k}.csv", index=False)
        info = {"set": args.set_name, "round1": args.round1, "groups": len(groups), "changed": len(table), "emphasis_not_added_token_limit": cut_counts,
                "per_seed_index": {str(k): int((table["seed_index"] == k).sum()) for k in range(len(RUNS))},
                "missing_per_group": dict(sorted(missing_counts.items(), key=lambda kv: -kv[1]))}
        if args.set_name == "dev":
            info["ids"] = [int(i) for i in table["id"]]
        record.write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps({k: v for k, v in info.items() if k != "ids"}, indent=1))
        return
    info = json.loads(record.read_text())
    target = HW3 / f"outputs/{args.set_name}/{args.name}"
    target.mkdir(parents=True, exist_ok=True)
    regenerated = {}
    for k, count in info["per_seed_index"].items():
        if count:
            odds = log_odds(probes, groups, args.set_name, f"{args.name}_s{k}")
            regenerated.update({i: (k, s) for i, s in score(odds, asked).items()})
    round1 = {run: score(log_odds(probes, groups, args.set_name, run), asked) for run in RUNS}
    rows = []
    for item_id, run in chosen.items():
        old = float(round1[run].loc[item_id])
        source = HW3 / f"outputs/{args.set_name}/{run}/{item_id}.wav"
        new, kept = np.nan, False
        if item_id in regenerated:
            k, new = regenerated[item_id]
            kept = bool(new > old)
            if kept:
                source = HW3 / f"outputs/{args.set_name}/{args.name}_s{k}/{item_id}.wav"
        shutil.copyfile(source, target / f"{item_id}.wav")
        rows.append({"id": item_id, "round1_run": run, "round1_score": old, "round2_score": new, "kept_round2": kept})
    table = pd.DataFrame(rows)
    results = HW3 / f"results/{args.set_name}__{args.name}"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "loop_selection.csv", index=False)
    tried = table["round2_score"].notna()
    print(f"regenerated {int(tried.sum())}, kept {int(table['kept_round2'].sum())}; mean probe score round 1 "
          f"{table['round1_score'].mean():+.3f}, after {np.where(table['kept_round2'], table['round2_score'], table['round1_score']).mean():+.3f}")


if __name__ == "__main__":
    main()
