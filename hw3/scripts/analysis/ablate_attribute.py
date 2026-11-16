import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.caption_vocab import LEXICON  # noqa: E402
ATTRIBUTES = ("genre", "instrument", "mood_theme", "vocal", "context")
# which MQAScore attribute holds the removed lexicon group
REMOVED_TO_ATTRIBUTE = {"instrument": ("instrument:", "timbre:"), "vocal": ("vocals:",)}
rng = np.random.default_rng(0)


def interval(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    boot = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return {"n": len(x), "mean": x.mean(), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)}


def main():
    record = json.loads((HW3 / "results/analysis/ablate_dev.json").read_text())
    after = pd.read_csv(HW3 / "results/dev__rescale12_ablate/mqa_per_item.csv", dtype={"id": str}).set_index("id")
    before = pd.read_csv(HW3 / "results/dev__rescale12/mqa_per_item.csv", dtype={"id": str}).set_index("id")
    rows = []
    for hit, prefixes in REMOVED_TO_ATTRIBUTE.items():
        ids = [str(i) for i in record["ids"] if any(g.startswith(prefixes) for g in record["removed"][str(i)])]
        share = (before.loc[ids, f"{hit}_n"] / before.loc[ids, "n_tags"]).mean()
        rows.append({"removed": hit, "measure": "mqascore", "tag_share": share,
                     **interval(after.loc[ids, "mqascore"] - before.loc[ids, "mqascore"])})
        for attribute in ATTRIBUTES:
            rows.append({"removed": hit, "measure": f"{attribute}_mean", "tag_share": share,
                         **interval(after.loc[ids, f"{attribute}_mean"] - before.loc[ids, f"{attribute}_mean"])})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(HW3 / "results/analysis/ablate_mqa_attributes.csv", index=False)
    print(table.to_string(index=False))
    tags = per_tag(record).round(3)
    tags.to_csv(HW3 / "results/analysis/ablate_mqa_tags.csv", index=False)
    print(tags.to_string(index=False))


def tag_scores(run):
    scores = {}
    for line in (HW3 / f"outputs/mqa/{run}/score.jsonl").read_text().splitlines():
        item = json.loads(line)
        for attribute, entries in item["per_attribute"].items():
            for entry in entries:
                scores[(item["id"], attribute, entry["tag"])] = entry["score"]
    return scores


def per_tag(record):
    patterns = {f"{category}: {group}": re.compile(r"\b(?:" + "|".join(words) + r")\b") for category, group, _, words in LEXICON}
    after, before = tag_scores("dev__rescale12_ablate"), tag_scores("dev__rescale12")
    ids = {str(i) for i in record["ids"]}
    rows = []
    for key in before.keys() & after.keys():
        item_id, attribute, tag = key
        if item_id in ids and attribute in ("instrument", "vocal"):
            removed = [g for g in record["removed"][item_id] if patterns[g].search(tag.lower())]
            kind = ("vocal" if removed[0].startswith("vocals:") else "instrument") if removed else "kept"
            rows.append({"tag_kind": f"removed {kind}" if removed else f"kept {attribute}", "before": before[key], "change": after[key] - before[key]})
    frame = pd.DataFrame(rows)
    return pd.DataFrame([{"tag_kind": kind, "before_mean": group["before"].mean(), **interval(group["change"])}
                         for kind, group in frame.groupby("tag_kind")])


if __name__ == "__main__":
    main()
