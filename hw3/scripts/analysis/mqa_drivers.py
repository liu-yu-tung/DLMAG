import json
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
RUNS = ("dev__rescale12", "dev__rescale12_seed1", "dev__rescale12_seed2")


def tag_rows(run):
    rows = []
    for line in (HW3 / f"outputs/mqa/{run}/score.jsonl").read_text().splitlines():
        item = json.loads(line)
        for attribute, entries in item["per_attribute"].items():
            rows += [{"run": run, "id": item["id"], "attribute": attribute, "score": e["score"], "n_tags": item["n_tags"]} for e in entries]
    return rows


def main():
    tags = pd.DataFrame([row for run in RUNS for row in tag_rows(run)])
    item = tags.groupby(["run", "id"])["score"].mean().rename("mqascore")
    tags = tags.join(item, on=["run", "id"])
    # contribution of an attribute to the caption mean = its tags' summed score / n_tags
    contribution = tags.assign(part=tags["score"] / tags["n_tags"]).pivot_table(index=["run", "id"], columns="attribute", values="part",
                                                                             aggfunc="sum", fill_value=0)
    centred = contribution.sub(contribution.groupby(level="id").transform("mean"))
    total = item.sub(item.groupby(level="id").transform("mean"))
    rows = []
    for attribute, group in tags.groupby("attribute"):
        rows.append({"attribute": attribute, "tag_share": len(group) / len(tags), "mean_score": group["score"].mean(),
                     "share_yes_above_0.5": (group["score"] > 0.5).mean(),
                     "variance_share_all_clips": np.cov(contribution[attribute], item)[0, 1] / item.var(),
                     "variance_share_across_seeds": np.cov(centred[attribute], total)[0, 1] / total.var()})
    table = pd.DataFrame(rows).sort_values("tag_share", ascending=False).round(3)
    table.to_csv(HW3 / "results/analysis/mqa_drivers.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
