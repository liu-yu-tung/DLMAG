import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import HW3, RESULTS, boot_ci, fmt, per_prompt

VOCAL = re.compile(r"\b(vocals?|vocalists?|sing|sings|singer|singers|singing|voices?|choir|choirs|rap|raps|rapper|rappers|rapping|lyrics?|chant|chants|chanting|vocalization|humming|hums|whistl\w*|spoken|speech|speaks?|talking|narrat\w*|shouts?|screams?|yells?|backing)\b", re.I)
NEGATION = re.compile(r"\b(no|without|lacks?|absent|instrumental|devoid of|zero)\b[^.]{0,30}\b(vocals?|voices?|singing|singer|lyrics|vocalist)\b|\binstrumental\b", re.I)
GENDER = re.compile(r"\b(males?|females?|man|woman|men|women|boy|girl|child|children)\b", re.I)


def load(set_name):
    captions = pd.read_csv(HW3 / "data" / f"{set_name}.csv")
    tags = {int(json.loads(l)["id"]): json.loads(l)["tags"] for l in (HW3 / "outputs" / "mqa" / f"tags_{set_name}.jsonl").read_text().splitlines() if l.strip()}
    return captions, tags


for set_name in ("dev", "test"):
    captions, tags = load(set_name)
    mqa = pd.read_csv(RESULTS / f"{set_name}__cfg7" / "mqa_per_item.csv").set_index("id")
    tag_vocal = mqa["vocal_n"].fillna(0) > 0
    kw = captions["caption"].apply(lambda t: bool(VOCAL.search(t))).to_numpy()
    neg = captions["caption"].apply(lambda t: bool(NEGATION.search(t))).to_numpy()
    gender = captions["caption"].apply(lambda t: bool(GENDER.search(t))).to_numpy()
    tv = tag_vocal.reindex(captions["id"]).to_numpy()
    print(f"\n== {set_name}: tag-vocal {tv.sum()}, keyword-vocal {kw.sum()}, negation phrase {neg.sum()}, gender word {gender.sum()}")
    print(f"tag & kw {np.sum(tv & kw)}, tag only {np.sum(tv & ~kw)}, kw only {np.sum(~tv & kw)}, neither {np.sum(~tv & ~kw)}")
    print(f"negation among tag-vocal {np.sum(neg & tv)}, among kw-only {np.sum(neg & ~tv & kw)}, among neither {np.sum(neg & ~tv & ~kw)}")
    ids_tag_only = captions["id"][tv & ~kw].tolist()
    ids_kw_only = captions["id"][~tv & kw].tolist()
    ids_neg_tag = captions["id"][neg & tv].tolist()
    print("ids tag-only", ids_tag_only, "| ids kw-only", ids_kw_only, "| ids negation & tag-vocal", ids_neg_tag)
    if "--show" in sys.argv:
        for i in ids_kw_only + ids_neg_tag + ids_tag_only:
            vt = tags.get(i, {}).get("vocal")
            m = VOCAL.findall(captions.set_index("id").loc[i, "caption"])
            n = NEGATION.findall(captions.set_index("id").loc[i, "caption"])
            print(f"  id {i}: vocal tags {vt}; keyword hits {m}; negation {n}")
    if set_name == "dev":
        groups = pd.DataFrame({"id": captions["id"], "tag": tv, "kw": kw, "neg": neg}).set_index("id")
        groups.to_csv(Path(__file__).parent / "vocal_groups_dev.csv")
