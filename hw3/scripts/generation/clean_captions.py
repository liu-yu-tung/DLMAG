import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.caption_vocab import LEXICON, hits  # noqa: E402

# fixed 11-15 20:15 from the P1 checks: probes barely hear these (low quality 0.67, use or setting 0.63 on held-out real dev) and the
# 6-12 kHz quality feature fails (AUC 0.52); live/crowd stays (probe 0.79, audible content)
DROP = {"recording: low quality", "recording: high quality", "context: use or setting"}
MUSIC = {"instrument", "vocals", "genre", "mood", "tempo", "dynamics", "timbre"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Caption cleaning: drop each sentence whose lexicon hits are only recording-quality or use/setting groups (no "
        "instrument, vocal, genre, mood, tempo, dynamics or timbre word). build: writes data/clean/<set>.csv (changed ids only) and "
        "results/analysis/clean_<set>.json (counts, ids). fill: copies the unchanged clips of --base into outputs/<set>/<name>, which "
        "the same seed and settings would reproduce bit-identically (null-edit check, results/analysis/null_edit.json).")
    parser.add_argument("stage", choices=("build", "fill"))
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--base", default="rescale12")
    parser.add_argument("--name", default="rescale12_clean")
    return parser.parse_args()


def groups_in(sentence: str) -> set:
    text = pd.Series([sentence.lower()])
    return {f"{category}: {group}" for category, group, _, patterns in LEXICON if hits(text, patterns).iloc[0]}


def clean(caption: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    kept = []
    for sentence in sentences:
        found = groups_in(sentence)
        if found & DROP and not {g for g in found if g.split(":")[0] in MUSIC}:
            continue
        kept.append(sentence)
    return " ".join(kept) if kept else caption


def main():
    args = parse_args()
    captions = pd.read_csv(HW3 / f"data/{args.set_name}.csv", dtype={"id": str})
    record = HW3 / f"results/analysis/clean_{args.set_name}.json"
    if args.stage == "build":
        cleaned = captions["caption"].map(clean)
        changed = cleaned != captions["caption"]
        out = HW3 / "data/clean"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": captions["id"][changed], "caption": cleaned[changed]}).to_csv(out / f"{args.set_name}.csv", index=False)
        words = (captions["caption"].str.split().str.len(), cleaned.str.split().str.len())
        record.write_text(json.dumps({"set": args.set_name, "drop_groups": sorted(DROP), "changed": int(changed.sum()),
                                      "ids": [int(i) for i in captions["id"][changed]],
                                      "median_words_before": float(words[0].median()), "median_words_after": float(words[1].median())},
                                     indent=1) + "\n")
        print(f"{args.set_name}: {int(changed.sum())} captions changed")
    else:
        changed = set(str(i) for i in json.loads(record.read_text())["ids"])
        target = HW3 / f"outputs/{args.set_name}/{args.name}"
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for item_id in captions["id"]:
            if item_id not in changed:
                shutil.copyfile(HW3 / f"outputs/{args.set_name}/{args.base}/{item_id}.wav", target / f"{item_id}.wav")
                copied += 1
        assert len(list(target.glob("*.wav"))) == len(captions), "changed clips missing"
        print(f"copied {copied} unchanged clips; {len(changed)} regenerated")


if __name__ == "__main__":
    main()
