import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.caption_vocab import LEXICON  # noqa: E402

# categories the word-swap test showed the model follows (edit_effects.csv, edit_spillover.csv); recording, mood, dynamics and
# context words are left out: ignored or rendered through bundles (predictions 11-16 01:02)
FOLLOWED = {"instrument", "vocals", "timbre", "genre"}
SKIP_GROUPS = {"vocals: no vocals"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Followed-word guidance. build: per caption, the caption's own instrument, vocal, timbre and genre phrases (lexicon "
        "matches, in caption order, de-duplicated) joined by commas -> data/followed/<set>_extra.csv (id,text) and "
        "data/followed/<set>_captions.csv (captions that have at least one phrase), record results/analysis/followed_<set>.json. "
        "fill: copies the rescale12 clip of captions without a phrase into outputs/<set>/<name> (same seed and settings, so "
        "bit-identical).")
    parser.add_argument("stage", choices=("build", "fill"))
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--name", default="rescale12_followed")
    parser.add_argument("--form", choices=("tags", "sentence"), default="tags",
                        help="sentence: adjacent matches merged into phrases, written as short sentences (side-agent note 01:09)")
    return parser.parse_args()


def matches(caption: str) -> list:
    lower = caption.lower()
    found = []
    for category, group, _, patterns in LEXICON:
        if category in FOLLOWED and f"{category}: {group}" not in SKIP_GROUPS:
            for match in re.finditer(r"\b(?:" + "|".join(patterns) + r")\b", lower):
                found.append((match.start(), match.end(), match.group(0), category))
    return sorted(found)


def sentence(caption: str) -> str:
    lower, merged = caption.lower(), []
    for start, end, text, category in matches(caption):
        if merged and start >= merged[-1][1] and re.fullmatch(r"[\s-]*", lower[merged[-1][1]:start]):
            merged[-1] = (merged[-1][0], end, lower[merged[-1][0]:end], merged[-1][3] if category == "genre" else category)
        elif not merged or start >= merged[-1][1]:
            merged.append((start, end, text, category))
    seen, sound, genre = set(), [], []
    for _, _, text, category in merged:
        if text not in seen:
            seen.add(text)
            (genre if category == "genre" else sound).append(text)
    join = lambda items: items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]  # noqa: E731
    parts = ([f"The music features {join(sound)}."] if sound else []) + ([f"The genre is {join(genre)}."] if genre else [])
    return " ".join(parts)


def phrases(caption: str) -> list:
    found = [(start, text) for start, _, text, _ in matches(caption)]
    ordered, seen = [], set()
    for _, text in sorted(found):
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def main():
    args = parse_args()
    captions = pd.read_csv(HW3 / f"data/{args.set_name}.csv", dtype={"id": str})
    suffix = "" if args.form == "tags" else "_sentence"
    record = HW3 / f"results/analysis/followed_{args.set_name}{suffix}.json"
    if args.stage == "build":
        texts = captions["caption"].map(sentence if args.form == "sentence" else (lambda c: ", ".join(phrases(c))))
        has = texts != ""
        out = HW3 / "data/followed"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": captions["id"][has], "text": texts[has]}).to_csv(out / f"{args.set_name}_extra{suffix}.csv", index=False)
        captions[has].to_csv(out / f"{args.set_name}_captions{suffix}.csv", index=False)
        info = {"set": args.set_name, "with_phrase": int(has.sum()), "median_phrases": float(texts[has].str.count(",").median() + 1)}
        if args.set_name == "dev":
            info["ids"] = [int(i) for i in captions["id"][has]]
        record.write_text(json.dumps(info, indent=1) + "\n")
        print(info if args.set_name == "test" else {k: v for k, v in info.items() if k != "ids"})
        return
    target = HW3 / f"outputs/{args.set_name}/{args.name}"
    has = set(pd.read_csv(HW3 / f"data/followed/{args.set_name}_extra{suffix}.csv", dtype={"id": str})["id"])
    for item_id in captions["id"]:
        if item_id not in has:
            shutil.copyfile(HW3 / f"outputs/{args.set_name}/rescale12/{item_id}.wav", target / f"{item_id}.wav")
    assert len(list(target.glob("*.wav"))) == len(captions)


if __name__ == "__main__":
    main()
