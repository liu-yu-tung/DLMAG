import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))
from analysis.caption_vocab import LEXICON, hits  # noqa: E402

GROUPS = {f"{category}: {group}": patterns for category, group, _, patterns in LEXICON}

# axis: (source group, target group, ordered (pattern, replacement) swaps). A caption qualifies when it has the source group,
# lacks the target group, and the swaps change it. Insert axes append one sentence instead and use captions with no genre word.
AXES = {
    "voice_male_to_female": ("vocals: male voice", "vocals: female voice",
                             [(r"\bmale\b", "female"), (r"\bman\b", "woman"), (r"\bmen\b", "women"), (r"\bhe\b", "she"), (r"\bhis\b", "her"),
                              (r"\bhim\b", "her"), (r"\bbaritone\b", "soprano")]),
    "guitar_to_piano": ("instrument: guitar", "instrument: piano, keys",
                        [(r"\b(?<!bass )guitars\b", "pianos"), (r"\b(?<!bass )guitar\b", "piano"), (r"\bguitarist\b", "pianist"),
                         (r"\bstrumm(?:ed|ing)\b", "played"), (r"\bstrums?\b", "plays")]),
    "acoustic_to_electric": ("timbre: acoustic", "timbre: electric, distorted", [(r"\bacoustic\b", "distorted electric")]),
    "soft_to_loud": ("dynamics: soft, calm", "dynamics: loud, intense",
                     [(r"\bsoftly\b", "loudly"), (r"\bsoft\b", "loud"), (r"\bmellow\b", "aggressive"), (r"\bgentle\b", "powerful"),
                      (r"\bcalm(?:ing)?\b", "intense"), (r"\bsoothing\b", "intense"), (r"\brelax(?:ing|ed)\b", "energetic"),
                      (r"\bquiet\b", "loud"), (r"\bpeaceful\b", "aggressive"), (r"\bmeditative\b", "intense"), (r"\beasygoing\b", "aggressive"),
                      (r"\btranquil\b", "intense"), (r"\bserene\b", "intense")]),
    "happy_to_sad": ("mood: happy, bright", "mood: sad, emotional",
                     [(r"\bhappy\b", "sad"), (r"\bcheerful\b", "melancholic"), (r"\bjoy(?:ful|ous)\b", "sorrowful"), (r"\bfun\b", "somber"),
                      (r"\bplayful\b", "mournful"), (r"\buplifting\b", "depressing"), (r"\bpositive\b", "gloomy"), (r"\blively\b", "melancholic"),
                      (r"\bexciting\b", "sad"), (r"\benthusiastic\b", "sorrowful"), (r"\bspirited\b", "melancholic"), (r"\boptimistic\b", "pessimistic")]),
    "fast_to_slow": ("tempo: fast", "tempo: slow",
                     [(r"\bfast\b", "slow"), (r"\bup-?tempo\b", "slow tempo"), (r"\bupbeat\b", "slow"), (r"\bquick\b", "slow"), (r"\brapid\b", "slow")]),
    "low_quality_to_studio": ("recording: low quality", "recording: high quality",
                              [(r"\blow quality\b", "high quality"), (r"\bpoor (audio )?quality\b", r"excellent \1quality"),
                               (r"\bbad (audio )?quality\b", r"excellent \1quality"), (r"\bpoor audio\b", "clean audio"),
                               (r"\bbad audio\b", "clean audio"), (r"\bnoisy\b", "clean"), (r"\bnoise\b", "clarity"),
                               (r"\bamateur\b", "professional studio"), (r"\bmuffled\b", "crisp"), (r"\btinny\b", "full"), (r"\bmono\b", "stereo")]),
}
INSERTS = {"insert_jazz": " The genre is jazz.", "insert_country": " The genre is country."}
# Added 11-15 19:15 (user: HW1 curated decade and market labels; held as a bonus until the validity checks pass, user 19:17). Detectors: HW1 TASK A decade and TASK B market classifiers
# (scripts/hw1_detectors.py); genre pair also by the caption-word probes. Same 12 captions as the genre inserts, plus 12 more.
ORIGIN_INSERTS = {"insert_1960s": " The song is from the 1960s.", "insert_2010s": " The song is from the 2010s.",
                  "insert_brazil": " The song is from Brazil.", "insert_germany": " The song is from Germany.",
                  "insert_latin": " The genre is Latin.", "insert_techno": " The genre is techno."}
ORIGIN_WORDS = r"\b(?:(?:19|20)?[0-9]0'?s|sixties|seventies|eighties|nineties|brazil\w*|german\w*|latin|techno|retro|vintage|modern)\b"
ORIGIN_EXTRA = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Word-swap edits of dev captions for the 'does the model follow the word' test. Writes data/edits/<axis>.csv "
        "(original dev ids, so generation reuses the original seed) and results/analysis/edits_manifest.json (ids per axis, no text).")
    parser.add_argument("--per-axis", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extend", action="store_true",
                        help="also write data/edits/<axis>_ext.csv with every other eligible caption (manifest key ext_ids); "
                        "the pre-registered csvs must come out unchanged")
    return parser.parse_args()


def apply(caption: str, swaps) -> str:
    for pattern, replacement in swaps:
        caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
    return caption


def write(out_dir: Path, axis: str, chosen: pd.Series, edited: dict, extend: bool):
    frame = pd.DataFrame({"id": chosen.to_numpy(), "caption": [edited[i] for i in chosen]})
    path = out_dir / f"{axis}.csv"
    if extend:
        assert frame.equals(pd.read_csv(path, dtype={"id": str})), f"{path}: pre-registered edits would change"
        rest = sorted(set(edited) - set(chosen), key=int)
        pd.DataFrame({"id": rest, "caption": [edited[i] for i in rest]}).to_csv(out_dir / f"{axis}_ext.csv", index=False)
    else:
        frame.to_csv(path, index=False)


def main():
    args = parse_args()
    dev = pd.read_csv(HW3 / "data/dev.csv", dtype={"id": str}).set_index("id")["caption"]
    lower = dev.str.lower()
    out_dir = HW3 / "data/edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for axis, (source, target, swaps) in AXES.items():
        eligible = lower.index[hits(lower, GROUPS[source]) & ~hits(lower, GROUPS[target])]
        edited = {i: apply(dev[i], swaps) for i in eligible}
        edited = {i: text for i, text in edited.items() if text != dev[i] and hits(pd.Series([text.lower()]), GROUPS[target]).iloc[0]}
        chosen = pd.Series(sorted(edited, key=int)).sample(min(args.per_axis, len(edited)), random_state=args.seed).sort_values(key=lambda s: s.astype(int))
        write(out_dir, axis, chosen, edited, args.extend)
        manifest[axis] = {"source": source, "target": target, "eligible": len(edited), "ids": [int(i) for i in chosen]}
        if args.extend:
            manifest[axis]["ext_ids"] = sorted(int(i) for i in set(edited) - set(chosen))
    genre_free = lower.index[~pd.concat([hits(lower, p) for g, p in GROUPS.items() if g.startswith("genre:")], axis=1).any(axis=1)]
    chosen = pd.Series(sorted(genre_free, key=int)).sample(min(args.per_axis, len(genre_free)), random_state=args.seed).sort_values(key=lambda s: s.astype(int))
    for axis, sentence in INSERTS.items():
        write(out_dir, axis, chosen, {i: dev[i].rstrip() + sentence for i in genre_free}, args.extend)
        manifest[axis] = {"source": None, "target": "genre: jazz, blues" if "jazz" in axis else "genre: folk, country",
                          "eligible": len(genre_free), "ids": [int(i) for i in chosen]}
        if args.extend:
            manifest[axis]["ext_ids"] = sorted(int(i) for i in set(genre_free) - set(chosen))
    clean = [i for i in genre_free if not re.search(ORIGIN_WORDS, lower[i])]
    assert set(chosen) <= set(clean), "a registered genre-insert caption already names a decade, country or the new genres"
    extra = pd.Series(sorted(set(clean) - set(chosen), key=int)).sample(ORIGIN_EXTRA, random_state=args.seed).sort_values(key=lambda s: s.astype(int))
    for axis, sentence in ORIGIN_INSERTS.items():
        for name, ids in ((axis, chosen), (f"{axis}_ext", extra)):
            frame = pd.DataFrame({"id": ids.to_numpy(), "caption": [dev[i].rstrip() + sentence for i in ids]})
            path = out_dir / f"{name}.csv"
            assert not path.exists() or frame.equals(pd.read_csv(path, dtype={"id": str})), f"{path}: edits would change"
            frame.to_csv(path, index=False)
        manifest[axis] = {"source": None, "target": sentence.strip(), "eligible": len(clean), "ids": [int(i) for i in chosen],
                          "ext_ids": [int(i) for i in extra]}
    (HW3 / "results/analysis/edits_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for axis, entry in manifest.items():
        print(f"{axis}: {entry['eligible']} eligible, {len(entry['ids'])} chosen")


if __name__ == "__main__":
    main()
