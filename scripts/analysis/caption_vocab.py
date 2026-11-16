import argparse
import ast
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
GENERIC = {"music", "song", "songs", "sound", "sounds", "sounding", "recording", "melody", "melodies", "track", "playing", "played", "plays",
           "with", "and", "the", "a", "an", "of", "in", "on", "for", "to", "is", "like", "feel", "vibe", "style", "background", "instrumental",
           "line", "lines", "sequence", "part", "parts", "rhythm", "tune", "beat", "beats", "pattern", "patterns", "tone", "piece"}

# Hand grouping of the frequent MusicCaps tag terms (main, 11-15). Grouping by CLAP text embeddings put 227 of 434 terms in one
# cluster, so the groups are by meaning. hear: how obvious the attribute is to a listener (our judgement).
# (category, group, hear, regex alternatives matched on lower-case captions with word boundaries)
LEXICON = [
    ("recording", "low quality", "yes", ["low quality", r"poor (?:audio )?quality", "poor audio", r"bad (?:audio )?quality", "bad audio", "noisy", "noise",
                                          "amateur", "muffled", "tinny", "mono", "unbalanced", "home video", "lo-?fi"]),
    ("recording", "high quality", "partly", ["high quality", "good quality", "studio", "clean", "well recorded", "professional", "crisp", "stereo"]),
    ("recording", "live, crowd", "yes", ["live", "concert", "crowd", "audience", "cheering", "applause", "clapping"]),
    ("recording", "reverb, room", "partly", ["reverb", "reverberant", "echo", "echoes", "echoing", "room", "spacious"]),
    ("tempo", "fast", "yes", ["fast", "uptempo", "up-tempo", "upbeat", "quick", "rapid"]),
    ("tempo", "medium", "partly", ["medium tempo", "moderate tempo", "mid-tempo", "medium fast"]),
    ("tempo", "slow", "yes", ["slow", "slowly", r"laid[- ]back", "leisurely"]),
    ("dynamics", "loud, intense", "yes", ["loud", "loudly", "intense", "aggressive", "powerful", "heavy", "vigorous", "energetic", "hard-hitting"]),
    ("dynamics", "soft, calm", "yes", ["soft", "softly", "mellow", "gentle", "calm", "calming", "soothing", "relaxing", "relaxed", "meditative",
                                        "quiet", "easygoing", "peaceful", "tranquil", "serene"]),
    ("mood", "happy, bright", "partly", ["happy", "cheerful", "joyful", "joyous", "fun", "playful", "uplifting", "positive", "buoyant", "lively",
                                          "exciting", "enthusiastic", "spirited", "optimistic"]),
    ("mood", "sad, emotional", "partly", ["sad", "melancholic", "melancholy", "emotional", "sentimental", "soulful", "romantic", "passionate",
                                           "nostalgic", "longing", "heartfelt"]),
    ("mood", "dark, tense", "partly", ["dark", "eerie", "haunting", "suspenseful", "ominous", "tense", "tension", "mysterious", "epic", "dramatic",
                                        "cinematic"]),
    ("vocals", "male voice", "yes", ["male vocals?", "male voices?", "male singers?", "male vocalists?", "man singing", "baritone"]),
    ("vocals", "female voice", "yes", ["female vocals?", "female voices?", "female singers?", "female vocalists?", "woman singing", "soprano"]),
    ("vocals", "choir, backing vocals", "yes", ["choir", "chorus", "backing vocals", "backup singers", r"vocal harmon\w*", r"harmoniz\w*", "choral"]),
    ("vocals", "rap, speech", "yes", ["rap", "rapping", "rapper", "talking", "speaking", "spoken", r"narrat\w*", "story telling"]),
    ("vocals", "no vocals", "yes", ["no voices?", "no singers?", "no vocals?", "instrumental", "without vocals"]),
    ("instrument", "guitar", "yes", ["guitars?", r"strumm\w*", "riffs?", "banjo", "ukulele"]),
    ("instrument", "bass", "partly", ["bass guitar", "bass ?lines?", "e-bass", "basslines?", "groovy bass", "synth bass", "upright bass", "double bass"]),
    ("instrument", "drums, percussion", "yes", ["drums?", "drumming", "kicks?", "snares?", r"hi[- ]hats?", "cymbals?", "percussions?", "percussive",
                                                "tambourine", "shakers?"]),
    ("instrument", "piano, keys", "yes", ["pianos?", "keyboards?", "keys", "organ", "harpsichord", "rhodes"]),
    ("instrument", "synth, electronic sounds", "yes", [r"synth\w*", r"arpeggi\w*", "drum machine", "808", "electronic drums", "digital drums"]),
    ("instrument", "strings", "yes", ["strings?", "stringed", "violins?", "cellos?", "viola", "orchestra", "harp", "fiddle"]),
    ("instrument", "brass, winds", "yes", ["brass", "trumpets?", "saxophones?", "sax", "horns?", "flutes?", "clarinet", "trombone", "harmonica",
                                           "accordion"]),
    ("instrument", "non-western instruments", "yes", ["tabla", "sitar", "didgeridoo", "erhu", "koto", "oud", "steel drums?", "bagpipes?"]),
    ("timbre", "acoustic", "yes", ["acoustic"]),
    ("timbre", "electric, distorted", "yes", ["electric", "distorted", "distortion", "overdriven?", "fuzz", "fuzzy"]),
    ("timbre", "groovy, punchy", "partly", ["groovy", "groove", "punchy", "bouncy", "syncopated"]),
    ("timbre", "shimmering, warm, lush", "partly", ["shimmering", r"sparkl\w*", "airy", "lush", "warm", "smooth", "bright"]),
    ("arrangement", "solo instrument", "yes", ["solo", "single instrument", "unaccompanied"]),
    ("genre", "rock, metal", "yes", ["rock", "metal", "punk", "grunge", "indie", "alternative"]),
    ("genre", "pop", "partly", ["pop"]),
    ("genre", "electronic, dance", "yes", ["electronic", "techno", "trance", "house", "edm", "dubstep", "club", "electro", "dance", "dancing"]),
    ("genre", "hip hop, r&b, funk", "yes", [r"hip[- ]?hop", "r ?& ?b", "r b", "funk", "funky", "soul", "gospel", "trap"]),
    ("genre", "jazz, blues", "yes", ["jazz", "jazzy", "blues", "bluesy", "swing", "bossa nova"]),
    ("genre", "folk, country", "yes", ["folk", "country", "bluegrass", "americana"]),
    ("genre", "classical, orchestral", "yes", ["classical", "orchestral", r"symphon\w*", "baroque", "opera"]),
    ("genre", "latin, reggae, world", "yes", ["latin", "reggae", "salsa", "world music", "indian", "african", "traditional", "regional"]),
    ("genre", "ambient, soundtrack", "partly", ["ambient", "soundtrack", "score", "chill", "chillout"]),
    ("context", "use or setting", "no", ["videos?", "movies?", "films?", "games?", r"advert\w*", "commercials?", "tutorial", "party", "wedding",
                                         "church", "meditation", "workout", "studying"]),
    ("other sounds", "non-music sounds", "yes", ["sound effects?", "birds?", "wind", "rain", "water", "footsteps", "engine", "sirens?", "beeps?",
                                                  "people talking", "children", "baby"]),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Caption vocabulary: frequent terms of the MusicCaps human tag lists (aspect_list), then hand-made groups of similar words "
        "and how many MusicCaps, dev and test captions use each group. Per-term counts are written for MusicCaps and dev only.")
    parser.add_argument("--min-unigram", type=int, default=60, help="minimum number of MusicCaps clips whose tags use the word")
    parser.add_argument("--min-bigram", type=int, default=40)
    return parser.parse_args()


def tokens(phrase: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-']*", phrase.lower())


def hits(captions: pd.Series, patterns: list[str]) -> pd.Series:
    return captions.str.contains(r"\b(?:" + "|".join(patterns) + r")\b", regex=True)


def main():
    args = parse_args()
    meta = pd.read_csv(HW3 / "data/musiccaps_meta/musiccaps-public.csv")
    clip_terms = []
    for aspects in meta["aspect_list"]:
        terms = set()
        for phrase in ast.literal_eval(aspects):
            words = tokens(phrase)
            terms |= {w for w in words if w not in GENERIC}
            terms |= {f"{a} {b}" for a, b in zip(words, words[1:]) if a not in GENERIC or b not in GENERIC}
        clip_terms.append(terms)
    counts = Counter(t for terms in clip_terms for t in terms)
    vocab = sorted((t for t, n in counts.items() if n >= (args.min_bigram if " " in t else args.min_unigram)), key=lambda t: -counts[t])

    sets = {"musiccaps": meta["caption"].str.lower(), **{name: pd.read_csv(HW3 / f"data/{name}.csv")["caption"].str.lower() for name in ("dev", "test")}}
    grouped = {t for _, _, _, patterns in LEXICON for t in vocab if re.search(r"\b(?:" + "|".join(patterns) + r")\b", t)}
    terms = pd.DataFrame({"term": vocab, "musiccaps_tag_clips": [counts[t] for t in vocab],
                          "dev_captions": [int(hits(sets["dev"], [re.escape(t)]).sum()) for t in vocab],
                          "in_a_group": [t in grouped for t in vocab]})
    rows = []
    for category, group, hear, patterns in LEXICON:
        row = {"category": category, "group": group, "hear": hear}
        for name, captions in sets.items():
            n = int(hits(captions, patterns).sum())
            row[f"{name}_captions"] = n if name != "musiccaps" else round(100 * n / len(captions), 1)
        row["words"] = ", ".join(p.replace("\\", "").replace("?:", "") for p in patterns[:8])
        rows.append(row)
    groups = pd.DataFrame(rows).rename(columns={"musiccaps_captions": "musiccaps_pct"})
    word_counts = {name: captions.str.split().str.len() for name, captions in sets.items()}
    out = HW3 / "results/analysis"
    terms.to_csv(out / "caption_vocab_terms.csv", index=False)
    groups.to_csv(out / "caption_vocab_groups.csv", index=False)
    print({name: f"median {w.median():.0f} words [{w.min()}-{w.max()}]" for name, w in word_counts.items()})
    print(f"{len(vocab)} frequent tag terms; {terms['in_a_group'].sum()} fall in a group; ungrouped top 40:",
          ", ".join(terms.loc[~terms["in_a_group"], "term"].head(40)))
    print(groups.drop(columns="words").to_string(index=False))


if __name__ == "__main__":
    main()
