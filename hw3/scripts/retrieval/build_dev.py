import argparse
import ast
import csv
import json
import random
from pathlib import Path

HW3 = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Build the MusicCaps dev set used to choose settings (never the test set).")
    parser.add_argument("--musiccaps-csv", type=Path, default=HW3 / "data/musiccaps_meta/musiccaps-public.csv")
    parser.add_argument("--audio-dir", type=Path, default=HW3 / "data/musiccaps/audio_mp3")
    parser.add_argument("--retrieval", type=Path, default=HW3 / "results/retrieval.json")
    parser.add_argument("--top5", type=Path, default=HW3 / "results/retrieval_top5.csv")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--min-words", type=int, default=9)
    parser.add_argument("--max-words", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    available = {path.stem for path in args.audio_dir.glob("*.mp3")}
    retrieved = json.loads(args.retrieval.read_text()).values()
    test_clips = {Path(value).stem for value in retrieved}
    assert len(test_clips) > 0 and test_clips <= available, "retrieval.json values must be MusicCaps ids of local clips"
    # Near-duplicates of a test target can sit in its top 5, so all top-5 clips stay out of dev.
    with open(args.top5, newline="") as handle:
        test_clips |= {row["youtube_id"] for row in csv.DictReader(handle)}
    assert test_clips <= available
    with open(args.musiccaps_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Word limits match the test captions (9-70 words), so dev and test differ less in prompt length.
    pool = [row for row in rows if row["ytid"] in available and row["ytid"] not in test_clips
            and args.min_words <= len(row["caption"].split()) <= args.max_words]
    pool.sort(key=lambda row: row["ytid"])
    chosen = random.Random(args.seed).sample(pool, args.size)

    data = HW3 / "data"
    reference_dir = HW3 / "outputs/ref/dev_real"
    reference_dir.mkdir(parents=True, exist_ok=True)
    with open(data / "dev.csv", "w", newline="") as captions_file, \
            open(data / "dev_meta.csv", "w", newline="") as meta_file, \
            open(data / "dev_tags_oracle.jsonl", "w") as tags_file:
        captions = csv.writer(captions_file)
        meta = csv.writer(meta_file)
        captions.writerow(["id", "caption"])
        meta.writerow(["id", "ytid", "aspect_list", "is_audioset_eval"])
        for index, row in enumerate(chosen):
            captions.writerow([index, row["caption"]])
            meta.writerow([index, row["ytid"], row["aspect_list"], row["is_audioset_eval"]])
            aspects = ast.literal_eval(row["aspect_list"])
            assert isinstance(aspects, list)
            tags_file.write(json.dumps({"id": str(index), "tags": aspects}) + "\n")
            link = reference_dir / f"{index}.mp3"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to((args.audio_dir / f"{row['ytid']}.mp3").resolve())
    print(f"dev set: {len(chosen)} of {len(pool)} eligible clips (excluded {len(test_clips)} top-5 test-retrieved clips); "
          "dev_tags_oracle.jsonl holds human aspect lists, for the oracle-tags ablation only")


if __name__ == "__main__":
    main()
