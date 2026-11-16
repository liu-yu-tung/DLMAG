import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3))

from text_to_music.eval_common import code_version  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-of-N selection: per prompt, keep the clip of the candidate run with the highest CLAP cosine "
        "to its caption. Needs each run's results/<set>__<run>/per_prompt.csv (written by evaluate.py). "
        "The selector is the evaluation CLAP model, so the selected run's CLAP score is biased upward; "
        "judge it by the other metrics.")
    parser.add_argument("--set", dest="set_name", choices=("dev", "test"), required=True)
    parser.add_argument("--runs", required=True, help="comma-separated candidate runs, e.g. cfg7,cfg7_seed1,cfg7_seed2")
    parser.add_argument("--name", required=True, help="name of the selected run, e.g. cfg7_best3")
    parser.add_argument("--score", default="clap_cosine", help="column of the score file to maximise")
    parser.add_argument("--score-file", default="per_prompt.csv", help="file in each run's results folder, e.g. mqa_per_item.csv")
    parser.add_argument("--out-root", type=Path, default=HW3 / "outputs")
    parser.add_argument("--results-root", type=Path, default=HW3 / "results")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = args.runs.split(",")
    assert len(runs) >= 2 and len(set(runs)) == len(runs), runs
    columns = {
        run: pd.read_csv(args.results_root / f"{args.set_name}__{run}" / args.score_file, dtype={"id": str})
        .set_index("id")[args.score]
        for run in runs
    }
    assert len({frozenset(column.index) for column in columns.values()}) == 1, "candidate runs do not cover the same ids"
    scores = pd.DataFrame(columns)
    # MQAScore is NaN when the judge returned no usable answer; such a clip can still be chosen, but only last.
    n_missing = int(scores.isna().sum().sum())
    scores = scores.fillna(-1.0)
    # idxmax keeps the first run on a tie, so the listed order (baseline first) decides ties.
    chosen = scores.idxmax(axis=1)

    audio_dir = args.out_root / args.set_name / args.name
    record_dir = args.results_root / f"{args.set_name}__{args.name}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    for item_id, run in chosen.items():
        source = args.out_root / args.set_name / run / f"{item_id}.wav"
        assert source.exists(), source
        shutil.copyfile(source, audio_dir / f"{item_id}.wav")

    table = scores.copy()
    table["chosen_run"] = chosen
    table["chosen_score"] = scores.max(axis=1)
    table.sort_index(key=lambda index: index.astype(int)).to_csv(record_dir / "selection.csv")
    summary = {
        "set": args.set_name, "name": args.name, "candidates": runs, "score": args.score, "score_file": args.score_file,
        "n_items": len(chosen), "n_missing_scores": n_missing, "chosen_counts": {run: int((chosen == run).sum()) for run in runs},
        "mean_score_per_candidate": {run: float(columns[run].mean()) for run in runs},
        "mean_score_selected": float(scores.max(axis=1).mean()),
        "code_version": code_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (record_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
