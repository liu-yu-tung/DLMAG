import argparse
import csv
import json
from pathlib import Path

import numpy as np

HW3 = Path(__file__).resolve().parents[2]
REFERENCE = {"dev": "dev__cfg7", "test": "test__cfg7"}
RESAMPLES, SEED = 1000, 0


def read_per_prompt(run_dir: Path) -> dict[str, np.ndarray]:
    with open(run_dir / "per_prompt.csv", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["id"]))
    return {key: np.array([float(row[key]) for row in rows]) for key in ("clap_cosine", "kl_passt_softmax")}


def paired_difference(values: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    difference = values - reference
    indices = np.random.default_rng(SEED).integers(0, len(difference), size=(RESAMPLES, len(difference)))
    means = difference[indices].mean(axis=1)
    return float(difference.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def value(metrics: dict, key: str) -> float | None:
    entry = metrics.get(key)
    return None if entry is None else float(entry["value"])


def main():
    parser = argparse.ArgumentParser(description="One row per evaluated run: metrics plus paired CLAP/KL differences "
                                                 "against the CFG-7 run of the same set (dev__cfg7, test__cfg7).")
    parser.add_argument("--out", type=Path, default=HW3 / "results/analysis/findings_table.csv")
    args = parser.parse_args()
    rows = []
    for run_dir in sorted((HW3 / "results").glob("*__*")):
        if not (run_dir / "metrics.json").exists() or not (run_dir / "per_prompt.csv").exists():
            continue
        set_name = run_dir.name.split("__")[0]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        per_prompt = read_per_prompt(run_dir)
        reference = read_per_prompt(HW3 / "results" / REFERENCE[set_name])
        info = json.loads((run_dir / "gen_info.json").read_text()) if (run_dir / "gen_info.json").exists() else {}
        mqa = json.loads((run_dir / "mqa_summary.json").read_text()) if (run_dir / "mqa_summary.json").exists() else {}
        clap_difference = paired_difference(per_prompt["clap_cosine"], reference["clap_cosine"])
        kl_difference = paired_difference(per_prompt["kl_passt_softmax"], reference["kl_passt_softmax"])
        rows.append({
            "run": run_dir.name,
            "clap": value(metrics, "LAION-CLAP-Score"),
            "clap_difference": clap_difference[0], "clap_difference_low": clap_difference[1], "clap_difference_high": clap_difference[2],
            "kl_passt": value(metrics, "KL-PASST-softmax"),
            "kl_difference": kl_difference[0], "kl_difference_low": kl_difference[1], "kl_difference_high": kl_difference[2],
            "fd_vgg_musiccaps": value(metrics, "FD-VGG_musiccaps"),
            "fd_mert_musiccaps": value(metrics, "fd_mert_musiccaps"),
            "clip_fraction_percent": 100 * info["mean_clip_fraction"] if "mean_clip_fraction" in info else None,
            "seconds_per_clip": info.get("mean_seconds_per_item"),
            "mqascore": mqa.get("mqascore_mean"),
            "mqascore_low": (mqa.get("mqascore_ci95") or [None, None])[0],
            "mqascore_high": (mqa.get("mqascore_ci95") or [None, None])[1],
            **{f"mqa_{name}": score for name, score in mqa.get("attribute_mean", {}).items()},
        })
    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(number, digits=3):
        return "" if number is None else f"{number:.{digits}f}"

    print("| run | CLAP | dCLAP vs cfg7 [95%] | KL-PaSST | dKL [95%] | FD-VGG mc | FD-MERT mc | clip % | s/clip | MQAScore |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        mqa_text = "" if row["mqascore"] is None else f"{row['mqascore']:.3f} [{row['mqascore_low']:.3f}, {row['mqascore_high']:.3f}]"
        print(f"| {row['run']} | {cell(row['clap'])} | {row['clap_difference']:+.3f} [{row['clap_difference_low']:+.3f}, {row['clap_difference_high']:+.3f}] "
              f"| {cell(row['kl_passt'])} | {row['kl_difference']:+.3f} [{row['kl_difference_low']:+.3f}, {row['kl_difference_high']:+.3f}] "
              f"| {cell(row['fd_vgg_musiccaps'], 2)} | {cell(row['fd_mert_musiccaps'], 1)} | {cell(row['clip_fraction_percent'], 3)} "
              f"| {cell(row['seconds_per_clip'], 1)} | {mqa_text} |")


if __name__ == "__main__":
    main()
