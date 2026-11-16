import argparse
import csv
import json
import time
import tomllib
from pathlib import Path

import numpy as np

HW3 = Path(__file__).resolve().parents[2]
RESULTS = HW3 / "results"
RESAMPLES, SEED = 1000, 0
DEFAULT = "dev__cfg7"
SOURCES = {"clap": ("per_prompt.csv", "clap_cosine"), "kl": ("per_prompt.csv", "kl_passt_softmax"),
           "mqa": ("mqa_per_item.csv", "mqascore"), "alm": ("alm_qwen_per_item.csv", "alm_qwen"),
           "vocal_db": ("vocal_stems.csv", "vocal_ratio_db")}


def per_prompt(run: str, metric: str) -> dict[str, float] | None:
    filename, column = SOURCES[metric]
    path = RESULTS / run / filename
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        return {row["id"]: float(row[column]) for row in csv.DictReader(handle) if row[column] not in ("", "nan")}


def paired(run: str, against: str, metric: str) -> tuple[float, float, float, float] | None:
    values, reference = per_prompt(run, metric), per_prompt(against, metric)
    if values is None or reference is None:
        return None
    ids = sorted(set(values) & set(reference), key=int)
    difference = np.array([values[i] - reference[i] for i in ids])
    difference = difference[~np.isnan(difference)]
    indices = np.random.default_rng(SEED).integers(0, len(difference), size=(RESAMPLES, len(difference)))
    means = difference[indices].mean(axis=1)
    return float(difference.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((difference > 0).mean())


def interval_text(result, digits=3) -> str:
    return "pending" if result is None else f"{result[0]:+.{digits}f} [{result[1]:+.{digits}f}, {result[2]:+.{digits}f}]"


def metric_value(run: str, key: str) -> float | None:
    path = RESULTS / run / "metrics.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(key)
    return None if entry is None else float(entry["value"])


def seconds_per_clip(run: str) -> float | None:
    path = RESULTS / run / "gen_info.json"
    return json.loads(path.read_text()).get("mean_seconds_per_item") if path.exists() else None


def paired_fd(set_name: str) -> dict:
    path = RESULTS / "analysis" / f"paired_fd_{set_name}.json"
    return json.loads(path.read_text())["runs"] if path.exists() else {}


def base_of(run: str) -> str:
    return run.split("_best3")[0] if "_best3" in run else DEFAULT


def board(lines: list[str]):
    fd = paired_fd("dev")
    runs = sorted(p.name for p in RESULTS.glob("dev__*") if (p / "per_prompt.csv").exists() and p.name not in (DEFAULT, "dev__real"))
    rows = []
    for run in runs:
        clap, kl, mqa = paired(run, DEFAULT, "clap"), paired(run, DEFAULT, "kl"), paired(run, DEFAULT, "mqa")
        vgg = fd.get(run.split("__", 1)[1], {}).get("VGG")
        fd_result = None if vgg is None else (vgg["difference"], *vgg["difference_ci95"])
        worse = [name for name, result in (("KL", kl), ("FD-VGG", fd_result)) if result is not None and result[1] > 0]
        if worse:
            verdict = "out: worse " + " and ".join(worse)
        elif fd_result is None:
            verdict = "FD-VGG pending"
        elif run.endswith("_best3"):
            verdict = "analysis only (ranked by the scoring checkpoint)"
        elif clap[1] > 0:
            verdict = "candidate"
        else:
            verdict = "tied with cfg7 (default wins)"
        rows.append((metric_value(run, "LAION-CLAP-Score") or 0, run, clap, kl, fd_result, mqa, verdict))
    rows.sort(reverse=True)
    lines += ["## Selection board (dev, paired differences vs dev__cfg7, 95% bootstrap over prompts)", "",
              "run | CLAP | dCLAP | dKL | dFD-VGG | dMQAScore | rule"]
    for clap_value, run, clap, kl, fd_result, mqa, verdict in rows:
        lines.append(f"{run} | {clap_value:.3f} | {interval_text(clap)} | {interval_text(kl, 2)} | {interval_text(fd_result, 2)} | {interval_text(mqa)} | {verdict}")
    lines += ["", "Rule: highest dev CLAP unless FD-VGG or KL is worse beyond its interval; ties go to cfg7. 'candidate' = passes and dCLAP excludes 0.",
              "The choice is made only at the check-in with the user.", ""]
    candidates = [row for row in rows if row[6] == "candidate"]
    lines += ["## Head-to-head among candidates (paired, row minus the next lower candidate)", ""]
    for upper, lower in zip(candidates, candidates[1:]):
        clap = paired(upper[1], lower[1], "clap")
        tie = "TIE (interval includes 0)" if clap[1] <= 0 <= clap[2] else "separated"
        lines.append(f"{upper[1]} vs {lower[1]}: dCLAP {interval_text(clap)}, win rate {clap[3]:.2f}, dKL {interval_text(paired(upper[1], lower[1], 'kl'), 2)} -> {tie}")
    lines += ["", "## Best-of-3 runs vs their own seed 0 (decision #2: better on one metric, not worse beyond its interval on any other)", ""]
    for _, run, *_ in rows:
        if "_best3" in run:
            base = base_of(run)
            kl, mqa = paired(run, base, "kl"), paired(run, base, "mqa")
            if mqa is None:
                verdict = "MQAScore pending"
            else:
                worse = kl[1] > 0 or mqa[2] < 0
                better = kl[0] < 0 or mqa[0] > 0
                verdict = "QUALIFIES" if better and not worse else "does not qualify"
            lines.append(f"{run} vs {base}: dCLAP {interval_text(paired(run, base, 'clap'))} ({'cross-checkpoint ranker' if run.endswith('_c8998') else 'circular'}), dKL {interval_text(kl, 2)}, dMQAScore {interval_text(mqa)} -> {verdict}")
    lines.append("")


def predictions(lines: list[str]):
    entries = tomllib.loads((HW3 / "docs/predictions.toml").read_text())["prediction"]
    lines += ["## Predictions (docs/predictions.toml): MISSED ones are where the strategy must change", ""]
    pending = []
    for entry in entries:
        run, against, metric = entry["run"], entry["against"], entry["metric"]
        label = f"{run} vs {against}, {metric} in [{entry['low']:+.3f}, {entry['high']:+.3f}] (written {entry['written']})"
        if metric == "kept_fraction":
            path = RESULTS / run / "loop_selection.csv"
            if path.exists():
                with open(path, newline="") as handle:
                    tried = [row["kept_round2"] == "True" for row in csv.DictReader(handle) if row["round2_score"] not in ("", "nan")]
                share = sum(tried) / len(tried)
                result = (share, share, share, 0.0)
            else:
                result = None
        elif metric == "seconds_per_clip":
            a, b = seconds_per_clip(run), seconds_per_clip(against)
            result = None if a is None or b is None else (a / b, a / b, a / b, 0.0)
        else:
            result = paired(run, against, metric)
        if result is None:
            pending.append(label)
            continue
        met = entry["low"] <= result[0] <= entry["high"]
        lines += [f"{'met   ' if met else 'MISSED'} {label}: got {interval_text(result)}",
                  f"       next: {entry['if_met'] if met else entry['if_missed']}"]
    lines += ["", "Pending:"] + [f"  {label}" for label in pending] + [""]


def main():
    parser = argparse.ArgumentParser(description="Selection board, candidate head-to-heads and prediction verdicts from the run records; "
                                                 "rewritten after every queue job (scripts/generation/post_job.sh). Reading order: docs/strategy_v3.txt.")
    parser.add_argument("--out", type=Path, default=RESULTS / "analysis/knowledge.md")
    args = parser.parse_args()
    lines = [f"HW3 knowledge sheet, rewritten {time.strftime('%m-%d %H:%M')} by scripts/analysis/knowledge.py", ""]
    predictions(lines)
    board(lines)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
