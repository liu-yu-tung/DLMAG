"""Quality vs transformer forward passes per clip, for dev cfg7 and its interval, compose, best-of-3 and rescale variants. Re-runnable; skips runs without metrics."""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HW3 = Path(__file__).resolve().parents[2]
RESULTS = HW3 / "results"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BASELINE = "dev__cfg7"
RUN_PATTERN = re.compile(r"^dev__(cfg7|.*(interval|compose|best3|rescale).*)$")
# steps (of 100) with sigma above the threshold, read off the SAO schedule; matches measured forward_samples of the interval runs
STEPS_ABOVE_SIGMA = {0.0: 100, 2.6: 71, 12.0: 50, 30.0: 38, 110.0: 21, math.inf: 0}
SCHEDULE_STEPS = 100
COLORS = {"cfg7": "#0b0b0b", "interval": "#2a78d6", "compose": "#1baf7a", "best3": "#e34948", "rescale": "#eb6834"}
# column, y label, lower is better
PANELS = [
    ("clap_difference", "dCLAP vs cfg7 (higher is better)", False),
    ("kl_passt", "KL-PaSST (lower is better)", True),
    ("fd_mert_musiccaps", "FD-MERT (lower is better)", True),
]


def family(run: str) -> str:
    return next((name for name in ("interval", "compose", "best3", "rescale") if name in run), "cfg7")


def forward_passes(run: str) -> float:
    info = json.loads((RESULTS / run / "gen_info.json").read_text())
    if info["settings"]["steps"] != SCHEDULE_STEPS:
        raise ValueError(f"{run}: the sigma counts are for {SCHEDULE_STEPS} steps")
    guidance = info["settings"]["guidance"]
    low, high = guidance.get("interval_sigma_low", 0.0), guidance.get("interval_sigma_high", math.inf)
    guided = STEPS_ABOVE_SIGMA[low] - STEPS_ABOVE_SIGMA[high]
    guided_passes = 3 if guidance.get("second_scale", 0) > 0 else 2
    return SCHEDULE_STEPS - guided + guided * guided_passes


def total_passes(run: str) -> float:
    selection = RESULTS / run / "selection.json"
    if not selection.exists():
        return forward_passes(run)
    set_name = run.split("__")[0]
    return sum(total_passes(f"{set_name}__{candidate}") for candidate in json.loads(selection.read_text())["candidates"])


def measured_passes(run: str) -> float | None:
    path = RESULTS / run / "quick.jsonl"
    if not path.exists():
        return None
    counts = [json.loads(line).get("forward_samples") for line in path.read_text().splitlines()]
    return float(np.mean(counts)) if counts and None not in counts else None


def seconds_per_clip(run: str) -> float | None:
    selection = RESULTS / run / "selection.json"
    if selection.exists():
        set_name = run.split("__")[0]
        parts = [seconds_per_clip(f"{set_name}__{c}") for c in json.loads(selection.read_text())["candidates"]]
        return None if None in parts else float(sum(parts))
    path = RESULTS / run / "gen_info.json"
    return json.loads(path.read_text()).get("mean_seconds_per_item") if path.exists() else None


def collect(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        if not RUN_PATTERN.match(row["run"]) or pd.isna(row["clap"]):
            continue
        run = row["run"]
        rows.append({**row.to_dict(), "family": family(run), "passes": total_passes(run), "measured_passes": measured_passes(run),
                     "seconds_per_clip_total": seconds_per_clip(run)})
    return pd.DataFrame(rows).sort_values("passes", ignore_index=True)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot(data: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=(10, 12), facecolor=SURFACE, layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.15)
    baseline = data[data["run"] == BASELINE].iloc[0]
    for ax, (column, label, _) in zip(axes, PANELS):
        ax.axhline(baseline[column], color=AXIS, lw=1, zorder=1)
        for fam, group in data.groupby("family"):
            marker = "D" if fam == "best3" else "o"
            if column == "clap_difference":
                err = [group[column] - group["clap_difference_low"], group["clap_difference_high"] - group[column]]
                ax.errorbar(group["passes"], group[column], yerr=err, fmt=marker, color=COLORS[fam], ms=7, mec=SURFACE, capsize=3, lw=1.2, label=fam, zorder=3)
            else:
                ax.scatter(group["passes"], group[column], marker=marker, color=COLORS[fam], s=50, edgecolor=SURFACE, label=fam, zorder=3)
        for _, row in data.iterrows():
            below = row["run"] == BASELINE
            ax.annotate(row["run"].removeprefix("dev__"), (row["passes"], row[column]), xytext=(5, -13 if below else 6), textcoords="offset points",
                        fontsize=8, color=INK if below else MUTED)
        ax.margins(y=0.15)
        ax.set_ylabel(label, fontsize=10, color=MUTED)
        style_axis(ax)
    axes[0].legend(loc="best", ncols=3, frameon=False, fontsize=9, labelcolor=MUTED)
    axes[-1].set_xlabel("Transformer forward passes per clip (best-of-3 counts all three candidates)", fontsize=10, color=MUTED)
    axes[-1].set_xlim(120, 650)
    fig.suptitle("Dev set: quality vs compute (dCLAP bars: paired bootstrap interval vs cfg7; best-of-3 is picked by CLAP)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=RESULTS / "analysis" / "findings_table.csv")
    ap.add_argument("--out_fig", type=Path, default=RESULTS / "analysis" / "figures" / "compute_dev.png")
    ap.add_argument("--out_csv", type=Path, default=RESULTS / "analysis" / "compute_dev.csv")
    args = ap.parse_args()

    data = collect(pd.read_csv(args.table))
    assert BASELINE in set(data["run"]), "cfg7 has no metrics"
    mismatch = data[data["measured_passes"].notna() & (data["measured_passes"] != data["passes"])]
    assert mismatch.empty, mismatch[["run", "passes", "measured_passes"]]
    single = data[data["family"] != "best3"]
    fit = np.polyfit(single["passes"], single["seconds_per_clip"], 1)
    data["seconds_per_clip_fit"] = np.where(data["family"] == "best3", np.nan, np.polyval(fit, data["passes"]))
    columns = ["run", "family", "passes", "measured_passes", "seconds_per_clip", "seconds_per_clip_total", "seconds_per_clip_fit",
               "clap", "clap_difference", "clap_difference_low", "clap_difference_high", "kl_passt", "fd_mert_musiccaps"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    data[columns].round(4).to_csv(args.out_csv, index=False)
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plot(data, args.out_fig)
    print(data[columns].round(3).to_string(index=False))
    print(f"fit over single-pass runs: seconds = {fit[1]:.2f} + {fit[0]:.4f} * passes")


if __name__ == "__main__":
    main()
