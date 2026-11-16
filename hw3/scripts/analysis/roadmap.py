import argparse
import textwrap
from pathlib import Path

from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
RESULTS = HW3 / "results"
RESAMPLES, SEED = 1000, 0
WRAP, LINE = 62, 1.12
DARK, LIGHT, BLUE, ORANGE, GREY, LBLUE, GREEN = "#22262e", "#f6f7f9", "#3d6bd6", "#d9822b", "#9aa1ab", "#8fb0f0", "#2e7d55"

# (stage title, cause, tendency, [(run, label, role)]); role: kept, chosen, rejected, circular, pilot.
# Every run is compared with dev cfg7 on the same seed (seed-0 runs vs cfg7 seed 0); "kept" rows are cfg7 itself.
STAGES = [
    ("1  Guidance scale", "stronger guidance pulls the clip toward the text",
     "CLAP 3 < 7 < 12; scale 12 is louder (-16.7 vs -17.4 dBFS), so it returns only with rescale (5)",
     [("cfg7", "kept: scale 7", "kept"), ("cfg3", "scale 3", "rejected"), ("cfg12", "scale 12", "rejected")]),
    ("2  Prompt format", "likely: the text encoder was trained on descriptive text; a tag list drops the links between words",
     "caption = tags + caption >> tags only, even with human-written tags",
     [("cfg7", "kept: the caption", "kept"), ("tags", "LLM tags only", "rejected"), ("tags_oracle", "human tags only", "rejected"),
      ("tags_caption", "tags + caption", "rejected")]),
    ("3  Guidance form (APG)", "APG drops the guidance part parallel to the prediction, so it acts like a lower scale",
     "APG12 ~ CFG12 on CLAP and worse FD-VGG: no gain of its own",
     [("cfg7", "kept: plain CFG", "kept"), ("apg7", "APG, scale 7", "rejected"), ("apg12", "APG, scale 12", "rejected"),
      ("apg12_r2", "APG 12, no norm cap", "rejected")]),
    ("4  Guidance interval", "early guidance seems washed out (audio 0.98 similar to cfg7); late steps carry the detail",
     "skipping early steps is free; skipping late steps costs, more the more are skipped",
     [("cfg7", "kept: guidance on all 100 steps", "kept"), ("interval_early", "off for sigma > 110 (21 steps)", "rejected"),
      ("interval_early30", "off for sigma > 30 (38 steps)", "rejected"),
      ("interval_late", "off for sigma <= 2.6", "rejected"), ("interval_late12", "off for sigma <= 12", "rejected")]),
    ("5  CFG rescale", "rescale pulls the guided output back to the spread of the conditional one, undoing over-saturation",
     "no effect at scale 7; at 12 CLAP and KL both improve; 18 and phi 1 help KL more (one seed)",
     [("rescale7", "rescale, scale 7", "rejected"), ("rescale12", "rescale, scale 12, phi 0.7", "chosen"),
      ("rescale18", "rescale, scale 18", "rejected"), ("rescale12_phi1", "rescale 12, phi 1", "rejected"),
      ("rescale12_valid", "pilot: std over clip frames only", "pilot")]),
    ("6  Replicate on seeds", "one seed can mislead: seed 0 alone left the KL gain unproven",
     "the gain holds on all 3 seeds: CLAP +0.028, KL drop 0.12, both intervals clear of 0",
     [("rescale12@pooled", "rescale12, 3 seeds pooled", "chosen")]),
    ("7  Best of 3 seeds", "ranking by the same model that scores rewards its errors (selection bias)",
     "scoring-model ranking +0.070 CLAP, much of it bias; a second checkpoint +0.042; both drop KL by about the same",
     [("rescale12_best3", "ranked by the scoring CLAP", "circular"), ("rescale12_best3_c8998", "ranked by a second CLAP (submitted)", "chosen"),
      ("rescale12_best3_mqa", "pilot: ranked by the judge", "pilot")]),
]


def per_prompt(set_name: str, run: str) -> pd.DataFrame | None:
    path = RESULTS / f"{set_name}__{run}" / "per_prompt.csv"
    return pd.read_csv(path, dtype={"id": str}).set_index("id") if path.exists() else None


def paired(set_name: str, run: str, against: str = "cfg7") -> dict | None:
    seeds = ["", "_seed1", "_seed2"] if run.endswith("@pooled") else [""]
    base = run.removesuffix("@pooled")
    clap, kl = [], []
    for suffix in seeds:
        values, reference = per_prompt(set_name, base + suffix), per_prompt(set_name, against + suffix)
        if values is None or reference is None:
            return None
        clap.append(values["clap_cosine"] - reference["clap_cosine"])
        kl.append(reference["kl_passt_softmax"] - values["kl_passt_softmax"])
    clap, kl = pd.concat(clap, axis=1).mean(axis=1).dropna(), pd.concat(kl, axis=1).mean(axis=1).dropna()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(clap), size=(RESAMPLES, len(clap)))
    row = {"run": run, "n": len(clap)}
    for name, values in (("clap", clap.to_numpy()), ("kl_drop", kl.to_numpy())):
        means = values[picks].mean(axis=1)
        row |= {name: values.mean(), f"{name}_low": np.percentile(means, 2.5), f"{name}_high": np.percentile(means, 97.5)}
    return row


def main():
    parser = argparse.ArgumentParser(description="Decision roadmap: every dev decision as a paired change vs dev cfg7, with its cause and tendency.")
    parser.add_argument("--out", type=Path, default=RESULTS / "analysis" / "figures" / "roadmap_dev.png")
    args = parser.parse_args()

    rows, y = [], 0
    for stage, cause, tendency, options in STAGES:
        rows.append({"kind": "stage", "y": y, "stage": stage, "cause": cause, "tendency": tendency})
        y += 1
        for run, label, role in options:
            result = paired("dev", run)
            rows.append({"kind": "option", "y": y, "label": label, "role": role, **(result or {"run": run})})
            y += 1
        lines = len(textwrap.wrap("cause: " + cause, WRAP)) + len(textwrap.wrap("tendency: " + tendency, WRAP))
        y = max(y, rows[-1 - len(options)]["y"] - 0.45 + LINE * lines + 0.25)
        y += 0.35
    table = pd.DataFrame(rows)
    options = table[table["kind"] == "option"]
    options.drop(columns=["kind", "stage", "cause", "tendency"], errors="ignore").to_csv(args.out.with_suffix(".csv").parent.parent / "roadmap_dev.csv", index=False)

    plt.rcParams.update({"font.family": "Noto Sans", "font.size": 11, "text.color": DARK, "axes.edgecolor": GREY,
                         "xtick.color": DARK, "ytick.color": DARK, "figure.facecolor": LIGHT, "axes.facecolor": LIGHT})
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.1, 1.1, 1.75], wspace=0.05, left=0.01, right=0.99, top=0.9, bottom=0.08)
    label_ax, clap_ax, kl_ax, text_ax = (fig.add_subplot(grid[0, i]) for i in range(4))
    height = y
    colors = {"kept": BLUE, "chosen": BLUE, "rejected": GREY, "circular": ORANGE, "pilot": GREEN}
    for ax in (label_ax, text_ax):
        ax.set_xlim(0, 1)
        ax.axis("off")
    for ax in (label_ax, clap_ax, kl_ax, text_ax):
        ax.set_ylim(height, -1.9)
    for ax, column, title, span in ((clap_ax, "clap", "CLAP gain vs cfg7  (right = better)", (-0.095, 0.085)),
                                    (kl_ax, "kl_drop", "KL-PaSST drop vs cfg7  (right = better)", (-0.65, 0.35))):
        ax.set_xlim(*span)
        ax.axvline(0, color=DARK, lw=1)
        ax.set_yticks([])
        ax.grid(axis="x", color="#dde1e8")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=12, loc="left")
        for _, row in table[table["kind"] == "stage"].iterrows():
            ax.axhspan(row["y"] - 0.5, row["y"] + 0.5, color="#e6eaf0", lw=0)
        route = []
        for _, row in options.iterrows():
            color = colors[row["role"]]
            if pd.isna(row.get(column)):
                ax.text(0, row["y"], "pending", ha="center", va="center", fontsize=9, color=color, style="italic")
                continue
            low, high = np.clip([row[f"{column}_low"], row[f"{column}_high"]], *span)
            ax.hlines(row["y"], low, high, color=color, lw=2.2, alpha=0.9)
            ax.plot(np.clip(row[column], *span), row["y"], "s" if row["role"] == "kept" else "o", color=color,
                    ms=7 if row["role"] in ("kept", "chosen") else 5.5,
                    mfc=color if row["role"] != "circular" else LIGHT, mew=1.6)
            if row["role"] in ("kept", "chosen"):
                route.append((row[column], row["y"]))
        ax.plot([0] + [x for x, _ in route], [-1.15] + [r for _, r in route], color=BLUE, lw=1.6, alpha=0.45, zorder=0)
        ax.plot(0, -1.15, "s", color=DARK, ms=6)
    label_ax.text(0.02, -1.15, "start: cfg7, caption prompt, 100 steps", va="center", fontsize=11, weight="bold")
    for _, row in table.iterrows():
        if row["kind"] == "stage":
            label_ax.text(0.02, row["y"], row["stage"], va="center", fontsize=11.5, weight="bold")
            cause = textwrap.fill("cause: " + row["cause"], WRAP)
            text_ax.text(0.02, row["y"] - 0.45, cause, va="top", fontsize=10.5, color=DARK, linespacing=1.15)
            tendency = textwrap.fill("tendency: " + row["tendency"], WRAP)
            text_ax.text(0.02, row["y"] - 0.45 + LINE * (cause.count("\n") + 1), tendency, va="top", fontsize=10.5, color=BLUE, linespacing=1.15)
        else:
            label_ax.text(0.06, row["y"], row["label"], va="center", fontsize=10.5,
                          color=colors[row["role"]] if row["role"] != "rejected" else DARK, weight="bold" if row["role"] == "chosen" else None)
    fig.suptitle("How each dev decision moved the score (paired per caption vs cfg7, 95% bootstrap intervals)", x=0.01, ha="left", fontsize=15, weight="bold")
    handles = [Line2D([], [], color=BLUE, marker="s", lw=1.6, alpha=0.8, label="route: kept or chosen"),
               Line2D([], [], color=GREY, marker="o", lw=2, label="rejected option"),
               Line2D([], [], color=ORANGE, marker="o", mfc=LIGHT, lw=2, label="circular (ranked by the scoring model)"),
               Line2D([], [], color=GREEN, marker="o", lw=2, label="pilot")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=4, frameon=False, fontsize=11)
    fig.text(0.99, 0.022,  "dev set, 100 MusicCaps captions; source: results/<run>/per_prompt.csv", color=GREY, fontsize=10, ha="right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=LIGHT)
    print(options[["run", "role", "clap", "clap_low", "clap_high", "kl_drop", "kl_drop_low", "kl_drop_high"]].round(3).to_string(index=False))
    print("->", args.out)


if __name__ == "__main__":
    main()
