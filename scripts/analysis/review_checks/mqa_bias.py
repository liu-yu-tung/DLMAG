import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, boot_ci, fmt, per_prompt

ATTRIBUTES = ["genre", "instrument", "mood_theme", "vocal", "context"]
GROUP_ATTRIBUTES = [a for a in ATTRIBUTES if a != "vocal"]
runs = sorted(p.parent.name for p in RESULTS.glob("dev__*/mqa_per_item.csv"))
frames = {run: per_prompt(run) for run in runs}
vocal = pd.read_csv(Path(__file__).parent / "vocal_groups_dev.csv").set_index("id").sort_index()["tag"].to_numpy()

print("== Spearman of MQAScore with tag count, per run (real clips included)")
for run, frame in frames.items():
    rho, p = stats.spearmanr(frame["n_tags"], frame["mqascore"], nan_policy="omit")
    print(f"{run}: rho {rho:+.3f} p {p:.4f} n {len(frame)}")

print("\n== |rho| of MQAScore with loudness proxies, generated runs (real clips have no audio stats here)")
for run, frame in frames.items():
    for column in ("rms", "peak", "clip_fraction"):
        if column not in frame:
            print(f"{run}: {column} missing")
            continue
        rho, p = stats.spearmanr(frame[column], frame["mqascore"], nan_policy="omit")
        print(f"{run}: {column} rho {rho:+.3f} p {p:.3f}")

print("\n== real minus cfg7, paired per prompt, by MQAScore attribute (mean over a caption's tags of that attribute)")
real, cfg7 = frames["dev__real"], frames["dev__cfg7"]
d = (real["mqascore"] - cfg7["mqascore"]).to_numpy()
print(f"overall mqascore: {fmt(boot_ci(d))}  real higher on {np.nanmean(d[~np.isnan(d)] > 0):.0%} of the {int((~np.isnan(d)).sum())} scored prompts; means {real['mqascore'].mean():.3f} vs {cfg7['mqascore'].mean():.3f}")
for attribute in ATTRIBUTES:
    if False:
        pass
    d = (real[f"{attribute}_mean"] - cfg7[f"{attribute}_mean"]).to_numpy()
    n = int((~np.isnan(d)).sum())
    print(f"{attribute}: {fmt(boot_ci(d))} n {n}; means {real[f'{attribute}_mean'].mean():.3f} vs {cfg7[f'{attribute}_mean'].mean():.3f}")

print("\n== vocal captions minus other captions (tag-based split, n_vocal = %d)" % int(vocal.sum()))


def group_gap(values: np.ndarray, label: str, digits: int = 3):
    a, b = values[vocal], values[~vocal]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    welch = stats.ttest_ind(a, b, equal_var=False).pvalue
    mann = stats.mannwhitneyu(a, b).pvalue
    print(f"{label}: {a.mean() - b.mean():+.{digits}f} (vocal n {len(a)}, other n {len(b)}) Welch p {welch:.4f}, Mann-Whitney p {mann:.4f}")


group_gap(real["n_tags"].to_numpy(float), "tag count")
group_gap(real["mqascore"].to_numpy(), "real MQAScore")
for attribute in GROUP_ATTRIBUTES:
    group_gap(real[f"{attribute}_mean"].to_numpy(), f"real {attribute}")
group_gap(real["clap_cosine"].to_numpy(), "real CLAP")
group_gap(cfg7["mqascore"].to_numpy(), "cfg7 MQAScore")
for attribute in GROUP_ATTRIBUTES:
    group_gap(cfg7[f"{attribute}_mean"].to_numpy(), f"cfg7 {attribute}")

print("\n== real-minus-cfg7 MQAScore gap by vocal group (interaction)")
gap = (real["mqascore"] - cfg7["mqascore"]).to_numpy()
print(f"vocal {np.nanmean(gap[vocal]):+.3f}, other {np.nanmean(gap[~vocal]):+.3f}")
