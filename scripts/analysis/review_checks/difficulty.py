import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import per_prompt

SIX = ["dev__cfg7", "dev__cfg7_seed1", "dev__cfg7_seed2", "dev__rescale12", "dev__rescale12_seed1", "dev__rescale12_seed2"]
frames = {run: per_prompt(run) for run in SIX + ["dev__real"]}
clap = pd.DataFrame({run: frames[run]["clap_cosine"] for run in SIX})
vocal = pd.read_csv(Path(__file__).parent / "vocal_groups_dev.csv").set_index("id").sort_index()["tag"]

print("== rescale12 per-prompt CLAP gain (seed 0) against cfg7 per-prompt CLAP")
gain = frames["dev__rescale12"]["clap_cosine"] - frames["dev__cfg7"]["clap_cosine"]
other_cfg = (frames["dev__cfg7_seed1"]["clap_cosine"] + frames["dev__cfg7_seed2"]["clap_cosine"]) / 2
for label, base in (("cfg7 seed 0 (shared noise)", frames["dev__cfg7"]["clap_cosine"]), ("cfg7 seeds 1-2 mean", other_cfg)):
    pearson, pp = stats.pearsonr(gain, base)
    spearman, sp = stats.spearmanr(gain, base)
    print(f"gain vs {label}: Pearson {pearson:+.2f} (p {pp:.3f}), Spearman {spearman:+.2f} (p {sp:.3f})")
gain_pooled = np.mean([frames[f"dev__rescale12{s}"]["clap_cosine"] - frames[f"dev__cfg7{s}"]["clap_cosine"] for s in ("", "_seed1", "_seed2")], axis=0)
print(f"pooled 3-seed gain vs mean of all 6 runs: Pearson {stats.pearsonr(gain_pooled, clap.mean(axis=1))[0]:+.2f}")

print("\n== how much of per-item CLAP variance is the prompt, over the 6 runs (cfg7 and rescale12, seeds 0-2)")
values = clap.to_numpy()
total_variance = values.var(ddof=1)
between = values.mean(axis=1).var(ddof=1)
within_mean = values.var(axis=1, ddof=1).mean()
print(f"share = variance of prompt means / total variance: {between / total_variance:.2f}")
print(f"one-way variance components: within {within_mean:.5f}, between-prompt variance of means {between:.5f}")
k = values.shape[1]
print(f"ICC(1) = (var of means - within/k) / (var of means + within*(k-1)/k): {(between - within_mean / k) / (between + within_mean * (k - 1) / k):.2f}")
print(f"1 - mean within-prompt variance / total variance: {1 - within_mean / total_variance:.2f}")
grand = values.mean()
prompt_ss = k * ((values.mean(axis=1) - grand) ** 2).sum()
run_ss = len(values) * ((values.mean(axis=0) - grand) ** 2).sum()
error_ss = ((values - grand) ** 2).sum() - prompt_ss - run_ss
prompt_ms, error_ms = prompt_ss / (len(values) - 1), error_ss / ((len(values) - 1) * (k - 1))
print(f"ICC(3,1), two-way with the run as a fixed effect: {(prompt_ms - error_ms) / (prompt_ms + (k - 1) * error_ms):.2f}")
rho = [stats.spearmanr(clap[a], clap[b])[0] for a, b in combinations(SIX, 2)]
print(f"mean pairwise Spearman between runs: {np.mean(rho):.2f} (min {min(rho):.2f}, max {max(rho):.2f}, {len(rho)} pairs)")

print("\n== generated vs real, per prompt")
real = frames["dev__real"]
print(f"CLAP: real vs cfg7 Spearman {stats.spearmanr(real['clap_cosine'], frames['dev__cfg7']['clap_cosine'])[0]:+.2f}, "
      f"real vs mean of the 6 runs Spearman {stats.spearmanr(real['clap_cosine'], clap.mean(axis=1))[0]:+.2f}, Pearson {np.corrcoef(real['clap_cosine'], clap.mean(axis=1))[0, 1]:+.2f}")
print(f"MQAScore: real vs cfg7 Spearman {stats.spearmanr(real['mqascore'], frames['dev__cfg7']['mqascore'], nan_policy='omit')[0]:+.2f}")

print("\n== hardest prompts (lowest mean CLAP over the 6 runs)")
hardest = clap.mean(axis=1).sort_values().index[:10]
print(f"ids {list(map(int, hardest))}; vocal captions among them {int(vocal.loc[hardest].sum())} of 10 (vocal share overall {vocal.mean():.0%})")
hardest_real = real["clap_cosine"].sort_values().index[:10]
print(f"hardest 10 by the real clips' own CLAP: vocal captions {int(vocal.loc[hardest_real].sum())} of 10")
n_tags = frames["dev__real"]["n_tags"]
print(f"Spearman of prompt-mean CLAP with tag count {stats.spearmanr(clap.mean(axis=1), n_tags, nan_policy='omit')[0]:+.2f}")
