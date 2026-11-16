import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import boot_ci, fmt, per_prompt

g = pd.read_csv(Path(__file__).parent / "vocal_groups_dev.csv").set_index("id").sort_index()
tag = g["tag"].to_numpy()
corrected = tag.copy()
corrected[[39, 40, 6, 59]] = True
corrected[94] = False
keyword = (g["kw"] & ~(g["neg"] & ~g["tag"])).to_numpy()
splits = {"tag (as reported)": tag, "corrected (+39,40,6,59 -94)": corrected, "keyword minus negated": keyword}
rng = np.random.default_rng(0)


def interaction(d, group, perms=20000):
    d = np.asarray(d, float)
    observed = d[group].mean() - d[~group].mean()
    null = np.empty(perms)
    for k in range(perms):
        p = rng.permutation(group)
        null[k] = d[p].mean() - d[~p].mean()
    pval = ((np.abs(null) >= abs(observed)).sum() + 1) / (perms + 1)
    boots = []
    vi, ni = np.where(group)[0], np.where(~group)[0]
    r2 = np.random.default_rng(1)
    for _ in range(2000):
        boots.append(d[r2.choice(vi, len(vi))].mean() - d[r2.choice(ni, len(ni))].mean())
    return observed, np.percentile(boots, 2.5), np.percentile(boots, 97.5), pval


def run(name):
    return per_prompt(f"dev__{name}")


comparisons = {
    "rescale12 - cfg7 (seed 0)": [("rescale12", "cfg7")],
    "rescale12 - cfg7 (seed 1)": [("rescale12_seed1", "cfg7_seed1")],
    "rescale12 - cfg7 (seed 2)": [("rescale12_seed2", "cfg7_seed2")],
    "rescale12 - cfg7 (pooled 3 seeds)": [("rescale12", "cfg7"), ("rescale12_seed1", "cfg7_seed1"), ("rescale12_seed2", "cfg7_seed2")],
    "cfg12 - cfg7": [("cfg12", "cfg7")],
    "rescale18 - cfg7": [("rescale18", "cfg7")],
    "rescale12_phi1 - cfg7": [("rescale12_phi1", "cfg7")],
    "best3_c8998 - cfg7": [("rescale12_best3_c8998", "cfg7")],
    "best3_c8998 - rescale12": [("rescale12_best3_c8998", "rescale12")],
    "cfg7_seed mean - cfg7 seed 0 (null check)": [("cfg7_seed1", "cfg7"), ("cfg7_seed2", "cfg7")],
}
for split_name, group in splits.items():
    print(f"\n######## split: {split_name}: vocal n={group.sum()}")
    for label, pairs in comparisons.items():
        for metric in ("clap_cosine", "kl_passt_softmax"):
            d = np.mean([(run(a)[metric] - run(b)[metric]).to_numpy() for a, b in pairs], axis=0)
            v, n = boot_ci(d[group]), boot_ci(d[~group])
            obs, lo, hi, p = interaction(d, group, perms=5000 if split_name != "tag (as reported)" else 20000)
            print(f"{label} {metric[:4]}: vocal {fmt(v)} | no_vocal {fmt(n)} | interaction {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] perm p {p:.3f}")

print("\n######## real clips by group (tag split and corrected)")
real, cfg7 = run("real"), run("cfg7")
for split_name, group in list(splits.items())[:2]:
    for metric in [m for m in ("clap_cosine", "clap8998_cosine", "mqascore", "mqa_genre", "genre_mean", "instrument_mean", "mood_theme_mean", "context_mean") if m in real and m in cfg7]:
        x = real[metric].to_numpy()
        obs, lo, hi, p = interaction(x, group, perms=20000)
        y = cfg7[metric].to_numpy()
        obs2, lo2, hi2, p2 = interaction(y, group, perms=20000)
        print(f"{split_name} {metric}: real vocal-minus-other {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] p {p:.3f}; cfg7 vocal-minus-other {obs2:+.3f} [{lo2:+.3f}, {hi2:+.3f}] p {p2:.4f}")
    gap = (real["mqascore"] - cfg7["mqascore"]).to_numpy()
    obs, lo, hi, p = interaction(gap, group, perms=20000)
    print(f"{split_name}: real-minus-cfg7 MQAScore gap, vocal minus other {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] p {p:.3f}")
    print(f"{split_name}: cfg7 KL-PaSST level vocal {cfg7['kl_passt_softmax'][group].mean():.3f} vs other {cfg7['kl_passt_softmax'][~group].mean():.3f}")
