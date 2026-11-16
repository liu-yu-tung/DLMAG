import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, boot_ci, fmt, holm, per_prompt, sign_flip_p

BASE = "dev__cfg7"
runs = sorted(p.name for p in RESULTS.glob("dev__*") if (p / "per_prompt.csv").exists() and p.name not in (BASE, "dev__real"))
base = per_prompt(BASE)
pvals, rows = {}, []
for run in runs:
    frame = per_prompt(run)
    for metric in ("clap_cosine", "kl_passt_softmax"):
        d = (frame[metric] - base[metric]).to_numpy()
        ci0 = boot_ci(d)
        flips = []
        for s in range(1, 21):
            ci = boot_ci(d, seed=s)
            flips.append(ci[1] > 0 or ci[2] < 0)
        big = boot_ci(d, resamples=100000, seed=123)
        t_low, t_high = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        p = sign_flip_p(d)
        pvals[(run, metric)] = p
        rows.append((run, metric, ci0, (ci0[1] > 0 or ci0[2] < 0), sum(flips), big, (t_low, t_high), p))

adjusted = holm(pvals)
print(f"family size {len(pvals)} (all dev runs vs cfg7, CLAP and KL)")
print("run | metric | seed-0 CI | excl0 | seeds1-20 excl0 | 1e5 CI | t CI | signflip p | Holm p")
for run, metric, ci0, ex, nflip, big, t, p in rows:
    flag = "" if (ex and nflip == 20) or (not ex and nflip == 0) else "  <-- FRAGILE"
    holm_flag = "" if not ex or adjusted[(run, metric)] < 0.05 else "  <-- not after Holm"
    print(f"{run} | {metric[:4]} | {fmt(ci0)} | {ex} | {nflip}/20 | [{big[1]:+.3f}, {big[2]:+.3f}] | [{t[0]:+.3f}, {t[1]:+.3f}] | {p:.4f} | {adjusted[(run, metric)]:.4f}{flag}{holm_flag}")

print("\n== seed-matched rescale12 vs cfg7, and pooled over seeds")
seeds = [("", ""), ("_seed1", "_seed1"), ("_seed2", "_seed2")]
pooled = {}
for metric in ("clap_cosine", "kl_passt_softmax", "kl_panns_softmax", "clap8998_cosine"):
    diffs = []
    for a, b in seeds:
        r, c = per_prompt(f"dev__rescale12{a}"), per_prompt(f"dev__cfg7{b}")
        if metric not in r or metric not in c:
            continue
        d = (r[metric] - c[metric]).to_numpy()
        diffs.append(d)
        print(f"seed{a or '_seed0'} {metric}: {fmt(boot_ci(d))}  signflip p {sign_flip_p(d):.4f}")
    if len(diffs) == 3:
        mean_d = np.mean(diffs, axis=0)
        print(f"POOLED (per-prompt mean of 3 seed pairs) {metric}: {fmt(boot_ci(mean_d))}  signflip p {sign_flip_p(mean_d):.5f}")

print("\n== head-to-heads that the tie-break by 'improve' depends on")
for a, b in [("dev__rescale12_phi1", "dev__rescale12"), ("dev__rescale18", "dev__rescale12"), ("dev__rescale12", "dev__cfg12"),
             ("dev__rescale12", "dev__apg12"), ("dev__apg12", "dev__cfg12"), ("dev__rescale7", "dev__cfg7"),
             ("dev__interval_early30", "dev__cfg7"), ("dev__interval_early", "dev__cfg7")]:
    x, y = per_prompt(a), per_prompt(b)
    out = []
    for metric in ("clap_cosine", "kl_passt_softmax", "mqascore"):
        if metric in x and metric in y:
            d = (x[metric] - y[metric]).to_numpy()
            out.append(f"{metric[:4]} {fmt(boot_ci(d))} p={sign_flip_p(d):.3f}")
    print(f"{a} - {b}: " + "; ".join(out))
