import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, boot_ci, fmt, per_prompt, sign_flip_p

for setting in ("cfg7", "rescale12"):
    seeds = [f"dev__{setting}", f"dev__{setting}_seed1", f"dev__{setting}_seed2"]
    frames = [per_prompt(s) for s in seeds]
    print(f"\n===== {setting}")
    for s, f in zip(seeds, frames):
        r = stats.pearsonr(f["clap_cosine"], f["clap8998_cosine"])[0]
        rho = stats.spearmanr(f["clap_cosine"], f["clap8998_cosine"])[0]
        print(f"{s}: per-item corr 90.14 vs 89.98 pearson {r:.3f} spearman {rho:.3f}")
    d9014, d8998 = [], []
    for i in range(3):
        for j in range(i + 1, 3):
            d9014.append((frames[i]["clap_cosine"] - frames[j]["clap_cosine"]).to_numpy())
            d8998.append((frames[i]["clap8998_cosine"] - frames[j]["clap8998_cosine"]).to_numpy())
    d9014, d8998 = np.concatenate(d9014), np.concatenate(d8998)
    slope = np.polyfit(d8998, d9014, 1)[0]
    print(f"seed-to-seed differences (what ranking sees): pearson {stats.pearsonr(d9014, d8998)[0]:.3f}, slope 90.14 on 89.98 {slope:.3f}")

    c9014 = np.stack([f["clap_cosine"].to_numpy() for f in frames], 1)
    c8998 = np.stack([f["clap8998_cosine"].to_numpy() for f in frames], 1)
    kl = np.stack([f["kl_passt_softmax"].to_numpy() for f in frames], 1)
    klp = np.stack([f["kl_panns_softmax"].to_numpy() for f in frames], 1)
    pick_same, pick_cross = c9014.argmax(1), c8998.argmax(1)
    rows = np.arange(100)
    agree = (pick_same == pick_cross).mean()
    print(f"picks agree between rankers on {agree:.2f} of prompts (chance 1/3 if independent)")
    print(f"pick counts by seed: same-ckpt {np.bincount(pick_same, minlength=3)}, cross-ckpt {np.bincount(pick_cross, minlength=3)}; "
          f"chi2 uniform p (cross) {stats.chisquare(np.bincount(pick_cross, minlength=3)).pvalue:.3f}")
    mean_seed = c9014.mean(1)
    for label, pick in (("same-ckpt", pick_same), ("cross-ckpt", pick_cross)):
        g_seed0 = c9014[rows, pick] - c9014[:, 0]
        g_mean = c9014[rows, pick] - mean_seed
        g8998 = c8998[rows, pick] - c8998.mean(1)
        kl_mean = kl[rows, pick] - kl.mean(1)
        klp_mean = klp[rows, pick] - klp.mean(1)
        print(f"{label}: dCLAP90.14 vs seed0 {fmt(boot_ci(g_seed0))}; vs seed-mean (random pick) {fmt(boot_ci(g_mean))}; "
              f"dCLAP89.98 vs seed-mean {fmt(boot_ci(g8998))}; dKL-PaSST vs seed-mean {fmt(boot_ci(kl_mean))}; dKL-PANNs vs seed-mean {fmt(boot_ci(klp_mean))}")
    oracle_kl = kl.min(1) - kl.mean(1)
    print(f"for scale: oracle pick by KL itself gives dKL {fmt(boot_ci(oracle_kl))}")
    print(f"seed-0 KL minus seed-mean KL {fmt(boot_ci(kl[:, 0] - kl.mean(1)))} (is seed 0 an unlucky baseline?)")
    print(f"seed-0 CLAP minus seed-mean CLAP {fmt(boot_ci(c9014[:, 0] - mean_seed))}")
    # winner's curse: expected max of 3 of the selecting score vs what an independent rescoring sees
    sel_gain_8998 = (c8998.max(1) - c8998.mean(1)).mean()
    transfer = (c9014[rows, pick_cross] - mean_seed).mean()
    print(f"selected 89.98 gain over seed mean {sel_gain_8998:+.4f}; transfers to 90.14 {transfer:+.4f} (ratio {transfer / sel_gain_8998:.2f}); "
          f"same-ckpt self gain {(c9014.max(1) - mean_seed).mean():+.4f}")

# MQAScore of both best3 versions vs seed 0 (only seed 0 has MQAScore)
print("\n===== independent judges, both rankers, vs seed 0 (MQAScore exists only for seed 0)")
for setting in ("cfg7", "rescale12"):
    base = per_prompt(f"dev__{setting}")
    for name in (f"dev__{setting}_best3", f"dev__{setting}_best3_c8998"):
        f = per_prompt(name)
        sel = pd.read_csv(RESULTS / name / "selection.csv").set_index("id")
        changed = (sel["chosen_run"] != setting).to_numpy()
        out = []
        for m in ("clap_cosine", "kl_passt_softmax", "kl_panns_softmax", "mqascore"):
            d = (f[m] - base[m]).to_numpy()
            out.append(f"{m[:8]} {fmt(boot_ci(d))}")
            if m == "mqascore":
                out.append(f"mqa on changed prompts only (n={changed.sum()}) {fmt(boot_ci(d[changed]))}")
        print(f"{name} - {setting}: " + "; ".join(out))
