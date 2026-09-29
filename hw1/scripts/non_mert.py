"""Is there real signal outside MERT? Uses the cached out-of-fold log-probs (features/cv_{A,B}.npz), no training.

1. Each non-MERT component alone vs chance, bootstrap CI.
2. Every combination of the non-MERT pool: equal-weight and temperature-calibrated (T fitted on OOF) fusion.
3. Leave-one-out inside the full non-MERT fusion: the unique contribution of each component, paired CI.
4. Complementarity: on clips where MERT's top-1 is wrong, the component's mean rank of the true class vs the
   uniform expectation (3.5 of 6); and the top-1 hit rate there vs chance (1/6).
Writes results/non_mert.json.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HW1 = Path(__file__).resolve().parents[1]
POOL = ["handcrafted", "clap", "mixbalance", "lang_vocals", "lang_mixture"]


def norm(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def per_clip(lp, y):
    o = np.argsort(-lp, 1)[:, :3]
    return (o[:, 0] == y) + 0.5 * (o == y[:, None]).any(1)


def ci(v, rng, n=2000):
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for key in ("A", "B"):
        z = dict(np.load(HW1 / "features" / f"cv_{key}.npz"))
        dem = HW1 / "features" / f"cv_{key}_demucs.npz"
        pool = list(POOL)
        if dem.exists():
            z["oof_demucs_all"] = dict(np.load(dem))["oof_demucs_all"]
            pool.append("demucs_all")
        y = z["y_train"]
        oof = {c: norm(z[f"oof_{c}"]) for c in pool + ["mert_mixture"]}
        T = {c: minimize_scalar(lambda t: -norm(oof[c] * t)[np.arange(len(y)), y].mean(), bounds=(0.05, 20),
                                method="bounded").x for c in oof}
        cal = {c: norm(oof[c] * T[c]) for c in oof}
        r = {"n": int(len(y)), "chance_S": 0.4167, "temperature": {c: round(float(t), 2) for c, t in T.items()}}

        r["alone"] = {c: {"S": round(float(per_clip(oof[c], y).mean()), 4), "ci95": ci(per_clip(oof[c], y), rng)}
                      for c in pool}
        rows = []
        for k in range(1, len(pool) + 1):
            for combo in itertools.combinations(pool, k):
                rows.append({"combo": "+".join(combo), "n": k,
                             "S_eq": round(float(per_clip(np.mean([oof[c] for c in combo], 0), y).mean()), 4),
                             "S_cal": round(float(per_clip(np.mean([cal[c] for c in combo], 0), y).mean()), 4)})
        r["combos_top"] = sorted(rows, key=lambda d: -d["S_cal"])[:8]

        full = per_clip(np.mean([cal[c] for c in pool], 0), y)
        r["full_pool_cal"] = {"S": round(float(full.mean()), 4), "ci95": ci(full, rng)}
        r["leave_one_out_cal"] = {}
        for c in pool:
            rest = per_clip(np.mean([cal[o] for o in pool if o != c], 0), y)
            r["leave_one_out_cal"][c] = {"drop_cost": round(float((full - rest).mean()), 4), "ci95": ci(full - rest, rng)}

        wrong = np.argmax(oof["mert_mixture"], 1) != y
        r["on_mert_wrong"] = {"n": int(wrong.sum())}
        for c in pool:
            rank = (np.argsort(-oof[c][wrong], 1) == y[wrong, None]).argmax(1) + 1
            hit = (rank == 1).astype(float)
            r["on_mert_wrong"][c] = {"mean_rank_true": round(float(rank.mean()), 3), "rank_ci95": ci(rank.astype(float), rng),
                                     "top1": round(float(hit.mean()), 4), "top1_ci95": ci(hit, rng)}
        report[key] = r

        print(f"\n== {key} (n={len(y)}, chance S 0.417; non-MERT pool: {', '.join(pool)})")
        print("alone:", {c: (v["S"], v["ci95"]) for c, v in r["alone"].items()})
        print("full pool (calibrated):", r["full_pool_cal"], "| MERT alone", round(float(per_clip(oof['mert_mixture'], y).mean()), 4))
        print("leave-one-out cost:", {c: (v["drop_cost"], v["ci95"]) for c, v in r["leave_one_out_cal"].items()})
        print("best combos:", [(d["combo"], d["S_eq"], d["S_cal"]) for d in r["combos_top"][:5]])
        print(f"on the {wrong.sum()} clips MERT gets wrong (uniform: rank 3.5, top-1 0.167):",
              {c: (v["mean_rank_true"], v["rank_ci95"], v["top1"]) for c, v in r["on_mert_wrong"].items() if c != "n"})
    (HW1 / "results" / "non_mert.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
