"""TASK B auto judge: flag each continuation (bars 9-32) on four checks taken from the T3 listening comments, each
against the real held-out range (worse than the real 10th / 90th percentile = flagged). Real reference: held-out
32-bar crops, their first 8 bars as the prompt. Also scores the T3 pairs to see if the judge agrees with the ear.

uv run python scripts/task_b_judge.py outputs/task_b_sweep/* --pairs results/listening/t3_continuations.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hw2.grade_features import piece_features
from hw2.remi import HW2_ROOT

CHECKS = {
    "key_drift": ("opening_key_fit_min4", "low"),
    "clashes": ("semitone_clash_share", "high"),
    "gap": ("longest_onset_gap_beats", "high"),
    "loop": ("loop_run", "high"),
}
REAL = ["outputs/ref/heldout_crop32", "outputs/ref/heldout_open32"]


def score(paths: list[Path], name: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        f = piece_features(path, first_bar=8)
        rows.append({"set": name, "piece": path.stem, "file": str(path.relative_to(HW2_ROOT)),
                     **{c: f[col] for c, (col, _) in CHECKS.items()}})
    return pd.DataFrame(rows)


def flag(df: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, (_, side) in CHECKS.items():
        limit = np.nanpercentile(real[c], 10 if side == "low" else 90)
        df[f"{c}_flag"] = df[c] < limit if side == "low" else df[c] > limit
    df["flags"] = df[[f"{c}_flag" for c in CHECKS]].sum(axis=1)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--pairs", type=str, help="listening session csv: positive pref = side B won")
    p.add_argument("--out", type=str, default="results/task_b_judge.csv")
    args = p.parse_args()

    real = score([m for r in REAL for m in sorted((HW2_ROOT / r).glob("*.mid"))], "real_heldout")
    sets = [score(sorted(Path(d).resolve().glob("*.mid")), f"{Path(d).parent.name}/{Path(d).name}") for d in args.dirs]
    pieces = flag(pd.concat([real, *sets]), real)
    pieces.to_csv(args.out, index=False)

    summary = pieces.groupby("set").agg(pieces=("piece", "size"), no_flag=("flags", lambda s: (s == 0).mean()),
                                        **{c: (f"{c}_flag", "mean") for c in CHECKS}).round(3)
    summary.to_csv(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"))
    print(summary.to_string())

    if args.pairs:
        ear_check(pd.read_csv(args.pairs), real, Path(args.out))


def ear_check(rated: pd.DataFrame, real: pd.DataFrame, out: Path):
    dirs = sorted({s for side in "ab" for s in rated[f"source_{side}"]})
    judged = flag(pd.concat([score(sorted((HW2_ROOT / d).glob("*.mid")), d) for d in dirs]), real)
    metrics = pd.concat([pd.read_csv(HW2_ROOT / d / "metrics.csv", dtype={"song_name": str}).assign(set=d) for d in dirs])
    judged = judged.merge(metrics.rename(columns={"song_name": "piece"}), on=["set", "piece"])
    base = pd.read_csv(HW2_ROOT / "results/baselines.csv").set_index("set").loc["real_crop32"]
    target = {**{c: real[c].median() for c in CHECKS}, **{m: base[f"{m}_mean"] for m in ["H1", "H4", "GS"]}}
    cost = {"flags": judged["flags"], **{m: (judged[m] - t).abs() for m, t in target.items()}}
    judged = judged.assign(**{f"cost_{m}": v for m, v in cost.items()}).set_index(["set", "piece"])
    rows = []
    for r in rated.itertuples():
        a, b = (judged.loc[(getattr(r, f"source_{s}"), Path(getattr(r, f"file_{s}")).stem)] for s in "ab")
        for m in cost:
            ca, cb = a[f"cost_{m}"], b[f"cost_{m}"]
            call = "tie" if np.isclose(ca, cb) else "b" if cb < ca else "a"
            rows.append({"pair": r.pair, "measure": m, "cost_a": ca, "cost_b": cb, "call": call, "ear": "b" if r.pref > 0 else "a"})
    check = pd.DataFrame(rows)
    check["agree"] = np.where(check.call == "tie", np.nan, check.call == check.ear)
    check.to_csv(out.with_name(out.stem + "_vs_ear.csv"), index=False)
    table = check.groupby("measure", sort=False).agg(agree=("agree", "sum"), called=("agree", "count"),
                                                    ties=("call", lambda s: (s == "tie").sum()))
    print(table.astype(int).to_string())


if __name__ == "__main__":
    main()
