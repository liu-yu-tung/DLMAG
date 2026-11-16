import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, per_prompt

METRICS = {"clap_cosine": True, "kl_passt_softmax": False, "kl_panns_softmax": False, "mqascore": True}
PICK = "rescale12_best3_c8998"


def load(session: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "listening" / f"{session}.csv")


print("== listener-vs-metric agreement against the marginals (Fisher exact, two-sided)")
print("session | metric | pairs | agree | expected agree under independence | Fisher p | recorded (agree/pairs)")
for session in ("hw3_s1", "hw3_s2"):
    table = load(session)
    decided = table[table["strength"] > 0]
    recorded = pd.read_csv(RESULTS / "listening" / f"{session}_agreement.csv").set_index("metric")
    frames = {run: per_prompt(f"dev__{run}") for run in sorted(set(table["run_a"]) | set(table["run_b"]))}
    run_first, run_second = decided["run_a"].iloc[0], decided["run_b"].iloc[0]
    for metric, higher in METRICS.items():
        gaps = np.array([frames[run_first].loc[pid, metric] - frames[run_second].loc[pid, metric] for pid in decided["prompt_id"]])
        usable = ~np.isnan(gaps) & (gaps != 0)
        metric_first = ((gaps > 0) if higher else (gaps < 0))[usable]
        ear_first = (decided["chosen_run"] == run_first).to_numpy()[usable]
        n = int(usable.sum())
        agree = int((metric_first == ear_first).sum())
        pe, pm = ear_first.mean(), metric_first.mean()
        expected = n * (pe * pm + (1 - pe) * (1 - pm))
        cell = [[int((ear_first & metric_first).sum()), int((ear_first & ~metric_first).sum())],
                [int((~ear_first & metric_first).sum()), int((~ear_first & ~metric_first).sum())]]
        p = stats.fisher_exact(cell)[1]
        rec = recorded.loc[metric]
        print(f"{session} | {metric} | {n} | {agree} | {expected:.1f} | {p:.4f} | {int(rec['agree'])}/{int(rec['pairs'])}")

print("\n== side bias")
for session in ("hw3_s1", "hw3_s2"):
    decided = load(session).query("strength > 0")
    side_b = int((decided["chosen_side"] == "B").sum())
    print(f"{session}: B chosen {side_b} of {len(decided)} decided, binomial p {stats.binomtest(side_b, len(decided)).pvalue:.3f}")
decided_both = pd.concat([load("hw3_s1"), load("hw3_s2")]).query("strength > 0")
side_b = int((decided_both["chosen_side"] == "B").sum())
print(f"both sessions: B chosen {side_b} of {len(decided_both)} decided, binomial p {stats.binomtest(side_b, len(decided_both)).pvalue:.3f}")

print("\n== session 1: the best-of-3 pick chosen, by the side it was on")
decided = load("hw3_s1").query("strength > 0")
on_b = decided[decided["run_b"] == PICK]
on_a = decided[decided["run_a"] == PICK]
print(f"pick on B: chosen {int((on_b['chosen_run'] == PICK).sum())} of {len(on_b)}")
print(f"pick on A: chosen {int((on_a['chosen_run'] == PICK).sum())} of {len(on_a)}")
