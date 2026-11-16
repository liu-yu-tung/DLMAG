import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, per_prompt

generated_with_mqa = sorted(p.parent.name for p in RESULTS.glob("dev__*/mqa_per_item.csv") if p.parent.name != "dev__real")
all_generated = sorted(p.parent.name for p in RESULTS.glob("dev__*/per_prompt.csv") if p.parent.name != "dev__real")
frames = {run: per_prompt(run) for run in all_generated}
# the settings the review's ranges cover; the best-of-N runs are selected on CLAP, so they are listed separately
CORE = ["dev__cfg3", "dev__cfg7", "dev__cfg12", "dev__rescale12"]


def rho(a, b):
    r, p = stats.spearmanr(a, b, nan_policy="omit")
    return r, p


print("== within a run: Spearman of per-prompt CLAP with MQAScore")
values = []
for run in generated_with_mqa:
    r, p = rho(frames[run]["clap_cosine"], frames[run]["mqascore"])
    values.append(r)
    print(f"{run}: {r:+.3f} (p {p:.3f})")
core_values = [rho(frames[r]["clap_cosine"], frames[r]["mqascore"])[0] for r in CORE]
print(f"range over all {len(values)} runs {min(values):+.3f} to {max(values):+.3f}; over the 4 non-selection runs {min(core_values):+.3f} to {max(core_values):+.3f}")

print("\n== per-prompt change between two runs: Spearman of the CLAP change with the MQAScore change / the KL change")
pairs = [(a, "dev__cfg7") for a in generated_with_mqa if a != "dev__cfg7"] + [("dev__rescale12_best3_c8998", "dev__rescale12"), ("dev__rescale12_best3", "dev__rescale12")]
mqa_values = []
for a, b in pairs:
    dc = frames[a]["clap_cosine"] - frames[b]["clap_cosine"]
    dm = frames[a]["mqascore"] - frames[b]["mqascore"]
    r, p = rho(dc, dm)
    mqa_values.append(r)
    print(f"{a} - {b}: CLAP vs MQAScore {r:+.3f} (p {p:.3f})")
core_mqa = [rho(frames[a]["clap_cosine"] - frames["dev__cfg7"]["clap_cosine"], frames[a]["mqascore"] - frames["dev__cfg7"]["mqascore"])[0] for a in CORE if a != "dev__cfg7"]
print(f"CLAP vs MQAScore change range over all {len(mqa_values)} pairs {min(mqa_values):+.3f} to {max(mqa_values):+.3f}; cfg3, cfg12, rescale12 vs cfg7 {min(core_mqa):+.3f} to {max(core_mqa):+.3f}")

kl_pairs = [(a, "dev__cfg7") for a in all_generated if a != "dev__cfg7"]
kl_values = []
for a, b in kl_pairs:
    dc = frames[a]["clap_cosine"] - frames[b]["clap_cosine"]
    dk = frames[a]["kl_passt_softmax"] - frames[b]["kl_passt_softmax"]
    r, p = rho(dc, dk)
    kl_values.append(r)
    print(f"{a} - {b}: CLAP vs KL-PaSST {r:+.3f} (p {p:.3f})")
core_kl = [rho(frames[a]["clap_cosine"] - frames["dev__cfg7"]["clap_cosine"], frames[a]["kl_passt_softmax"] - frames["dev__cfg7"]["kl_passt_softmax"])[0] for a in CORE if a != "dev__cfg7"]
print(f"CLAP vs KL change range over all {len(kl_values)} pairs {min(kl_values):+.3f} to {max(kl_values):+.3f}; cfg3, cfg12, rescale12 vs cfg7 {min(core_kl):+.3f} to {max(core_kl):+.3f}")
cross = [(a, b) for a, b in [("dev__rescale12", "dev__cfg7"), ("dev__rescale12_seed1", "dev__cfg7_seed1"), ("dev__rescale12_seed2", "dev__cfg7_seed2"),
                             ("dev__cfg12", "dev__cfg7"), ("dev__cfg3", "dev__cfg7")]]
print("subset with the main comparisons only:")
for a, b in cross:
    dc = frames[a]["clap_cosine"] - frames[b]["clap_cosine"]
    dk = frames[a]["kl_passt_softmax"] - frames[b]["kl_passt_softmax"]
    print(f"  {a} - {b}: CLAP vs KL {rho(dc, dk)[0]:+.3f}")
