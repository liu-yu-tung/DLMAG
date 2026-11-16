import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, boot_ci, fmt, per_prompt, sign_flip_p

print("== apg12 vs cfg12 and vs cfg7, dCLAP (bootstrap 1000, seed 0)")
apg, cfg12, cfg7 = per_prompt("dev__apg12"), per_prompt("dev__cfg12"), per_prompt("dev__cfg7")
for label, a, b in (("apg12 - cfg12", apg, cfg12), ("apg12 - cfg7", apg, cfg7), ("cfg12 - cfg7", cfg12, cfg7)):
    d = (a["clap_cosine"] - b["clap_cosine"]).to_numpy()
    print(f"{label}: {fmt(boot_ci(d))}, sign-flip p {sign_flip_p(d):.3f}")

print("\n== CLAP-audio cosine to cfg7 (paired_change_mean) and loudness (rms_dbfs_mean), results/analysis/extra_metrics.csv")
extra = pd.read_csv(RESULTS / "analysis" / "extra_metrics.csv").query("set == 'dev'").set_index("run")
for run in ("apg12", "cfg12", "interval_early", "cfg7", "cfg7_seed1", "cfg7_seed2"):
    print(f"{run}: cosine to cfg7 {extra.loc[run, 'paired_change_mean']:.3f}, loudness {extra.loc[run, 'rms_dbfs_mean']:.1f} dBFS")

print("\n== clipping, mean clipped-sample percentage (gen_info.json mean_clip_fraction x 100)")
for run in ("cfg7", "cfg7_seed1", "cfg7_seed2", "cfg12", "apg12", "rescale12", "rescale12_seed1", "rescale12_seed2"):
    info = json.loads((RESULTS / f"dev__{run}" / "gen_info.json").read_text())
    print(f"{run}: {info['mean_clip_fraction'] * 100:.3f}%")
