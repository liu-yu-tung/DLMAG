import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3 / "scripts"))

BUNDLE = "drums, percussion, groovy beat"
NEUTRAL = "The clip is ten seconds long."
WEIGHT = 4.0
# test: (edit axis, sign of the extra term); loud is pushed away from its bundle, sad gets it back
TESTS = {"soft_to_loud": -1, "happy_to_sad": 1}
BUNDLE_GROUPS = ["instrument: drums, percussion", "timbre: groovy, punchy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Can the bundle be cut? (Composable Diffusion negation and conjunction, Liu et al. ECCV 2022, cited in the HW3 "
        "spec.) The edited captions of soft_to_loud get an extra guidance term away from the bundle text; happy_to_sad gets it "
        "toward the bundle. The caption keeps its weight 12 (text_to_music/sampler.py extra_term_cfg); a control arm uses a meaning-free "
        "extra text at the same weight. build: data/edits/bundle_<axis>.csv (edited captions, same ids and seeds) and the queue "
        "commands; score: per arm, the target-minus-source probe and the bundle probes against the plain edited clip, in seed-spread "
        "units. Run with the av-benchmark venv. Writes results/analysis/bundle_guidance.csv.")
    parser.add_argument("stage", choices=("build", "score"))
    return parser.parse_args()


def build():
    out = HW3 / "data/edits"
    for axis in TESTS:
        parts = [pd.read_csv(out / f"{axis}{s}.csv") for s in ("", "_ext") if (out / f"{axis}{s}.csv").exists()]
        pd.concat(parts)[["id", "caption"]].to_csv(out / f"bundle_{axis}.csv", index=False)
        print(axis, sum(len(p) for p in parts))


def score():
    from analysis.attribute_probes import fit_probes
    from analysis.edit_effects import probe_scores
    manifest = json.loads((HW3 / "results/analysis/edits_manifest.json").read_text())
    rows = []
    for kind in ("mert", "clap90"):
        for axis, sign in TESTS.items():
            source, target = manifest[axis]["source"], manifest[axis]["target"]
            groups = sorted({source, target, *BUNDLE_GROUPS})
            probes = fit_probes(kind, groups)
            if not set(groups) <= set(probes):
                continue
            base = probe_scores(probes, "rescale12", kind)
            seeds = [probe_scores(probes, f"rescale12_seed{k}", kind) for k in (1, 2)]
            edited = pd.concat([p for p in (probe_scores(probes, f"edit_{axis}{s}", kind) for s in ("", "_ext")) if p is not None])
            rng = np.random.default_rng(0)
            for arm in ("bundle", "neutral"):
                run = probe_scores(probes, f"bundle_{axis}_{arm}", kind)
                if run is None:
                    continue
                ids = [i for i in run.index if i in edited.index]
                measures = {"target minus source": lambda f: f[target] - f[source], **{g: (lambda f, g=g: f[g]) for g in BUNDLE_GROUPS}}
                for label, measure in measures.items():
                    delta = (measure(run.loc[ids]) - measure(edited.loc[ids])).to_numpy()
                    edit_delta = (measure(edited.loc[ids]) - measure(base.loc[ids])).to_numpy()
                    noise_sd = float(pd.concat([measure(s.loc[ids]) - measure(base.loc[ids]) for s in seeds]).std(ddof=1))
                    means = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
                    rows.append({"kind": kind, "axis": axis, "extra_sign": sign, "arm": arm, "measure": label, "n": len(ids),
                                 "change_vs_edited_sd": float(delta.mean() / noise_sd),
                                 "ci_low_sd": float(np.quantile(means, 0.025) / noise_sd),
                                 "ci_high_sd": float(np.quantile(means, 0.975) / noise_sd),
                                 "edit_effect_sd": float(edit_delta.mean() / noise_sd)})
    table = pd.DataFrame(rows)
    table.to_csv(HW3 / "results/analysis/bundle_guidance.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))


def main():
    args = parse_args()
    build() if args.stage == "build" else score()


if __name__ == "__main__":
    main()
