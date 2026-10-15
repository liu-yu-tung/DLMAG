"""Bonus ablation: melody only vs. melody one bar ahead vs. melody + chord tokens.

The chord variant also reads the prompt's chord_midi.txt, the file Chord Accuracy is scored against,
so its CA is not comparable as "accompaniment from melody"; report it next to the melody-only runs.

For each variant: train (same schedule), take the kept checkpoint nearest the lowest validation loss,
generate with each config and seed, run eval_bonus.py, and write mean/std over seeds to results/bonus_ablation.csv.

uv run python scripts/bonus_ablation.py
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from hw2.remi import DATA_ROOT, HW2_ROOT

VARIANTS = {"melody": "bonus_ppb16", "lookahead": "bonus_la", "chords": "bonus_chd"}
CONFIGS = {"t1_p0.95_free": [], "t1_p0.95_grammar": ["--grammar"]}


def sh(cmd, cwd=HW2_ROOT, log=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True, stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT)


def best_checkpoint(run):
    log = pd.read_csv(HW2_ROOT / "runs" / run / "log.csv").dropna(subset=["val_loss"])
    best = int(log.loc[log["val_loss"].idxmin(), "step"])
    kept = sorted(int(p.name[5:]) for p in (HW2_ROOT / "checkpoints" / run).glob("step_*"))
    pick = min(kept, key=lambda s: abs(s - best))
    return HW2_ROOT / "checkpoints" / run / f"step_{pick}", best, float(log["val_loss"].min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps", type=int, default=12000)
    p.add_argument("--keep_every", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    args = p.parse_args()
    prompt_dir = DATA_ROOT / "prompts" / "bonus"

    rows = []
    for name in args.variants.split(","):
        tok_dir = DATA_ROOT.parent / "tokens" / VARIANTS[name]
        run = f"abl_{name}"
        if not (HW2_ROOT / "checkpoints" / run / "last").exists():
            (HW2_ROOT / "runs" / run).mkdir(parents=True, exist_ok=True)
            with open(HW2_ROOT / "runs" / run / "stdout.log", "w") as log:
                sh(["python", "scripts/train.py", "--run_name", run, "--tokens_dir", tok_dir, "--max_steps", args.max_steps,
                    "--keep_every", args.keep_every, "--save_at", "0.0"], log=log)
        ckpt, best_step, best_val = best_checkpoint(run)
        for cfg, extra in CONFIGS.items():
            for seed in range(args.seeds):
                out = HW2_ROOT / "outputs" / "bonus_ablation" / name / cfg / f"seed{seed}"
                if not (out / "metrics.csv").exists():
                    sh(["python", "scripts/generate_bonus.py", "--ckpt", ckpt, "--tok2idx", tok_dir / "tok2idx.json",
                        "--output_dir", out, "--seed", seed, "--wav", *extra])
                    sh(["python", "eval_bonus.py", "--prompt_dir", prompt_dir, "--gen_midi_dir", out, "--output_csv", out / "metrics.csv"],
                       cwd=HW2_ROOT / "scripts" / "ta")
                m = pd.read_csv(out / "metrics.csv")
                w = m["n_bars"]
                rows.append({"variant": name, "config": cfg, "seed": seed, "ckpt": ckpt.name, "best_val_step": best_step,
                             "best_val_loss": round(best_val, 4),
                             **{c: (m[c] * w).sum() / w.sum() for c in m.columns[2:]}})

    df = pd.DataFrame(rows)
    out = HW2_ROOT / "results"
    df.round(4).to_csv(out / "bonus_ablation_seeds.csv", index=False)
    metrics = [c for c in df.columns if c.startswith(("CA", "D"))]
    summary = df.groupby(["variant", "config", "ckpt", "best_val_loss"])[metrics].agg(["mean", "std"]).round(4)
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out / "bonus_ablation.csv", index=False)
    print(summary.to_string())
    (out / "bonus_ablation.json").write_text(json.dumps({"note": "chords variant also reads chord_midi.txt (the CA reference)"}))


if __name__ == "__main__":
    main()
