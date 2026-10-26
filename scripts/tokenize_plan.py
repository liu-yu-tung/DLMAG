"""Tokenize the training set for the plan-then-fill pipeline (see hw2.plan).

Writes data/tokens/plan/{tok2idx,sequences,bins}.json (plan model) and data/tokens/fill/{tok2idx,songs,bins}.json
(fill model; one block per bar). Both share one vocabulary: the TA tokens plus hw2.plan.vocab_tokens().
Bin edges come from all 896 training songs (labels only, no note content).

uv run python scripts/tokenize_plan.py   (songs are split over all CPU cores)
uv run python scripts/tokenize_plan.py --repeat   (adds repeat labels; writes data/tokens/{plan,fill}_rep)
uv run python scripts/tokenize_plan.py --repeat --relative   (fill pitches as Rel_ tokens; writes data/tokens/fill_rel only,
    with the plan_rep vocabulary plus the Rel_ tokens at the end, so plan_rep models keep their token ids)
"""

import argparse

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from hw2.plan import bin_edges, song_raw, song_tokens, vocab_tokens
from hw2.remi import DATA_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", action="store_true", help="add a repeat label per bar (hw2.plan.repeat_label)")
    p.add_argument("--relative", action="store_true", help="fill pitches measured from the chord root (Rel_ tokens)")
    args = p.parse_args()
    assert args.repeat or not args.relative, "--relative is built on the repeat token set"
    suffix = "_rel" if args.relative else "_rep" if args.repeat else ""
    base = json.loads((DATA_ROOT.parent / "tokens" / "ppb16" / "tok2idx.json").read_text())
    tok2idx = dict(base)
    for t in vocab_tokens(args.repeat, args.relative):
        tok2idx.setdefault(t, len(tok2idx))

    songs = sorted(p for p in (DATA_ROOT / "training").iterdir() if p.is_dir())
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        raws = [r for r in ex.map(song_raw, songs, chunksize=8) if r is not None]
    edges = bin_edges(raws)
    plans, fills = [], []
    for r in raws:
        plan, fill = song_tokens(r, edges, args.repeat, args.relative)
        plans.append({"song_name": r["song_name"], "tokens": plan})
        fills.append(fill)

    unknown = {t for p in plans for t in p["tokens"]} | {t for f in fills for b in f["blocks"] for k in ("prefix", "melody", "piano") for t in b[k]}
    unknown -= set(tok2idx)
    assert not unknown, f"tokens missing from the vocabulary: {sorted(unknown)[:20]}"

    outputs = [("plan", "sequences.json", plans), ("fill", "songs.json", fills)]
    for name, payload_name, payload in outputs[1:] if args.relative else outputs:
        out = DATA_ROOT.parent / "tokens" / f"{name}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tok2idx.json").write_text(json.dumps(tok2idx, indent=1))
        (out / "bins.json").write_text(json.dumps(edges, indent=1))
        (out / payload_name).write_text(json.dumps(payload))

    plan_len = np.array([len(p["tokens"]) for p in plans])
    bars = np.array([len(f["blocks"]) for f in fills])
    prefix = np.array([len(b["prefix"]) for f in fills for b in f["blocks"]])
    full = np.array([len(b["prefix"]) + 2 + len(b["melody"]) + len(b["piano"]) for f in fills for b in f["blocks"]])
    shifts = np.array([f["shift"] for f in fills])
    keyshifts = sum(t.startswith("KeyShift") for p in plans for t in p["tokens"])
    print(f"songs {len(plans)}, vocab {len(tok2idx)}, bin edges {json.dumps({k: [round(x, 2) for x in v] for k, v in edges.items()})}")
    print(f"plan tokens per song p50 {np.median(plan_len):.0f} p90 {np.percentile(plan_len, 90):.0f} max {plan_len.max()} (bars max {bars.max()})")
    print(f"plan tokens per bar {plan_len.sum() / bars.sum():.2f}; fill prefix per bar p50 {np.median(prefix):.0f} max {prefix.max()}")
    print(f"fill tokens per bar p50 {np.median(full):.0f} p90 {np.percentile(full, 90):.0f}; bars per 2048 at p50 {2048 / np.median(full):.0f}")
    print(f"transposition shift counts {dict(zip(*np.unique(shifts, return_counts=True)))}; KeyShift tokens {keyshifts}")


if __name__ == "__main__":
    main()
