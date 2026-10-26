"""Teacher-forced held-out loss per token type and part, for the main, plan and fill checkpoints; the fill models also
with each song's bar labels shuffled between its bars or removed (how much the fill model reads the plan).

uv run python scripts/loss_review.py --main step_6000 step_10000 loss_0.40 --plan plan_rep/best --fill fill_rep/best fill_rep/step_10000
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

from hw2.data import split_songs
from hw2.generate import load_model
from hw2.remi import HW2_ROOT, load_vocab

MIN_CONTEXT_BARS = 16


def kind(token: str) -> str:
    return token.split("_")[0]


def bars_of(tokens: list[str]) -> tuple[list[str], list[list[str]]]:
    first = tokens.index("Bar")
    starts = [i for i, t in enumerate(tokens) if t == "Bar"] + [len(tokens)]
    return tokens[:first], [tokens[a:b] for a, b in zip(starts, starts[1:])]


def plain_bars(tokens: list[str], part: str) -> tuple[list[str], list[list[tuple[str, str]]]]:
    head, bars = bars_of(tokens)
    return head, [[(t, part) for t in bar] for bar in bars]


def fill_bars(song: dict, labels: str, rng: random.Random) -> tuple[list[str], list[list[tuple[str, str]]]]:
    prefixes = [b["prefix"] for b in song["blocks"]]
    if labels == "shuffled":
        prefixes = [prefixes[0][:1] + p[1:] for p in rng.sample(prefixes, len(prefixes))]
    elif labels == "removed":
        prefixes = [p[:1] for p in prefixes]
    bars = [[(t, "label") for t in pre] + [("Mel", "mark")] + [(t, "melody") for t in b["melody"]]
            + [("Acc", "mark")] + [(t, "piano") for t in b["piano"]] for pre, b in zip(prefixes, song["blocks"])]
    return ["<BOS>"] + song["header"], bars


@torch.no_grad()
def score(model, tok2idx, head: list[str], bars: list[list[tuple[str, str]]], song: str, device="cuda") -> list[dict]:
    """Windows of whole bars up to the context; after the first window, only bars with 16 bars of context are scored."""
    ctx = model.config.n_positions
    rows, s, n = [], 0, len(bars)
    while s < n:
        toks = [(t, "head") for t in head] if s == 0 else []
        owner = [-1] * len(toks)
        last = s - 1
        for b in range(s, n):
            if len(toks) + len(bars[b]) > ctx:
                break
            toks += bars[b]
            owner += [b] * len(bars[b])
            last = b
        if last < s:
            break
        ids = torch.tensor([[tok2idx[t] for t, _ in toks]], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids).logits[0, :-1].float()
        nll = -torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
        first_scored = s if s == 0 else s + MIN_CONTEXT_BARS
        rows += [{"song": song, "part": toks[j][1], "kind": kind(toks[j][0]), "nll": float(nll[j - 1])}
                 for j in range(1, len(toks)) if owner[j] >= first_scored]
        if last == n - 1:
            break
        s = max(s + 1, last + 1 - MIN_CONTEXT_BARS)
    return rows


def run(name: str, tokens_dir: str, items, label_sets=("true",)) -> list[dict]:
    tok2idx, _ = load_vocab(str(HW2_ROOT / tokens_dir / "tok2idx.json"))
    model = load_model(HW2_ROOT / "checkpoints" / name, "cuda")
    rows = []
    for labels in label_sets:
        rng = random.Random(0)
        for song_name, make in items:
            head, bars = make(labels, rng)
            rows += [{"ckpt": name, "labels": labels, **r} for r in score(model, tok2idx, head, bars, song_name)]
        print(name, labels, "done", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--main", nargs="*", default=[], help="gpt2_ppb16 checkpoint names")
    p.add_argument("--plan", nargs="*", default=[], help="<run>/<checkpoint> on plan_rep tokens")
    p.add_argument("--fill", nargs="*", default=[], help="<run>/<checkpoint> on fill_rep tokens")
    p.add_argument("--n_val", type=int, default=45)
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "loss_review.csv")
    args = p.parse_args()

    rows = []
    if args.main:
        _, val = split_songs(json.loads((HW2_ROOT / "data/tokens/ppb16/sequences.json").read_text()), args.n_val)
        items = [(s["song_name"], lambda _l, _r, t=s["tokens"]: plain_bars(t, "all")) for s in val]
        for c in args.main:
            rows += run(f"gpt2_ppb16/{c}", "data/tokens/ppb16", items)
    if args.plan:
        _, val = split_songs(json.loads((HW2_ROOT / "data/tokens/plan_rep/sequences.json").read_text()), args.n_val)
        items = [(s["song_name"], lambda _l, _r, t=s["tokens"]: plain_bars(t, "plan")) for s in val]
        for c in args.plan:
            rows += run(c, "data/tokens/plan_rep", items)
    if args.fill:
        _, val = split_songs(json.loads((HW2_ROOT / "data/tokens/fill_rep/songs.json").read_text()), args.n_val)
        items = [(s["song_name"], lambda labels, rng, s=s: fill_bars(s, labels, rng)) for s in val]
        for c in args.fill:
            rows += run(c, "data/tokens/fill_rep", items, ("true", "shuffled", "removed"))

    df = pd.DataFrame(rows)
    table = df.groupby(["ckpt", "labels", "part", "kind"])["nll"].agg(["mean", "size"]).round(4).reset_index()
    if args.out.exists():
        old = pd.read_csv(args.out)
        table = pd.concat([old[~old.ckpt.isin(table.ckpt.unique())], table])
    table.to_csv(args.out, index=False)
    scored = df[~df.part.isin(["head", "mark", "label"])]
    print(scored.groupby(["ckpt", "labels", "part"])["nll"].mean().unstack("part").round(3).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
