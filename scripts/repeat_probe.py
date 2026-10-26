"""Teacher-forced melody loss on held-out songs by true repeat kind, with the repeat labels kept, set to Rep_none,
or removed. Shows whether the fill model uses the labels and whether it copies earlier bars at all."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hw2.data import split_songs
from hw2.generate import load_model
from hw2.remi import HW2_ROOT, load_vocab

CTX, MIN_CONTEXT_BARS = 2048, 16


def bar_tokens(block, cond):
    prefix = list(block["prefix"])
    if cond == "none":
        prefix = [("Rep_none" if t.startswith("Rep_") else t) for t in prefix]
    elif cond == "removed":
        prefix = [t for t in prefix if not t.startswith("Rep_")]
    toks = prefix + ["Mel"] + block["melody"] + ["Acc"] + block["piano"]
    part = ["prefix"] * len(prefix) + ["mark"] + ["melody"] * len(block["melody"]) + ["mark"] + ["piano"] * len(block["piano"])
    return toks, part


@torch.no_grad()
def song_nll(model, tok2idx, song, labels, cond, device):
    bars = [bar_tokens(b, cond) for b in song["blocks"]]
    head = ["<BOS>"] + song["header"]
    rows, s, n = [], 0, len(bars)
    while s < n:
        toks, part, owner = (list(head), ["head"] * len(head), [-1] * len(head)) if s == 0 else ([], [], [])
        last = s - 1
        for b in range(s, n):
            if len(toks) + len(bars[b][0]) > CTX:
                break
            toks += bars[b][0]
            part += bars[b][1]
            owner += [b] * len(bars[b][0])
            last = b
        if last < s:
            break
        ids = torch.tensor([[tok2idx[t] for t in toks]], device=device)
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        nll = -logp.gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
        first_scored = s if s == 0 else s + MIN_CONTEXT_BARS
        for j in range(1, len(toks)):
            b = owner[j]
            if part[j] == "melody" and b >= first_scored:
                rows.append({"song": song["song_name"], "bar": b, "kind": labels[b], "token": toks[j].split("_")[0], "nll": nll[j - 1]})
        if last == n - 1:
            break
        s = max(s + 1, last + 1 - MIN_CONTEXT_BARS)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n_val", type=int, default=45)
    p.add_argument("--out", type=Path, default=HW2_ROOT / "results" / "repeat_probe.csv")
    args = p.parse_args()
    rep_songs = json.loads((HW2_ROOT / "data/tokens/fill_rep/songs.json").read_text())
    _, val = split_songs(rep_songs, args.n_val)
    labels = {s["song_name"]: [next((t.split("_")[1] for t in b["prefix"] if t.startswith("Rep_")), "none") for b in s["blocks"]] for s in val}
    plain = {s["song_name"]: s for s in json.loads((HW2_ROOT / "data/tokens/fill/songs.json").read_text())}
    runs = [("fill_rep", "data/tokens/fill_rep", c) for c in ("true", "none", "removed")] + [("fill_v1", "data/tokens/fill", "removed")]
    rows = []
    for ckpt, tokens_dir, cond in runs:
        tok2idx, _ = load_vocab(str(HW2_ROOT / tokens_dir / "tok2idx.json"))
        model = load_model(HW2_ROOT / "checkpoints" / ckpt / "best", "cuda")
        for s in val:
            song = s if ckpt == "fill_rep" else plain[s["song_name"]]
            assert len(song["blocks"]) == len(labels[s["song_name"]])
            for r in song_nll(model, tok2idx, song, labels[s["song_name"]], cond, "cuda"):
                rows.append({"model": ckpt, "labels": cond, **r})
        del model
        print(ckpt, cond, "done", flush=True)
    df = pd.DataFrame(rows)
    df["kind"] = pd.Categorical(df["kind"], ["exact", "var", "rhythm", "none"])
    out = df.groupby(["model", "labels", "kind"], observed=True)["nll"].mean().unstack("kind")
    pitch = df[df.token == "Pitch"].groupby(["model", "labels", "kind"], observed=True)["nll"].mean().unstack("kind")
    table = pd.concat({"melody tokens": out, "Pitch tokens": pitch}, axis=1).round(3)
    table["n_tokens"] = df.groupby(["model", "labels"]).size()
    table.to_csv(args.out)
    print(table.to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
