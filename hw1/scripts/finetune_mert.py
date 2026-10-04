"""Fine-tune the top MERT-v2 blocks, trained on cached outputs of the frozen lower blocks.

Step 1 (--cache): one full MERT-v2 pass per clip; the output of block K (after K of the 24 conformer blocks) is saved
as float16 for every clip: features/mert_hidden_{key}_L{K}.npy (N, 750, 1024) in manifest order.
Step 2: blocks K+1..24 (copied from MERT-v2, fp32) + a head (time mean, LayerNorm, dropout 0.2, linear) are trained on
those cached states. Settings are fixed in advance and never picked on validation: 15 epochs, AdamW (blocks 1e-5, head
1e-3, weight decay 0.01), 1-epoch warmup then cosine, batch 8, random 20 s crops (500 frames) in training and the full
30 s at evaluation, bf16 autocast. Task A uses ordinal soft labels (eps 0.1 per neighbouring decade), B plain labels.
The same 5 stratified train folds as every screen give out-of-fold log-probs; a fit on all of train gives validation
and test log-probs. Output: features/ft_{key}_L{K}_s{seed}.npz and checkpoints/ft_{key}_L{K}_s{seed}.pt.
"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn

from hw1.data import LABELS, load_audio, load_manifest
from hw1.features.mert import load_mert

HW1 = Path(__file__).resolve().parents[1]
FEAT = HW1 / "features"
FRAMES, CROP, DIM = 750, 500, 1024


def cache(key: str, layers: list[int], batch: int = 4) -> None:
    df = load_manifest(key)
    model = load_mert(device="cuda")
    outs = {k: np.lib.format.open_memmap(FEAT / f"mert_hidden_{key}_L{k}.npy", mode="w+", dtype=np.float16,
                                         shape=(len(df), FRAMES, DIM)) for k in layers}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(df), batch):
            x = torch.as_tensor(np.stack([load_audio(p) for p in df["path"][i : i + batch]]), device="cuda")
            hs = model(input_values=x, output_hidden_states=True, return_dict=True).hidden_states
            for k in layers:
                outs[k][i : i + len(x)] = hs[k - 1].half().cpu().numpy()
            if i % 200 == 0:
                print(f"cache {key} {i + len(x)}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    for m in outs.values():
        m.flush()
    np.save(FEAT / f"mert_hidden_{key}_ids.npy", df["sample_id"].to_numpy(str))


class Top(nn.Module):
    def __init__(self, mert, k: int, n_cls: int):
        super().__init__()
        self.blocks = copy.deepcopy(mert.layers[k:]).float()
        self.rot = copy.deepcopy(mert.embed_positions).float()
        # drop any cos/sin table from an earlier fp32 MERT pass: training builds it under bf16 autocast
        self.rot._cos = self.rot._sin = None
        self.rot._sequence_length = 0
        self.head = nn.Sequential(nn.LayerNorm(DIM), nn.Dropout(0.2), nn.Linear(DIM, n_cls))

    def forward(self, h):
        pos = self.rot(h)
        for b in self.blocks:
            h = b(h, pos)
        return self.head(h.mean(1))


def soft_targets(y: torch.Tensor, k: int, eps: float) -> torch.Tensor:
    t = F.one_hot(y, k).float() * 1.0
    if eps > 0:
        for d in (-1, 1):
            nb = (y + d).clamp(0, k - 1)
            ok = ((y + d) >= 0) & ((y + d) < k)
            t[ok, nb[ok]] += eps
            t[ok, y[ok]] -= eps
    return t


def predict(model, X, idx, batch=16) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), batch):
            h = torch.as_tensor(X[idx[i : i + batch]], device="cuda").float()
            out.append(torch.log_softmax(model(h).float(), -1).cpu().numpy())
    return np.concatenate(out)


def train(base, X, y, idx, args, k_cls, seed) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Top(base, args.from_layer, k_cls).cuda()
    opt = torch.optim.AdamW([{"params": model.blocks.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(idx) / args.batch)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(idx)
        loss_sum = 0.0
        for i in range(0, len(order), args.batch):
            b = np.sort(order[i : i + args.batch])
            st = rng.integers(0, FRAMES - CROP + 1, len(b))
            h = torch.as_tensor(np.stack([X[j, s : s + CROP] for j, s in zip(b, st)]), device="cuda").float()
            t = soft_targets(torch.as_tensor(y[b], device="cuda"), k_cls, args.eps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(h)
            loss = -(t * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{args.epochs} loss {loss_sum / len(idx):.4f}", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    ap.add_argument("--from-layer", type=int, default=20)
    ap.add_argument("--cache", type=int, nargs="*", help="only cache these block outputs, then exit")
    ap.add_argument("--folds", default="all", help="'all' (5 folds + full fit), 'full', or a fold number")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eps", type=float, default=None, help="ordinal soft-label eps; default 0.1 for A, 0 for B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grouped", action="store_true",
                    help="5 folds from scripts/grouped_cv.py (features/cvg_{key}.npz), no full fit; tag ..._g_s{seed}")
    args = ap.parse_args()
    key = args.dataset
    if args.cache:
        cache(key, args.cache)
        return
    args.eps = (0.1 if key == "A" else 0.0) if args.eps is None else args.eps

    df = load_manifest(key)
    ids = np.load(FEAT / f"mert_hidden_{key}_ids.npy")
    assert (ids == df["sample_id"].to_numpy(str)).all()
    X = np.load(FEAT / f"mert_hidden_{key}_L{args.from_layer}.npy", mmap_mode="r")
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    tr, va, te = (np.where(split == s)[0] for s in ("train", "validation", "test"))
    base = load_mert(device="cpu")
    k_cls = len(LABELS[key])
    tag = f"ft_{key}_L{args.from_layer}{'_g' if args.grouped else ''}_s{args.seed}"
    out_path = FEAT / f"{tag}.npz"
    res = dict(np.load(out_path)) if out_path.exists() else {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]))
    if args.grouped:
        ref = np.load(FEAT / f"{key}_mertv2.npz")
        ref_tr = ref["sample_id"][ref["split"] == "train"]
        g = np.load(FEAT / f"cvg_{key}.npz")
        pos = {s: i for i, s in enumerate(ids[tr])}
        to_df = np.array([pos[s] for s in ref_tr])
        assert (g["y_train"] == y[tr][to_df]).all()
        folds = [(to_df[a], to_df[b]) for a, b in
                 StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ref_tr, g["y_train"], g["grouped_groups"])]
    todo = list(range(5)) + ["full"] if args.folds == "all" else ["full" if args.folds == "full" else int(args.folds)]
    if args.grouped:
        todo = [f for f in todo if f != "full"]
    for f in todo:
        t0 = time.time()
        if f == "full":
            model = train(base, X, y, tr, args, k_cls, args.seed)
            res.update(val=predict(model, X, va), val_id=ids[va], test=predict(model, X, te), test_id=ids[te])
            torch.save({"blocks": model.blocks.state_dict(), "head": model.head.state_dict(), "from_layer": args.from_layer,
                        "labels": LABELS[key], "args": vars(args)}, HW1 / "checkpoints" / f"{tag}.pt")
        else:
            a, b = tr[folds[f][0]], tr[folds[f][1]]
            model = train(base, X, y, a, args, k_cls, args.seed)
            oof = res.get("oof", np.zeros((len(tr), k_cls)))
            oof[folds[f][1]] = predict(model, X, b)
            res.update(oof=oof, oof_id=ids[tr])
            res[f"done_f{f}"] = np.array(1)
        np.savez(out_path, **res)
        print(f"{tag} fold {f} done in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
