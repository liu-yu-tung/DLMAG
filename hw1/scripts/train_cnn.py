"""Train the Short-Chunk CNN from scratch on random log-mel crops (train split only).

Epoch count is fixed in advance (--epochs); validation is logged per epoch for the report curve and never used
to pick an epoch. Clip score = mean of crop log-softmax over evenly spaced crops. Resumable via --resume.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hw1.data import LABELS
from hw1.features.logmel import HOP, SR
from hw1.metrics import evaluate
from hw1.models.shortchunk import ShortChunkCNN, crop_frames, eval_starts, random_starts

HW1 = Path(__file__).resolve().parents[1]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(key: str, suffix: str) -> tuple[np.ndarray, dict]:
    mel = np.load(HW1 / "features" / f"{key}_logmel{suffix}.npy", mmap_mode="r")
    with np.load(HW1 / "features" / f"{key}_logmel{suffix}_meta.npz") as z:
        return mel, {k: z[k] for k in z.files}


def augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x: (B, 1, 128, T) log-mel. Random gain (+-6 dB as a ln-power shift) and one time and one frequency mask."""
    b, _, f, t = x.shape
    x = x + torch.as_tensor(rng.uniform(-1.4, 1.4, size=(b, 1, 1, 1)), dtype=x.dtype, device=x.device)
    for i in range(b):
        fw, tw = int(rng.integers(0, 17)), int(rng.integers(0, 25))
        f0, t0 = int(rng.integers(0, f - fw + 1)), int(rng.integers(0, t - tw + 1))
        fill = x[i].mean()
        x[i, :, f0 : f0 + fw, :] = fill
        x[i, :, :, t0 : t0 + tw] = fill
    return x


@torch.no_grad()
def clip_logprobs(model: torch.nn.Module, mel: np.ndarray, idx: np.ndarray, crop: int, n_crops: int, device: str) -> np.ndarray:
    model.eval()
    out = []
    starts = eval_starts(mel.shape[2], crop, n_crops)
    for i in idx:
        x = torch.as_tensor(np.stack([mel[i, :, s : s + crop] for s in starts]).astype(np.float32), device=device)[:, None]
        out.append(F.log_softmax(model(x), dim=-1).mean(0).cpu().numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--crops-per-clip", type=int, default=4, help="random crops drawn per train clip per epoch")
    ap.add_argument("--crop-s", type=float, default=3.7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval-crops", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train clips (dry run)")
    ap.add_argument("--tag", default="", help="suffix for run/output names, e.g. _dryrun")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    key, labels = args.dataset, LABELS[args.dataset]
    seed_all(args.seed)
    name = f"cnn_{key}_s{args.seed}{args.tag}"
    run_dir = HW1 / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    mel, meta = load_cache(key, "")
    mel = np.asarray(mel)
    split, sid = meta["split"], meta["sample_id"]
    y = np.array([labels.index(l) if l else -1 for l in meta["label"]])
    tr = np.where(split == "train")[0][: args.limit]
    va, te = np.where(split == "validation")[0], np.where(split == "test")[0]
    crop = crop_frames(args.crop_s, SR, HOP)

    model = ShortChunkCNN(len(labels)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = int(np.ceil(len(tr) * args.crops_per_clip / args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    ckpt_path, log_path = run_dir / "last.pt", run_dir / "log.csv"
    start = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["epoch"]
        print(f"resumed at epoch {start}")
    elif log_path.exists():
        log_path.unlink()

    for epoch in range(start, args.epochs):
        t0 = time.time()
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        model.train()
        clip_ids = np.repeat(tr, args.crops_per_clip)
        rng.shuffle(clip_ids)
        losses = []
        for b in range(0, len(clip_ids), args.batch):
            ids = clip_ids[b : b + args.batch]
            if len(ids) < 2:
                continue
            starts = random_starts(rng, mel.shape[2], crop, len(ids))
            x = torch.as_tensor(np.stack([mel[i, :, s : s + crop] for i, s in zip(ids, starts)]).astype(np.float32),
                                device=args.device)[:, None]
            x = augment(x, rng)
            loss = F.cross_entropy(model(x), torch.as_tensor(y[ids], device=args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        m = evaluate(np.exp(clip_logprobs(model, mel, va, crop, args.eval_crops, args.device)), y[va], labels)
        row = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4), "val_top1": round(m["top1"], 4),
               "val_top3": round(m["top3"], 4), "val_S": round(m["S"], 4), "seconds": round(time.time() - t0, 1)}
        new = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(row, flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1}, ckpt_path)

    lp_va = clip_logprobs(model, mel, va, crop, args.eval_crops, args.device)
    lp_te = clip_logprobs(model, mel, te, crop, args.eval_crops, args.device)
    m = evaluate(np.exp(lp_va), y[va], labels)
    (HW1 / "checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), HW1 / "checkpoints" / f"{name}.pt")
    np.savez(HW1 / "features" / f"{key}_cnn_s{args.seed}{args.tag}_logp.npz", val=lp_va, test=lp_te,
             val_id=sid[va], test_id=sid[te])
    res = {"dataset": key, "seed": args.seed, "epochs": args.epochs, "crop_s": args.crop_s,
           **{k: round(m[k], 4) for k in ("top1", "top3", "S")}}
    (HW1 / "results" / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("final validation", res)


if __name__ == "__main__":
    main()
