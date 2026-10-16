"""Train a GPT-2 decoder from scratch on TA-REMI POP909 tokens.

Saves a checkpoint each time the smoothed training loss first drops below a level in --save_at
(compared in results/task_a_checkpoints.csv), plus `last` every --save_every steps.

uv run python scripts/train.py --run_name gpt2_ppb16
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

from hw2 import code_version
from hw2.data import CropDataset, FillDataset, split_songs
from hw2.remi import DATA_ROOT, HW2_ROOT, load_sequences, load_vocab


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--pos_per_beat", type=int, default=16)
    p.add_argument("--tokens_dir", type=str, default=None, help="folder with tok2idx.json + sequences.json (default data/tokens/ppb<pos_per_beat>)")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=60000)
    p.add_argument("--max_shift", type=int, default=5, help="transposition range in semitones (0 = off)")
    p.add_argument("--n_val", type=int, default=45, help="songs held out for validation loss")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--keep_every", type=int, default=0, help="also keep a step_<N> copy every N steps (0 = off)")
    p.add_argument("--save_best", action="store_true", help="also keep `best`, the checkpoint with the lowest val loss so far")
    p.add_argument("--save_at", type=str, default="1.0,0.6,0.4,0.3,0.2,0.1")
    p.add_argument("--stop_loss", type=float, default=0.08, help="stop once the smoothed train loss is below this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fill", action="store_true", help="plan-fill songs (tokens_dir/songs.json, hw2.data.FillDataset)")
    p.add_argument("--drop_p", type=float, default=0.15, help="with --fill: chance to drop each attribute group per crop")
    return p.parse_args()


def masked_labels(batch, pad_id):
    """GPT2LMHeadModel shifts labels itself; only padding needs masking."""
    labels = batch.clone()
    labels[labels == pad_id] = -100
    return labels


@torch.no_grad()
def val_loss(model, batches, device, pad_id):
    model.eval()
    losses = []
    for batch in batches:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(input_ids=batch, labels=masked_labels(batch, pad_id)).loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    tok_dir = Path(args.tokens_dir) if args.tokens_dir else DATA_ROOT.parent / "tokens" / f"ppb{args.pos_per_beat}"
    tok2idx, _ = load_vocab(tok_dir / "tok2idx.json")
    train_songs, val_songs = split_songs(load_sequences(tok_dir / ("songs.json" if args.fill else "sequences.json")), args.n_val, args.seed)

    run_dir = HW2_ROOT / "runs" / args.run_name
    ckpt_dir = HW2_ROOT / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args) | {"code_version": code_version()}, indent=2))
    (run_dir / "val_songs.json").write_text(json.dumps(sorted(s["song_name"] for s in val_songs)))

    if args.fill:
        train_ds = FillDataset(train_songs, tok2idx, args.ctx, args.drop_p, args.seed)
        val_ds = FillDataset(val_songs, tok2idx, args.ctx, 0.0, args.seed + 1)
    else:
        train_ds = CropDataset(train_songs, tok2idx, args.ctx, args.max_shift, args.seed)
        val_ds = CropDataset(val_songs, tok2idx, args.ctx, 0, args.seed + 1)
    val_batches = [torch.stack([val_ds[0] for _ in range(args.batch_size)]) for _ in range(8)]
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)  # fixed shape: no recompiles

    config = GPT2Config(
        vocab_size=len(tok2idx),
        n_positions=args.ctx,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=tok2idx["<BOS>"],
        eos_token_id=tok2idx["<EOS>"],
        pad_token_id=tok2idx["<PAD>"],
    )
    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M, train songs {len(train_songs)}, val songs {len(val_songs)}, crops/epoch {len(train_ds)}", flush=True)

    # compiled copy for training steps (2.5x faster on the 4060 Ti); eval and saving use the plain module
    train_model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.max_steps)))),
    )
    save_at = sorted((float(x) for x in args.save_at.split(",")), reverse=True)
    pad_id = tok2idx["<PAD>"]

    log = open(run_dir / "log.csv", "a")
    if log.tell() == 0:
        log.write("step,epoch,train_loss,train_loss_ema,val_loss,lr,elapsed_s\n")
    ema, step, t0, best = None, 0, time.time(), float("inf")
    model.train()
    while step < args.max_steps:
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = train_model(input_ids=batch, labels=masked_labels(batch, pad_id)).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()

            vl = ""
            if step % args.val_every == 0:
                v = val_loss(model, val_batches, device, pad_id)
                vl = f"{v:.4f}"
                if args.save_best and v < best:
                    best = v
                    save(model, ckpt_dir / "best", step, ema)
                    (ckpt_dir / "best" / "train_state.json").write_text(json.dumps({"step": step, "train_loss_ema": ema, "val_loss": v}))
            if step % 50 == 0 or vl:
                epoch = step * args.batch_size / len(train_ds)
                log.write(f"{step},{epoch:.3f},{loss.item():.4f},{ema:.4f},{vl},{sched.get_last_lr()[0]:.2e},{time.time() - t0:.0f}\n")
                log.flush()
                if vl:
                    print(f"step {step} epoch {epoch:.2f} loss {ema:.4f} val {vl} ({time.time() - t0:.0f}s)", flush=True)

            while save_at and step > args.warmup and ema < save_at[0]:
                level = save_at.pop(0)
                save(model, ckpt_dir / f"loss_{level:.2f}", step, ema)
                print(f"saved loss_{level:.2f} at step {step}", flush=True)
            if step % args.save_every == 0:
                save(model, ckpt_dir / "last", step, ema)
            if args.keep_every and step % args.keep_every == 0:
                save(model, ckpt_dir / f"step_{step}", step, ema)
            if step >= args.max_steps or ema < args.stop_loss:
                break
        if ema is not None and ema < args.stop_loss:
            break
    save(model, ckpt_dir / "last", step, ema)
    print(f"done at step {step}, loss {ema:.4f}")


def save(model, path, step, ema):
    model.save_pretrained(path)
    (path / "train_state.json").write_text(json.dumps({"step": step, "train_loss_ema": ema}))


if __name__ == "__main__":
    main()
