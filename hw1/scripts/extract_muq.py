"""MuQ-large (OpenMuQ/MuQ-large-msd-iter, week 2b p.23) hidden states for every clip: per-layer mean and std over time.

Output: features/{key}_muq.npz with feats (N, L, 2*D) float16 (same layout as the MERT file), sample_id, split,
label. --check loads the model twice and compares the features of the first clips (remote-model buffer check).
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from muq import MuQ
from tqdm import tqdm

from hw1.data import load_audio, load_manifest

HW1 = Path(__file__).resolve().parents[1]
MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def load(device: str = "cuda"):
    """MuQ passes a plain EasyDict as the transformers Wav2Vec2Conformer config; transformers 5 also reads
    `_attn_implementation` from it, so set eager attention on every such config."""
    model = MuQ.from_pretrained(MODEL_ID)
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
    _restore_hidden_states(model)
    return model.eval().to(device)


def _restore_hidden_states(model) -> None:
    """transformers 5's Wav2Vec2ConformerEncoder no longer returns hidden_states, which MuQ reads. Rebuild the
    transformers 4 tuple with forward hooks: (input, outputs of layers 1..N-1, layer-normed output of layer N)."""
    for enc in model.modules():
        if type(enc).__name__ != "Wav2Vec2ConformerEncoder":
            continue
        orig = enc.forward

        def forward(hidden_states, attention_mask=None, _orig=orig, _enc=enc, **kw):
            kw.pop("output_hidden_states", None)
            outs, x0 = [], hidden_states.clone()
            hooks = [layer.register_forward_hook(lambda m, i, o: outs.append(o)) for layer in _enc.layers]
            try:
                res = _orig(hidden_states, attention_mask=attention_mask, **kw)
            finally:
                for h in hooks:
                    h.remove()
            res["hidden_states"] = (x0, *outs[:-1], res["last_hidden_state"])
            return res

        enc.forward = forward


@torch.inference_mode()
def pooled(model, wavs: list[np.ndarray]) -> np.ndarray:
    n = min(len(w) for w in wavs)
    x = torch.as_tensor(np.stack([w[:n] for w in wavs]), dtype=torch.float32, device=next(model.parameters()).device)
    hs = torch.stack(model(x, output_hidden_states=True).hidden_states, 1).float()  # (B, L, T, D)
    return torch.cat([hs.mean(2), hs.std(2)], -1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["A", "B"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        wavs = [load_audio(p) for p in load_manifest("B")["path"].head(4)]
        a = pooled(load(), wavs)
        torch.cuda.empty_cache()
        b = pooled(load(), wavs)
        print("shape", a.shape, "finite", bool(np.isfinite(a).all()),
              "max rel diff across loads", float(np.abs(a - b).max() / np.abs(a).max()))
        return

    df = load_manifest(args.dataset)
    model = load()
    paths = df["path"].tolist()
    batches = [paths[i : i + args.batch] for i in range(0, len(paths), args.batch)]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(4) as pool:
        for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc=f"MuQ {args.dataset}"):
            out.append(pooled(model, wavs))
    feats = np.concatenate(out)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite MuQ features")
    path = HW1 / "features" / f"{args.dataset}_muq.npz"
    np.savez(path, feats=feats.astype(np.float16), sample_id=df["sample_id"].to_numpy(str),
             split=df["split"].to_numpy(str), label=df["label"].to_numpy(str))
    print(path, feats.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
