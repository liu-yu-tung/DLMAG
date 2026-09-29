"""Qwen2-Audio internal features for a probe: one forward pass per clip with the plain question, no answer.

Saved per clip (fp16):
  enc       (33, 1280)  audio-encoder layer outputs (32 layers + the pooled, normed tower output), mean over time
  llm_audio (33, 4096)  LLM hidden states (embeddings + 32 layers), mean over the audio-token positions
  llm_last  (33, 4096)  LLM hidden states at the last prompt position (right before the answer)
Shards of --shard clips go to features/qwenhid_{key}/ (resumable); merged into features/{key}_qwenhid.npz at the end.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from alm_qwen import PROMPTS, SR_QWEN, load_model, prompt_text, to16
from hw1.data import load_audio, load_manifest

HW1 = Path(__file__).resolve().parents[1]


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.inference_mode()
def clip_features(proc, model, wav16: np.ndarray, prefix: str, enc_store: list) -> dict[str, np.ndarray]:
    enc_store.clear()
    inp = proc(text=[prefix], audio=[wav16], sampling_rate=SR_QWEN, return_tensors="pt", padding=True).to(model.device)
    inp["input_features"] = inp["input_features"].to(torch.float16)
    out = model(**inp, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    tok = getattr(model.config, "audio_token_id", None) or model.config.audio_token_index
    audio = inp["input_ids"][0] == tok
    hs = torch.stack(out.hidden_states)[:, 0].float()
    return {"enc": torch.stack([e[0].float().mean(0) for e in enc_store]).half().cpu().numpy(),
            "llm_audio": hs[:, audio].mean(1).half().cpu().numpy(),
            "llm_last": hs[:, -1].half().cpu().numpy()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=100)
    args = ap.parse_args()

    key = args.dataset
    df = load_manifest(key)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    proc, model = load_model()
    enc_store: list = []
    tower = model.model.audio_tower if hasattr(model, "model") else model.audio_tower
    for layer in tower.layers:
        layer.register_forward_hook(lambda m, i, o: enc_store.append(_out(o)))
    tower.register_forward_hook(lambda m, i, o: enc_store.append(_out(o.last_hidden_state if hasattr(o, "last_hidden_state") else o)))
    prefix = prompt_text(proc, PROMPTS[key]["plain"])

    sdir = HW1 / "features" / (f"qwenhid_{key}" + (f"_limit{args.limit}" if args.limit else ""))
    sdir.mkdir(parents=True, exist_ok=True)
    t0, n_new = time.time(), 0
    for s in range(0, len(df), args.shard):
        path = sdir / f"part_{s // args.shard:03d}.npz"
        if path.exists():
            continue
        rows = df.iloc[s : s + args.shard]
        feats = [clip_features(proc, model, to16(load_audio(p)), prefix, enc_store) for p in rows["path"]]
        np.savez(path, **{k: np.stack([f[k] for f in feats]) for k in feats[0]},
                 sample_id=rows["sample_id"].to_numpy(str), split=rows["split"].to_numpy(str),
                 label=rows["label"].to_numpy(str))
        n_new += len(rows)
        print(f"{key} {s + len(rows)}/{len(df)} {(time.time() - t0) / n_new:.2f}s/clip", flush=True)

    parts = [np.load(p) for p in sorted(sdir.glob("part_*.npz"))]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    out = HW1 / "features" / (f"{key}_qwenhid" + (f"_limit{args.limit}" if args.limit else "") + ".npz")
    np.savez(out, **merged)
    print("saved", out, {k: v.shape for k, v in merged.items()})


if __name__ == "__main__":
    main()
