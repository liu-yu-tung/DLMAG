import sys
import time

import torch
import transformers

from hw1.data import load_manifest, load_audio
from hw1.features.mert import DEFAULT_MODEL, DTYPES, load_mert, pooled_layers

model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
path = load_manifest("A")["path"].iloc[0]
wav = load_audio(path)
print("transformers", transformers.__version__, "torch", torch.__version__, "clip", path, wav.shape)

ref = None
for name in DTYPES:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_mert(model_id, name)
    x = torch.from_numpy(wav)[None].cuda()
    with torch.inference_mode():
        out = model(input_values=x, output_hidden_states=True, return_dict=True)
    print(name, "hidden_states:", len(out.hidden_states), tuple(out.hidden_states[0].shape), out.hidden_states[0].dtype,
          "mask:", tuple(out.feature_attention_mask.shape), int(out.feature_attention_mask.sum()))
    t = time.time()
    pooled = torch.from_numpy(pooled_layers(model, wav[None]))
    torch.cuda.synchronize()
    print(f"  pooled {tuple(pooled.shape)} in {time.time() - t:.2f}s, finite={bool(pooled.isfinite().all())}, "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    if ref is None:
        ref = pooled
    else:
        cos = torch.nn.functional.cosine_similarity(pooled[0], ref[0], dim=-1)
        rel = (pooled - ref).norm(dim=-1) / ref.norm(dim=-1)
        print(f"  vs fp32: cosine min {cos.min():.5f} mean {cos.mean():.5f}; rel err max {rel.max():.4f}")
    del model, out
