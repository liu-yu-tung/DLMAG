"""Hybrid Demucs encoder activations as clip features (week 4 p.70: the U-Net's down-sampled bottleneck "can be
considered as ... features"). Forward hooks on every spectral (freq_encoder, 6 layers, 48-1536 channels) and
waveform (time_encoder, 5 layers, 48-768 channels) encoder layer; each output is pooled to per-channel mean and
std over all non-channel axes. Input handling matches separation.separate(): 24 kHz mono -> 44.1 kHz stereo,
normalized by the clip mean/std, cut into 10 s segments that run as one batch; segment pools are averaged.
"""

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

from hw1.features.separation import SR_IN


def layer_names(model) -> list[str]:
    return [f"fe{i}" for i in range(len(model.freq_encoder))] + [f"te{i}" for i in range(len(model.time_encoder))]


@torch.inference_mode()
def encoder_features(model: torch.nn.Module, y: np.ndarray, segment_s: float = 10.0) -> dict[str, np.ndarray]:
    """y: (T,) mono at 24 kHz -> {layer: (2*C,) float32 mean and std}."""
    device = next(model.parameters()).device
    sr = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
    x = torchaudio.functional.resample(torch.as_tensor(y, device=device)[None], SR_IN, sr)[0]
    x = (x - x.mean()) / x.std().clamp_min(1e-8)
    seg = int(segment_s * sr)
    n = max(1, len(x) // seg)
    segs = torch.stack([x[i * seg : (i + 1) * seg] for i in range(n)]) if len(x) >= seg else x[None]
    batch = segs[:, None].expand(-1, 2, -1).contiguous()

    acts: dict[str, torch.Tensor] = {}
    hooks = []
    mods = list(model.freq_encoder) + list(model.time_encoder)
    for name, mod in zip(layer_names(model), mods):
        hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o)))
    try:
        model(batch)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for name, a in acts.items():
        a = a.float().transpose(0, 1).reshape(a.shape[1], a.shape[0], -1)  # (C, B, rest)
        mean, std = a.mean(-1).mean(-1), a.std(-1).mean(-1)
        out[name] = torch.cat([mean, std]).cpu().numpy()
    return out
