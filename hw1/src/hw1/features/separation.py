"""Source separation with torchaudio's Hybrid Demucs (HDEMUCS_HIGH_MUSDB_PLUS: vocals, drums, bass, other).

The model expects 44.1 kHz stereo; our clips are 24 kHz mono, so the input is resampled and duplicated to two
channels, and each stem is averaged back to mono and resampled to 24 kHz. Long inputs are processed in
overlapping segments with linear cross-fades (overlap-and-add, as in the torchaudio Hybrid Demucs tutorial).
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

from hw1.data import DATA_ROOT

SR_IN = 24000
STEMS = ["drums", "bass", "other", "vocals"]
STEM_ROOT = DATA_ROOT / "stems"


def stem_dir(key: str, sample_id: str, root: Path = STEM_ROOT) -> Path:
    return root / f"dataset_{key}" / sample_id


def load_separator(device: str = "cuda") -> torch.nn.Module:
    return HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(device)


def _fade(n: int, fade_in: int, fade_out: int, device) -> torch.Tensor:
    w = torch.ones(n, device=device)
    if fade_in:
        w[:fade_in] = torch.linspace(0, 1, fade_in, device=device)
    if fade_out:
        w[-fade_out:] = torch.linspace(1, 0, fade_out, device=device)
    return w


@torch.inference_mode()
def separate(model: torch.nn.Module, y: np.ndarray, segment_s: float = 10.0, overlap_s: float = 0.1) -> dict[str, np.ndarray]:
    """y: (T,) mono float at 24 kHz -> {stem: (T,) mono float32 at 24 kHz}."""
    device = next(model.parameters()).device
    sr = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
    x = torchaudio.functional.resample(torch.as_tensor(y, device=device)[None], SR_IN, sr)
    mix = x.expand(2, -1)[None]  # (1, 2, T)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std().clamp_min(1e-8)
    mix = (mix - mean) / std

    length = mix.shape[-1]
    seg, ov = int(segment_s * sr), int(overlap_s * sr)
    out = torch.zeros(1, len(model.sources), 2, length, device=device)
    start = 0
    while start < length:
        end = min(start + seg, length)
        chunk = model(mix[..., start:end])
        w = _fade(end - start, ov if start > 0 else 0, ov if end < length else 0, device)
        out[..., start:end] += chunk * w
        if end == length:
            break
        start = end - ov
    out = out * std + mean

    stems = torchaudio.functional.resample(out[0].mean(dim=1), sr, SR_IN)[:, : len(y)]
    return {name: stems[i].float().cpu().numpy() for i, name in enumerate(model.sources)}
