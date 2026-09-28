from types import SimpleNamespace

import numpy as np
import torch

from hw1.features.mert import pooled_layers


class FakeMert(torch.nn.Module):
    """Mimics MERT-v2 output: 2 layers, 1 frame per 100 samples, hidden value = frame index * (layer + 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_values, attention_mask=None, output_hidden_states=True, return_dict=True):
        b, t = input_values.shape
        frames = t // 100
        base = torch.arange(frames, dtype=torch.float32)[None, :, None].expand(b, frames, 3)
        mask = torch.ones(b, frames, dtype=torch.long)
        if attention_mask is not None:
            mask = attention_mask[:, ::100][:, :frames]
        return SimpleNamespace(hidden_states=(base * 1.0, base * 2.0), feature_attention_mask=mask)


def test_mean_std_per_layer_without_mask():
    out = pooled_layers(FakeMert(), np.zeros((1, 1000), dtype=np.float32))
    assert out.shape == (1, 2, 6)
    frames = np.arange(10, dtype=np.float32)
    np.testing.assert_allclose(out[0, 0, :3], frames.mean(), rtol=1e-6)
    np.testing.assert_allclose(out[0, 1, :3], 2 * frames.mean(), rtol=1e-6)
    np.testing.assert_allclose(out[0, 0, 3:], frames.std(), rtol=1e-5)


def test_mask_excludes_padding():
    out = pooled_layers(FakeMert(), np.zeros((2, 1000), dtype=np.float32), lengths=[1000, 500])
    np.testing.assert_allclose(out[1, 0, :3], np.arange(5).mean(), rtol=1e-6)
    np.testing.assert_allclose(out[0, 0, :3], np.arange(10).mean(), rtol=1e-6)
