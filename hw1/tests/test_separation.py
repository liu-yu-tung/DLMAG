import numpy as np
import torch

from hw1.features.separation import separate


class HalfSplit(torch.nn.Module):
    """Two fake sources, each half the input: stems must sum back to the mixture."""

    sources = ["a", "b"]

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, mix):  # (1, 2, T) -> (1, 2 sources, 2 channels, T)
        return torch.stack([0.5 * mix, 0.5 * mix], dim=1)


def test_overlap_add_reconstructs_mixture_across_segments():
    rng = np.random.default_rng(0)
    t = np.arange(24000 * 3) / 24000
    y = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.normal(size=t.size)).astype(np.float32)
    stems = separate(HalfSplit(), y, segment_s=1.0, overlap_s=0.1)
    assert set(stems) == {"a", "b"} and all(s.shape == y.shape for s in stems.values())
    np.testing.assert_allclose(stems["a"], stems["b"], atol=1e-6)
    err = stems["a"] + stems["b"] - y
    snr = 10 * np.log10(np.sum(y**2) / np.sum(err**2))
    assert snr > 30, snr
