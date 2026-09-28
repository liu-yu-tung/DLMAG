"""Short-Chunk CNN (Won et al., SMC 2020) on log-mel input, plus crop helpers."""

import numpy as np
import torch
from torch import nn

from hw1.features.logmel import HOP, SR

CHANNELS = (128, 128, 256, 256, 256, 256, 512)


def crop_frames(crop_s: float = 3.7, sr: int = SR, hop: int = HOP) -> int:
    return int(round(crop_s * sr / hop))


def random_starts(rng: np.random.Generator, n_total: int, crop: int, n: int) -> np.ndarray:
    """n random crop start frames in [0, n_total - crop]."""
    return rng.integers(0, max(n_total - crop, 0) + 1, size=n)


def eval_starts(n_total: int, crop: int, n_crops: int = 8) -> np.ndarray:
    """Evenly spaced crop starts covering the whole clip (first at 0, last ends at n_total)."""
    return np.round(np.linspace(0, max(n_total - crop, 0), n_crops)).astype(int)


class ShortChunkCNN(nn.Module):
    def __init__(self, n_classes: int = 6, n_mels: int = 128, channels: tuple[int, ...] = CHANNELS, dropout: float = 0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(1)
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(c_in, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T) log-mel -> (B, n_classes) logits."""
        x = self.blocks(self.input_bn(x))
        return self.head(x.amax(dim=(2, 3)))
