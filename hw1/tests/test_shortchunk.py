import numpy as np
import torch

from hw1.features.logmel import HOP, SR, logmel, n_frames
from hw1.models.shortchunk import ShortChunkCNN, crop_frames, eval_starts, random_starts


def test_output_shape_and_finite():
    model = ShortChunkCNN().eval()
    out = model(torch.randn(2, 1, 128, 173))
    assert out.shape == (2, 6) and torch.isfinite(out).all()


def test_crop_frames_matches_3_7_seconds():
    assert crop_frames(3.7, SR, HOP) == round(3.7 * SR / HOP)


def test_eval_starts_cover_clip():
    s = eval_starts(1407, 173, 8)
    assert len(s) == 8 and s[0] == 0 and s[-1] == 1407 - 173 and (np.diff(s) > 0).all()


def test_random_starts_in_range_and_short_clip():
    rng = np.random.default_rng(0)
    s = random_starts(rng, 1407, 173, 200)
    assert s.min() >= 0 and s.max() <= 1407 - 173
    assert (random_starts(rng, 100, 173, 5) == 0).all()


def test_logmel_shape_on_synthetic_signal():
    y = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32)
    m = logmel(y)
    assert m.shape == (128, n_frames(len(y))) and np.isfinite(m).all()
