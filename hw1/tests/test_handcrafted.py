import numpy as np
import pytest

from hw1.features.handcrafted import SR, extract, feature_group

DUR = 10


def _sine(freq: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seed: int = 0) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(SR * DUR)).astype(np.float32)


def _clicks(bpm: float) -> np.ndarray:
    y = np.zeros(SR * DUR, dtype=np.float32)
    step = int(SR * 60 / bpm)
    for i in range(0, len(y) - 200, step):
        y[i : i + 200] = np.hanning(200) * 0.8
    return y


@pytest.fixture(scope="module")
def sine_feats() -> dict[str, float]:
    return extract(_sine(440.0))


def test_names_stable_and_finite(sine_feats):
    other = extract(_noise())
    assert list(sine_feats) == list(other)
    assert all(np.isfinite(v) for v in sine_feats.values())
    assert all(np.isfinite(v) for v in other.values())


def test_groups(sine_feats):
    assert {feature_group(n) for n in sine_feats} == {"timbre", "spectral", "energy", "rhythm", "harmony"}


def test_sine_centroid_and_band(sine_feats):
    assert abs(sine_feats["spectral_centroid_mean"] - 440) < 50
    bands = {n: v for n, v in sine_feats.items() if n.startswith("energy_band_")}
    assert max(bands, key=bands.get) == "energy_band_400_1000hz_ratio"
    assert abs(sum(bands.values()) - 1) < 1e-3


def test_noise_flatter_than_sine(sine_feats):
    assert extract(_noise())["spectral_flatness_mean"] > 10 * sine_feats["spectral_flatness_mean"]


def test_loudness_scales_by_20db():
    loud = extract(_sine(440.0, amp=0.5))["energy_rms_db_mean"]
    quiet = extract(_sine(440.0, amp=0.05))["energy_rms_db_mean"]
    assert abs((loud - quiet) - 20) < 0.5


def test_click_tempo():
    tempo = extract(_clicks(120))["rhythm_tempo_bpm"]
    assert min(abs(tempo - t) for t in (60, 120, 240)) < 6
