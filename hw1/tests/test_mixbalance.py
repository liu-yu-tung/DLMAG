import numpy as np

from hw1.features.mixbalance import extract
from hw1.features.separation import STEMS


def test_loud_vocals_and_exact_sum():
    rng = np.random.default_rng(0)
    n = 24000 * 4
    gains = {"drums": 0.02, "bass": 0.05, "other": 0.05, "vocals": 0.5}
    stems = {s: (g * rng.normal(size=n)).astype(np.float32) for s, g in gains.items()}
    mix = sum(stems.values())
    f = extract(stems, mix)
    assert all(np.isfinite(v) for v in f.values())
    assert f["vocals_share"] > 0.9 and abs(sum(f[f"{s}_share"] for s in STEMS) - 1) < 1e-6
    assert 16 < f["vocals_to_rest_db"] < 18
    assert f["vocals_active"] == 1.0 and f["drums_active"] < 0.1
    assert f["separation_residual_db"] < -60


def test_silent_stem_is_finite():
    rng = np.random.default_rng(1)
    n = 24000 * 2
    stems = {s: (0.1 * rng.normal(size=n)).astype(np.float32) for s in STEMS}
    stems["vocals"][:] = 0
    f = extract(stems, sum(stems.values()))
    assert all(np.isfinite(v) for v in f.values())
    assert f["vocals_active"] == 0.0 and f["vocals_to_acc_db_when_sung"] == 0.0
