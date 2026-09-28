"""Mix-balance features from Demucs stems: how loud, how dynamic and how bright each stem is within the mix.

Hypothesis (H7): mixing conventions change by decade (vocal level over the band, drum and bass prominence,
compression of individual stems) and possibly by market; these are hard to read off the full mixture.
"""

import librosa
import numpy as np

from hw1.features.separation import SR_IN, STEMS

FRAME, HOP = 2048, 512
ACTIVE_DB = -20.0  # a stem frame is active if within 20 dB of the mixture frame
EPS = 1e-10


def _frame_power(y: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0] ** 2


def _db(p) -> np.ndarray:
    return 10 * np.log10(np.asarray(p) + EPS)


def extract(stems: dict[str, np.ndarray], mix: np.ndarray) -> dict[str, float]:
    p_mix = _frame_power(mix)
    energy = {s: float(np.sum(stems[s] ** 2)) for s in STEMS}
    total, e_mix = sum(energy.values()) + EPS, float(np.sum(mix**2)) + EPS
    f: dict[str, float] = {}
    for s in STEMS:
        y = stems[s]
        p = _frame_power(y)
        active = _db(p) > _db(p_mix) + ACTIVE_DB
        f[f"{s}_share"] = energy[s] / total
        f[f"{s}_rel_db"] = float(_db(energy[s] / e_mix))
        f[f"{s}_active"] = float(active.mean())
        f[f"{s}_frame_db_std"] = float(_db(p[active]).std()) if active.sum() > 1 else 0.0
        f[f"{s}_crest_db"] = float(20 * np.log10((np.abs(y).max() + EPS) / (np.sqrt(np.mean(y**2)) + EPS)))
        if active.any():
            cen = librosa.feature.spectral_centroid(y=y, sr=SR_IN, n_fft=FRAME, hop_length=HOP)[0][: len(active)]
            f[f"{s}_centroid_hz"] = float(cen[active].mean())
        else:
            f[f"{s}_centroid_hz"] = 0.0

    rest = lambda s: sum(energy[o] for o in STEMS if o != s)
    for s in ("vocals", "drums", "bass"):
        f[f"{s}_to_rest_db"] = float(_db(energy[s] / (rest(s) + EPS)))
    p_voc = _frame_power(stems["vocals"])
    p_acc = sum(_frame_power(stems[s]) for s in STEMS if s != "vocals")
    sung = _db(p_voc) > _db(p_mix) + ACTIVE_DB
    f["vocals_to_acc_db_when_sung"] = float(np.median(_db(p_voc[sung]) - _db(p_acc[sung]))) if sung.any() else 0.0
    f["separation_residual_db"] = float(_db(np.sum((mix - sum(stems.values())) ** 2) / e_mix))
    return f
