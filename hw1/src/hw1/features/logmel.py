"""Log-mel spectrogram for the Short-Chunk CNN.

Choice: natural log of (1e-6 + mel power), not dB. Power spectrogram (power=2), Slaney mel
filters (librosa default), 128 bins from 0 Hz to Nyquist (12 kHz), centered frames.
The CNN starts with a BatchNorm on the input, so no global scaling is baked into the cache.
"""

import librosa
import numpy as np

SR = 24000
N_FFT = 2048
HOP = 512
N_MELS = 128
EPS = 1e-6


def n_frames(n_samples: int, hop: int = HOP) -> int:
    return 1 + n_samples // hop


def logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(n_samples,) float32 audio -> (128, T) float32 log-mel, T = 1 + n_samples // HOP."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return np.log(EPS + mel).astype(np.float32)
