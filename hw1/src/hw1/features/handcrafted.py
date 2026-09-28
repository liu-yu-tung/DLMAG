"""Hand-crafted clip-level audio features (librosa), grouped for ablation and analysis."""

import librosa
import numpy as np

SR = 24000
N_FFT = 2048
HOP = 512
N_MFCC = 20
N_MEL = 64
BAND_EDGES_HZ = [0, 60, 150, 400, 1000, 2500, 6000, 12000]
EPS = 1e-10


def _stats(prefix: str, x: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": float(np.std(x))}


def _timbre(S_mel_db: np.ndarray, S: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(S=S_mel_db, n_mfcc=N_MFCC)
    for i, row in enumerate(mfcc):
        f.update(_stats(f"timbre_mfcc{i:02d}", row))
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, n_fft=N_FFT, hop_length=HOP)
    for i, row in enumerate(contrast):
        f.update(_stats(f"timbre_contrast{i}", row))
    return f


def _spectral(S: np.ndarray, y: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    f.update(_stats("spectral_centroid", librosa.feature.spectral_centroid(S=S, sr=SR)[0]))
    f.update(_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]))
    f.update(_stats("spectral_rolloff85", librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]))
    f.update(_stats("spectral_flatness", librosa.feature.spectral_flatness(S=S)[0]))
    f.update(_stats("spectral_zcr", librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]))
    return f


def _energy(S: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    power = S**2
    rms_db = 10 * np.log10(power.sum(axis=0) + EPS)
    f.update(_stats("energy_rms_db", rms_db))
    active = rms_db[rms_db > rms_db.max() - 60]
    f["energy_dynamic_range_db"] = float(np.percentile(active, 95) - np.percentile(active, 5))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    total = power.sum() + EPS
    for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        band = power[(freqs >= lo) & (freqs < hi)].sum()
        f[f"energy_band_{lo}_{hi}hz_ratio"] = float(band / total)
    flux = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0))
    f.update(_stats("energy_spectral_flux", flux))
    return f


def _rhythm(y: np.ndarray, S_mel_db: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    onset_env = librosa.onset.onset_strength(S=S_mel_db, sr=SR, hop_length=HOP)
    f.update(_stats("rhythm_onset_strength", onset_env))
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    f["rhythm_tempo_bpm"] = float(np.atleast_1d(tempo)[0])
    ibi = np.diff(librosa.frames_to_time(beats, sr=SR, hop_length=HOP))
    f["rhythm_beat_interval_cv"] = float(ibi.std() / ibi.mean()) if len(ibi) > 1 else 0.0
    ac = librosa.autocorrelate(onset_env - onset_env.mean())
    f["rhythm_pulse_clarity"] = float(ac[1:].max() / (ac[0] + EPS)) if len(ac) > 1 else 0.0
    return f


def _harmony(y: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    shifted = np.roll(chroma.mean(axis=1), -int(np.argmax(chroma.mean(axis=1))))
    for i, v in enumerate(shifted):
        f[f"harmony_chroma_rel{i:02d}"] = float(v)
    p = chroma / (chroma.sum(axis=0, keepdims=True) + EPS)
    f.update(_stats("harmony_chroma_entropy", -(p * np.log(p + EPS)).sum(axis=0)))
    f.update(_stats("harmony_chroma_change", np.abs(np.diff(chroma, axis=1)).sum(axis=0)))
    return f


def extract(y: np.ndarray) -> dict[str, float]:
    """All features for one 24 kHz mono clip, as an ordered name -> value dict."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MEL, fmax=SR // 2), ref=1.0
    )
    f: dict[str, float] = {}
    f.update(_timbre(S_mel_db, S))
    f.update(_spectral(S, y))
    f.update(_energy(S))
    f.update(_rhythm(y, S_mel_db))
    f.update(_harmony(y))
    return f


def feature_group(name: str) -> str:
    return name.split("_", 1)[0]
