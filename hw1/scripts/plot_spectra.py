"""Per-class long-term average spectra and example spectrograms (train split)."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from hw1.data import LABELS, load_audio, load_split

HW1 = Path(__file__).resolve().parents[1]
SR = 24000
N_FFT = 4096
EPS = 1e-12


def ltas_db(path: str) -> np.ndarray:
    S = np.abs(librosa.stft(load_audio(path), n_fft=N_FFT, hop_length=N_FFT // 2)) ** 2
    return 10 * np.log10(S.mean(axis=1) + EPS)


def plot_ltas(key: str, spectra: np.ndarray, y: np.ndarray, path: Path) -> None:
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    keep = freqs >= 30
    shape = spectra - 10 * np.log10(np.power(10, spectra / 10).sum(axis=1, keepdims=True))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    for c, name in enumerate(LABELS[key]):
        a1.semilogx(freqs[keep], spectra[y == c].mean(0)[keep], label=name, lw=1.2)
        rel = shape[y == c].mean(0) - shape.mean(0)
        a2.semilogx(freqs[keep], rel[keep], label=name, lw=1.2)
    a1.set(title="Absolute level (dB, class mean)", xlabel="frequency (Hz)", ylabel="power (dB)")
    a2.set(title="Spectral shape vs all-class mean (loudness removed)", xlabel="frequency (Hz)", ylabel="dB difference")
    a2.axhline(0, color="gray", lw=0.8)
    for a in (a1, a2):
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"Dataset {key}: long-term average spectrum per class (train, n={len(y)}, 24 kHz so max 12 kHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(key: str, df, path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True, layout="constrained")
    for ax, (c, name) in zip(axes.flat, enumerate(LABELS[key])):
        row = df[df["y"] == c].iloc[rng.integers((df["y"] == c).sum())]
        mel = librosa.feature.melspectrogram(y=load_audio(row["path"]), sr=SR, n_fft=2048, hop_length=512, n_mels=128)
        img = librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max), sr=SR, hop_length=512, x_axis="time", y_axis="mel", ax=ax, vmin=-80, vmax=0
        )
        ax.set_title(f"{name}  ({row['sample_id']})", fontsize=9)
        if c < 3:
            ax.set_xlabel("")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6)
    fig.suptitle(f"Dataset {key}: one random train clip per class, log-mel spectrogram")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = HW1 / "results"
    out.mkdir(exist_ok=True)
    for key in ("A", "B"):
        df = load_split(key, "train")
        spectra = np.asarray(Parallel(n_jobs=6)(delayed(ltas_db)(p) for p in df["path"]))
        np.savez(HW1 / "features" / f"{key}_ltas_train.npz", spectra=spectra, y=df["y"].to_numpy(), sample_id=df["sample_id"].to_numpy(str))
        plot_ltas(key, spectra, df["y"].to_numpy(), out / f"spectrum_{key}_ltas.png")
        plot_examples(key, df, out / f"spectrum_{key}_examples.png")
        print(key, spectra.shape)


if __name__ == "__main__":
    main()
