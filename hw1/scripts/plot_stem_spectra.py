"""Class-mean power spectra of the mixture and each Demucs stem (train clips only), one figure per dataset.

Each clip's Welch PSD is converted to dB, then averaged per class, so the curves show the typical spectral
balance of a decade (A) or market (B). Output: results/stem_spectra_{A,B}.png and results/tables/stem_spectra_{key}.csv.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from hw1.data import LABELS, load_audio, load_manifest
from hw1.features.separation import SR_IN, STEMS, stem_dir

HW1 = Path(__file__).resolve().parents[1]
PANELS = ["mixture", "vocals", "drums", "bass", "other"]


def psd_db(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(y, fs=SR_IN, nperseg=2048)
    return f, 10 * np.log10(p + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    (HW1 / "results" / "tables").mkdir(parents=True, exist_ok=True)

    for key in ("A", "B"):
        df = load_manifest(key)
        df = df[df["split"] == "train"]
        rows, curves = [], {}
        for label in LABELS[key]:
            ids = df[df["label"] == label]
            ids = ids.iloc[np.sort(rng.choice(len(ids), min(args.per_class, len(ids)), replace=False))]
            acc = {p: [] for p in PANELS}
            for _, r in ids.iterrows():
                acc["mixture"].append(psd_db(load_audio(r["path"])))
                for s in PANELS[1:]:
                    acc[s].append(psd_db(load_audio(stem_dir(key, r["sample_id"]) / f"{s}.flac")))
            for p in PANELS:
                f = acc[p][0][0]
                curves[(p, label)] = (f, np.mean([a[1] for a in acc[p]], axis=0))
        for (p, label), (f, m) in curves.items():
            rows += [{"dataset": key, "input": p, "class": label, "freq_hz": round(float(a), 1), "mean_db": round(float(b), 3)}
                     for a, b in zip(f[::8], m[::8])]
        pd.DataFrame(rows).to_csv(HW1 / "results" / "tables" / f"stem_spectra_{key}.csv", index=False)

        fig, axes = plt.subplots(1, len(PANELS), figsize=(20, 3.8), sharex=True)
        cmap = plt.get_cmap("viridis" if key == "A" else "tab10")
        for ax, p in zip(axes, PANELS):
            for i, label in enumerate(LABELS[key]):
                f, m = curves[(p, label)]
                ax.plot(f / 1000, m, color=cmap(i / (len(LABELS[key]) - 1) if key == "A" else i), lw=1.2, label=label)
            ax.set_title(p)
            ax.set_xlabel("kHz")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("mean PSD (dB)")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"Dataset {key} train: class-mean spectrum, {args.per_class} clips per class")
        fig.tight_layout()
        fig.savefig(HW1 / "results" / f"stem_spectra_{key}.png", dpi=110)
        print("wrote", f"stem_spectra_{key}.png")


if __name__ == "__main__":
    main()
