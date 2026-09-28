from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

LABELS = {
    "A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}


def dataset_dir(key: str, root: Path | None = None) -> Path:
    root = root if root is not None else DATA_ROOT
    return root / f"dataset_{key}"


def _infer_key(d: Path) -> str:
    name = d.name
    if name.endswith("_A") or name == "A":
        return "A"
    if name.endswith("_B") or name == "B":
        return "B"
    for row in pd.read_csv(d / "manifest.csv", nrows=1).itertuples():
        return str(row.sample_id).split("_")[0]
    raise ValueError(f"cannot infer dataset key from {d}")


def load_manifest(key_or_dir: str | Path) -> pd.DataFrame:
    if isinstance(key_or_dir, str) and key_or_dir in ("A", "B"):
        key = key_or_dir
        d = dataset_dir(key)
    else:
        d = Path(key_or_dir)
        key = _infer_key(d)

    df = pd.read_csv(d / "manifest.csv", dtype={"label": str})
    df["label"] = df["label"].fillna("")
    df["path"] = df["audio_path"].apply(lambda p: str((d / p).resolve()))

    labels = LABELS[key]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    unknown = set(df["label"]) - set(labels) - {""}
    if unknown:
        raise ValueError(f"unknown labels in {d / 'manifest.csv'}: {sorted(unknown)}")
    df["y"] = df["label"].map(label_to_idx).fillna(-1).astype(int)

    return df


def load_split(key_or_dir: str | Path, split: str) -> pd.DataFrame:
    df = load_manifest(key_or_dir)
    return df[df["split"] == split].reset_index(drop=True)


def load_audio(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != 24000:
        raise ValueError(f"expected 24000 Hz, got {sr} in {path}")
    return audio
