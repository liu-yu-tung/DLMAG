from __future__ import annotations

import numpy as np
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

MODEL_NAME = "laion/larger_clap_music"
SOURCE_SR = 24000
TARGET_SR = 48000
WINDOW_S = 10
N_WINDOWS = 3

ADJECTIVES_B = {
    "US": "American",
    "UK": "British",
    "Brazil": "Brazilian",
    "Spain": "Spanish",
    "Germany": "German",
    "Italy": "Italian",
}
COUNTRIES_B = {
    "US": "the United States",
    "UK": "the United Kingdom",
    "Brazil": "Brazil",
    "Spain": "Spain",
    "Germany": "Germany",
    "Italy": "Italy",
}


def load_model(device: str = "cuda") -> tuple[ClapModel, ClapProcessor]:
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device).eval()
    return model, processor


def resample_to_48k(waveform: np.ndarray, src_sr: int = SOURCE_SR) -> np.ndarray:
    t = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    t = torchaudio.functional.resample(t, src_sr, TARGET_SR)
    return t.numpy()


def split_windows(waveform_48k: np.ndarray, n_windows: int = N_WINDOWS, window_s: int = WINDOW_S) -> list[np.ndarray]:
    win_len = window_s * TARGET_SR
    total = win_len * n_windows
    if waveform_48k.shape[0] < total:
        waveform_48k = np.pad(waveform_48k, (0, total - waveform_48k.shape[0]))
    return [waveform_48k[i * win_len:(i + 1) * win_len] for i in range(n_windows)]


@torch.no_grad()
def embed_waveform_batch(
    model: ClapModel,
    processor: ClapProcessor,
    waveforms: list[np.ndarray],
    device: str,
    sampling_rate: int = TARGET_SR,
) -> np.ndarray:
    """Batch of 1-D float32 waveforms at `sampling_rate` in, raw (N, D) audio embeddings out."""
    inputs = processor(audio=waveforms, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.get_audio_features(**inputs)
    out = getattr(out, "pooler_output", out)  # transformers >= 5 returns an output object
    return out.float().cpu().numpy()


def embed_clips(
    model: ClapModel,
    processor: ClapProcessor,
    waveforms_24k: list[np.ndarray],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """List of 24kHz 30s clips in, (N_clips, D) L2-normalized embeddings out (mean of 3 windows)."""
    windows: list[np.ndarray] = []
    for wav in waveforms_24k:
        w48 = resample_to_48k(wav)
        windows.extend(split_windows(w48))

    embs: list[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        embs.append(embed_waveform_batch(model, processor, batch, device))
    all_embs = np.concatenate(embs, axis=0)  # (N_clips*N_WINDOWS, D)

    d = all_embs.shape[1]
    per_clip = all_embs.reshape(len(waveforms_24k), N_WINDOWS, d).mean(axis=1)
    norms = np.linalg.norm(per_clip, axis=1, keepdims=True)
    return (per_clip / np.clip(norms, 1e-12, None)).astype(np.float32)


@torch.no_grad()
def embed_text(model: ClapModel, processor: ClapProcessor, prompts: list[str], device: str) -> np.ndarray:
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.get_text_features(**inputs)
    out = getattr(out, "pooler_output", out).float().cpu().numpy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.clip(norms, 1e-12, None)).astype(np.float32)


def class_text_embeddings(
    model: ClapModel,
    processor: ClapProcessor,
    prompt_groups: list[list[str]],
    device: str,
) -> np.ndarray:
    """prompt_groups[c] is one or more prompt templates for class c. Embeds each, averages within
    the class, then L2-normalizes the averaged vector. Returns (n_classes, D)."""
    flat = [p for group in prompt_groups for p in group]
    flat_emb = embed_text(model, processor, flat, device)
    out = []
    i = 0
    for group in prompt_groups:
        n = len(group)
        mean = flat_emb[i:i + n].mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        out.append(mean)
        i += n
    return np.stack(out).astype(np.float32)


def build_prompts_A() -> dict[str, list[list[str]]]:
    decades = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
    p1 = [[f"music from the {d}"] for d in decades]
    p2 = [
        [
            f"a {d} song with typical {d} production",
            f"a song produced in the {d} with {d}-style production and mixing",
        ]
        for d in decades
    ]
    return {"p1": p1, "p2": p2}


def build_prompts_B() -> dict[str, list[list[str]]]:
    countries = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
    p1 = [[f"music released in {COUNTRIES_B[c]} in the 1980s"] for c in countries]
    p2 = [
        [
            f"a 1980s {ADJECTIVES_B[c]} pop song",
            f"a {ADJECTIVES_B[c]} pop song from the 1980s",
        ]
        for c in countries
    ]
    return {"p1": p1, "p2": p2}


def build_prompts(dataset_key: str) -> dict[str, list[list[str]]]:
    if dataset_key == "A":
        return build_prompts_A()
    if dataset_key == "B":
        return build_prompts_B()
    raise ValueError(f"unknown dataset key: {dataset_key}")


def cosine_scores(audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """audio_emb (N, D) and text_emb (C, D), both L2-normalized. Returns (N, C) cosine similarities."""
    return audio_emb @ text_emb.T
