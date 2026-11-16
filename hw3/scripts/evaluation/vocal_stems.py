"""Vocal-stem energy ratio and Whisper speech/language scores per clip, split by vocal vs other captions.

Hybrid Demucs (HW1's separator) splits each clip; the vocals stem gives a level ratio against the mix, and Whisper
large-v3-turbo (HW1's language-ID head) gives no_speech probability and top-language probability on that stem.
Caption text is only used in memory to set the vocal / other flag. Run with HW1's venv: python -P scripts/evaluation/vocal_stems.py ...
Writes results/<set>__<run>/vocal_stems.{json,csv} (suffix _limit<N> with --limit).
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

HW3_ROOT = Path(__file__).resolve().parents[2]
WHISPER_ID = "openai/whisper-large-v3-turbo"
SR_DEMUCS = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
SR_WHISPER = 16000
SEGMENT_S, OVERLAP_S = 10.0, 0.1
N_PROMPTS = 100
NO_SPEECH_TOKEN = "<|nospeech|>"
# Same pattern as VOCAL_PATTERN in scripts/evaluation/extra_metrics.py (checked against its source at start)
VOCAL_PATTERN = (
    r"\b(vocals?|vocalists?|sing|sings|singer|singers|singing|voices?|choir|choirs|rap|raps|rapper|rappers|rapping|lyrics?|males?|females?|chant|chants|chanting)\b"
)
METRICS = ["vocal_ratio_db", "vocal_rms_dbfs", "no_speech_prob", "top_lang_prob"]


def audio_dir(set_name: str, run: str) -> Path:
    return HW3_ROOT / "outputs" / "ref" / "dev_real" if (set_name, run) == ("dev", "real") else HW3_ROOT / "outputs" / set_name / run


def vocal_caption_flags(set_name: str, ids: list[int]) -> np.ndarray:
    source = (HW3_ROOT / "scripts" / "extra_metrics.py").read_text()
    assert VOCAL_PATTERN in source, "vocal keyword pattern drifted from scripts/evaluation/extra_metrics.py"
    captions = pd.read_csv(HW3_ROOT / "data" / f"{set_name}.csv")
    assert list(captions["id"]) == list(range(N_PROMPTS)), "caption ids must be 0..99 in order"
    pattern = re.compile(VOCAL_PATTERN, re.IGNORECASE)
    return np.array([bool(pattern.search(captions["caption"][i])) for i in ids])


def fade(length: int, fade_in: int, fade_out: int, device) -> torch.Tensor:
    weight = torch.ones(length, device=device)
    if fade_in:
        weight[:fade_in] = torch.linspace(0, 1, fade_in, device=device)
    if fade_out:
        weight[-fade_out:] = torch.linspace(1, 0, fade_out, device=device)
    return weight


@torch.inference_mode()
def separate_vocals(model: torch.nn.Module, mix: torch.Tensor, source: str = "vocals") -> torch.Tensor:
    """mix: (2, T) at 44.1 kHz on the model device -> vocals (2, T)."""
    device = mix.device
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std().clamp_min(1e-8)
    normalized = ((mix - mean) / std)[None]
    length, segment, overlap = normalized.shape[-1], int(SEGMENT_S * SR_DEMUCS), int(OVERLAP_S * SR_DEMUCS)
    out = torch.zeros(1, len(model.sources), 2, length, device=device)
    start = 0
    while start < length:
        end = min(start + segment, length)
        weight = fade(end - start, overlap if start > 0 else 0, overlap if end < length else 0, device)
        out[..., start:end] += model(normalized[..., start:end]) * weight
        if end == length:
            break
        start = end - overlap
    out = out * std + mean
    return out[0, list(model.sources).index(source)]


def rms_db(signal: torch.Tensor) -> float:
    return float(20 * torch.log10(signal.double().pow(2).mean().sqrt().clamp_min(1e-10)))


class SpeechHead:
    def __init__(self, device: str):
        self.device, self.dtype = device, torch.float16 if device == "cuda" else torch.float32
        self.extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID, dtype=self.dtype).eval().to(device)
        config = self.model.generation_config
        self.language_ids = torch.tensor(list(config.lang_to_id.values()), device=device)
        self.language_codes = [token.strip("<|>") for token in config.lang_to_id]
        self.start = config.decoder_start_token_id
        self.no_speech_id = WhisperTokenizerFast.from_pretrained(WHISPER_ID).convert_tokens_to_ids(NO_SPEECH_TOKEN)
        assert isinstance(self.no_speech_id, int) and self.no_speech_id > 0

    @torch.inference_mode()
    def score(self, wav16: np.ndarray) -> tuple[float, str, float]:
        """One decoder step after <|startoftranscript|>: no_speech from the full-vocabulary softmax (as in openai-whisper),
        language from the softmax restricted to language tokens (as HW1's LangID)."""
        features = self.extractor(wav16, sampling_rate=SR_WHISPER, return_tensors="pt").input_features.to(self.device, self.dtype)
        decoder_ids = torch.full((1, 1), self.start, device=self.device)
        logits = self.model(input_features=features, decoder_input_ids=decoder_ids).logits[0, -1].float()
        no_speech = float(torch.softmax(logits, -1)[self.no_speech_id])
        language = torch.softmax(logits[self.language_ids], -1)
        top = int(language.argmax())
        return no_speech, self.language_codes[top], float(language[top])


def group_means(frame: pd.DataFrame) -> dict:
    out = {"n": int(len(frame))}
    for metric in METRICS:
        out[metric] = float(frame[metric].mean()) if len(frame) else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="set_name", required=True, choices=["dev", "test"])
    parser.add_argument("--runs", required=True, help="comma-separated run names; 'real' with --set dev is outputs/ref/dev_real")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="first N ids only (check runs; files get a _limit<N> suffix)")
    args = parser.parse_args()
    runs = [run for run in args.runs.split(",") if run]
    assert runs
    count = N_PROMPTS if args.limit is None else args.limit
    assert 1 <= count <= N_PROMPTS
    ids = list(range(count))
    flags = vocal_caption_flags(args.set_name, ids)
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    if args.device == "cuda":
        assert torch.cuda.is_available()

    separator = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(args.device)
    assert "vocals" in separator.sources
    speech = SpeechHead(args.device)
    resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

    def resample(signal: torch.Tensor, source: int, target: int) -> torch.Tensor:
        if source == target:
            return signal
        if (source, target) not in resamplers:
            resamplers[(source, target)] = torchaudio.transforms.Resample(source, target).to(signal.device)
        return resamplers[(source, target)](signal)

    for run in runs:
        folder = audio_dir(args.set_name, run)
        rows = []
        started = time.time()
        for clip_id in ids:
            samples, rate = sf.read(folder / f"{clip_id}.wav", dtype="float32", always_2d=True)
            mix = resample(torch.from_numpy(samples.T).to(args.device), rate, SR_DEMUCS)
            mix = mix[:1].expand(2, -1) if mix.shape[0] == 1 else mix[:2]
            vocals = separate_vocals(separator, mix)
            mix_db, vocal_db = rms_db(mix), rms_db(vocals)
            vocals16 = resample(vocals.mean(0, keepdim=True), SR_DEMUCS, SR_WHISPER)[0].cpu().numpy()
            no_speech, language, language_prob = speech.score(vocals16)
            rows.append({"id": clip_id, "vocal_caption": bool(flags[clip_id]), "vocal_ratio_db": vocal_db - mix_db,
                         "vocal_rms_dbfs": vocal_db, "mix_rms_dbfs": mix_db, "no_speech_prob": no_speech,
                         "top_lang": language, "top_lang_prob": language_prob})
        frame = pd.DataFrame(rows)
        summary = {
            "set": args.set_name, "run": run, "n_clips": len(frame), "limit": args.limit,
            "demucs": "torchaudio HDEMUCS_HIGH_MUSDB_PLUS", "whisper": WHISPER_ID, "vocal_pattern": VOCAL_PATTERN,
            "all": group_means(frame), "vocal_captions": group_means(frame[frame["vocal_caption"]]),
            "other_captions": group_means(frame[~frame["vocal_caption"]]),
            "seconds": round(time.time() - started, 1),
        }
        if args.device == "cuda":
            summary["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        out_dir = HW3_ROOT / "results" / f"{args.set_name}__{run}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / f"vocal_stems{suffix}.csv", index=False, float_format="%.6g")
        (out_dir / f"vocal_stems{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: summary[key] for key in ("set", "run", "n_clips", "all", "vocal_captions", "other_captions", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
