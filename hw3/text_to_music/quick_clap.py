import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

# HF port used in HW1 (hw1/src/hw1/features/clap.py); only a running estimate, the reported CLAP uses the TA's 90.14 .pt.
MODEL_NAME = "laion/larger_clap_music"
CLAP_SAMPLE_RATE = 48000


class QuickClap:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = ClapProcessor.from_pretrained(MODEL_NAME)
        self.model = ClapModel.from_pretrained(MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def score(self, audio: torch.Tensor, sample_rate: int, caption: str) -> float:
        mono = audio.float().mean(0).cpu()
        mono = torchaudio.functional.resample(mono, sample_rate, CLAP_SAMPLE_RATE).numpy()
        audio_inputs = self.processor(audio=[mono], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        text_inputs = self.processor(text=[caption], return_tensors="pt", padding=True)
        audio_features = self.model.get_audio_features(**{k: v.to(self.device) for k, v in audio_inputs.items()})
        text_features = self.model.get_text_features(**{k: v.to(self.device) for k, v in text_inputs.items()})
        audio_features = getattr(audio_features, "pooler_output", audio_features)
        text_features = getattr(text_features, "pooler_output", text_features)
        return float(torch.nn.functional.cosine_similarity(audio_features, text_features).item())
