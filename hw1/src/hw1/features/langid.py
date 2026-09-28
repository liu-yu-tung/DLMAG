"""Spoken/sung language probabilities from Whisper: one decoder step after <|startoftranscript|>, softmax
restricted to the language tokens (the same logits Whisper's own detect_language takes the argmax of)."""

import numpy as np
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

MODEL_ID = "openai/whisper-large-v3-turbo"
SR_WHISPER = 16000


class LangID:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.float16):
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).eval().to(device)
        gc = self.model.generation_config
        self.codes = [t.strip("<|>") for t in gc.lang_to_id]
        self.lang_ids = torch.tensor(list(gc.lang_to_id.values()), device=device)
        self.start = gc.decoder_start_token_id
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def log_probs(self, wavs: list[np.ndarray], sr: int) -> np.ndarray:
        """wavs: mono float arrays (at most 30 s) at sr -> (B, n_languages) log-probabilities."""
        x = [torchaudio.functional.resample(torch.as_tensor(w), sr, SR_WHISPER).numpy() for w in wavs]
        feats = self.fe(x, sampling_rate=SR_WHISPER, return_tensors="pt").input_features.to(self.device, self.dtype)
        dec = torch.full((len(x), 1), self.start, device=self.device)
        logits = self.model(input_features=feats, decoder_input_ids=dec).logits[:, -1]
        return torch.log_softmax(logits[:, self.lang_ids].float(), dim=-1).cpu().numpy()
