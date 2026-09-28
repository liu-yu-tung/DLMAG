import numpy as np
import torch
from transformers import AutoModel

DEFAULT_MODEL = "m-a-p/MERT-v2-30s"
SAMPLE_RATE = 24000
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_mert(model_id: str = DEFAULT_MODEL, dtype: str = "fp32", device: str = "cuda") -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=DTYPES[dtype], attn_implementation="sdpa"
    )
    return model.eval().to(device)


@torch.inference_mode()
def pooled_layers(
    model: torch.nn.Module, wavs: np.ndarray | torch.Tensor, lengths: list[int] | None = None
) -> np.ndarray:
    """(B, T) float waveforms at 24 kHz -> (B, L, 2*D) float32, per-layer masked mean and std over time."""
    device = next(model.parameters()).device
    x = torch.as_tensor(wavs, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x[None]
    mask = None
    if lengths is not None:
        mask = (torch.arange(x.shape[1], device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]).long()
    out = model(input_values=x, attention_mask=mask, output_hidden_states=True, return_dict=True)
    h = torch.stack(out.hidden_states, dim=1).float()  # (B, L, T, D)
    m = out.feature_attention_mask[:, None, :, None].float()
    n = m.sum(dim=2).clamp_min(1.0)
    mean = (h * m).sum(dim=2) / n
    var = (((h - mean[:, :, None]) ** 2) * m).sum(dim=2) / n
    return torch.cat([mean, var.sqrt()], dim=-1).cpu().numpy()
