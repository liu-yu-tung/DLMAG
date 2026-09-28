"""Learned softmax-weighted sum of encoder layers + linear head (MERT paper / lecture 2b probe)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LayerMix(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_classes: int = 6, dropout: float = 0.3) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = (x * self.layer_weights()[None, :, None]).sum(dim=1)
        return self.head(self.drop(mixed))


@dataclass
class FittedLayerMix:
    model: LayerMix
    mean: torch.Tensor
    std: torch.Tensor

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = (torch.as_tensor(X, dtype=torch.float32, device=self.mean.device) - self.mean) / self.std
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def layer_weights(self) -> np.ndarray:
        return self.model.layer_weights().detach().cpu().numpy()


def fit_layermix(
    X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-2,
    dropout: float = 0.3, seed: int = 0, device: str | None = None,
) -> FittedLayerMix:
    """X: (N, L, D) train features, y: (N,) labels. Full-batch AdamW, fixed epochs (no validation-based stopping)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    mean, std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True) + 1e-6
    x = (x - mean) / std
    t = torch.as_tensor(y, dtype=torch.long, device=device)
    model = LayerMix(X.shape[1], X.shape[2], int(t.max()) + 1, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), t).backward()
        opt.step()
    return FittedLayerMix(model, mean, std)
