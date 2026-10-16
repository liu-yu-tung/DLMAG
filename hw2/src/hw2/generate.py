"""Batched sampling from a trained GPT-2 REMI model, with a sliding window for pieces longer than the context."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel


@dataclass
class SampleConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @property
    def name(self) -> str:
        return f"t{self.temperature:g}_k{self.top_k}_p{self.top_p:g}"


def load_model(ckpt: str | Path, device: str = "cuda") -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(ckpt).to(device).eval()
    return model


def filter_logits(logits: torch.Tensor, cfg: SampleConfig) -> torch.Tensor:
    logits = logits / max(cfg.temperature, 1e-6)
    if cfg.top_k > 0:
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > cfg.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: GPT2LMHeadModel,
    prompts: list[list[int]],
    n_bars: int,
    cfg: SampleConfig,
    tok2idx: dict[str, int],
    seed: int = 0,
    max_tokens: int = 8192,
) -> tuple[list[list[int]], list[float]]:
    """Continue each prompt until it has n_bars new Bar-started bars.

    A sample stops when it emits the (n_bars+1)-th new Bar (not kept) or hits max_tokens.
    <PAD>/<BOS> are never sampled; <EOS> is blocked until n_bars bars exist.
    Returns the token ids (prompt + continuation) and the mean NLL of the sampled tokens under the
    untempered model (the nll column of eval_generated.py).
    """
    device = next(model.parameters()).device
    ctx = model.config.n_positions
    bar, eos = tok2idx["Bar"], tok2idx["<EOS>"]
    banned = [tok2idx["<PAD>"], tok2idx["<BOS>"]]
    gen = torch.Generator(device=device).manual_seed(seed)

    B = len(prompts)
    hist = [list(p) for p in prompts]
    new_bars = [0] * B
    nll = [[] for _ in range(B)]
    done = [False] * B
    past, next_pos = None, None

    def window_start(h):
        """Start of the re-encoded window: the first Bar in the last ctx//2 tokens."""
        lo = max(0, len(h) - ctx // 2)
        return next((i for i in range(lo, len(h)) if h[i] == bar), lo)

    def encode(full: bool):
        nonlocal past, next_pos
        windows = [h[window_start(h) :] if len(h) >= ctx or full else h for h in hist]
        width = max(len(w) for w in windows)
        ids = torch.zeros(B, width, dtype=torch.long, device=device)
        mask = torch.zeros(B, width, dtype=torch.long, device=device)
        for i, w in enumerate(windows):
            ids[i, width - len(w) :] = torch.tensor(w, device=device)
            mask[i, width - len(w) :] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        past, next_pos = out.past_key_values, pos[:, -1] + 1
        return out.logits[:, -1].float(), mask

    logits, mask = encode(full=False)
    eos_ok = torch.tensor([n_bars <= 0] * B, device=device)
    while not all(done):
        raw = torch.log_softmax(logits, dim=-1)
        logits[:, banned] = float("-inf")
        logits[:, eos] = logits[:, eos].masked_fill(~eos_ok, float("-inf"))
        probs = torch.softmax(filter_logits(logits, cfg), dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        step_nll = (-raw.gather(1, nxt[:, None]).squeeze(1)).tolist()
        for i, t in enumerate(nxt.tolist()):
            if done[i]:
                continue
            if t == bar:
                new_bars[i] += 1
            if (t == bar and new_bars[i] > n_bars) or t == eos or len(hist[i]) >= max_tokens:
                done[i] = True
                continue
            hist[i].append(t)
            nll[i].append(step_nll[i])
        if all(done):
            break
        eos_ok = torch.tensor([n >= n_bars for n in new_bars], device=device)
        if int(next_pos.max()) >= ctx:
            logits, mask = encode(full=True)
            continue
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=next_pos[:, None], past_key_values=past, use_cache=True)
        past, next_pos = out.past_key_values, next_pos + 1
        logits = out.logits[:, -1].float()
    return hist, [sum(x) / max(1, len(x)) for x in nll]
