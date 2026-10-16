"""Training data: random crops of TA-REMI songs that start at a Bar token, with pitch transposition."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


PITCH_MIN, PITCH_MAX = 21, 108


def clean_tokens(tokens: list[str], tok2idx: dict[str, int]) -> list[str]:
    """Drop events whose Pos_ is outside the vocabulary (2 bars in POP909 run past 8 beats)."""
    out, skip = [], False
    for tok in tokens:
        if tok.startswith("Pos_"):
            skip = tok not in tok2idx
        elif tok == "Bar" or tok.startswith("<"):
            skip = False
        if not skip:
            out.append(tok)
    return out


def split_songs(songs: list[dict], n_val: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    order = sorted(songs, key=lambda s: s["song_name"])
    random.Random(seed).shuffle(order)
    return order[n_val:], order[:n_val]


def transpose(tokens: list[str], shift: int) -> list[str]:
    if shift == 0:
        return tokens
    return [f"Pitch_{int(t[6:]) + shift}" if t.startswith("Pitch_") else t for t in tokens]


class CropDataset(Dataset):
    """Each item is a random window of ctx token ids starting at a Bar (or at <BOS>).

    One epoch = enough crops to cover the token count once. Transposition picks a shift in
    [-max_shift, max_shift] that keeps every pitch of the window inside the 88 piano keys.
    Uses its own RNG, so keep num_workers=0 (items are cheap numpy slices).
    """

    def __init__(self, songs, tok2idx, ctx, max_shift=0, seed=0):
        self.ctx = ctx
        self.max_shift = max_shift
        self.pad = tok2idx["<PAD>"]
        self.pitch_lo = tok2idx[f"Pitch_{PITCH_MIN}"]
        self.pitch_hi = tok2idx[f"Pitch_{PITCH_MAX}"]
        assert self.pitch_hi - self.pitch_lo == PITCH_MAX - PITCH_MIN, "Pitch tokens must be contiguous"
        self.rng = np.random.default_rng(seed)
        self.songs = [np.array([tok2idx[t] for t in clean_tokens(s["tokens"], tok2idx)], dtype=np.int64) for s in songs]
        bar = tok2idx["Bar"]
        self.starts = [np.concatenate([[0], np.flatnonzero(ids == bar)]) for ids in self.songs]
        lengths = np.array([len(ids) for ids in self.songs], dtype=np.float64)
        self.probs = lengths / lengths.sum()
        self.length = int(lengths.sum()) // ctx

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        song = self.rng.choice(len(self.songs), p=self.probs)
        ids = self.songs[song]
        start = self.rng.choice(self.starts[song])
        window = ids[start : start + self.ctx].copy()
        if self.max_shift:
            is_pitch = (window >= self.pitch_lo) & (window <= self.pitch_hi)
            if is_pitch.any():
                lo, hi = window[is_pitch].min() - self.pitch_lo, self.pitch_hi - window[is_pitch].max()
                window[is_pitch] += self.rng.integers(-min(lo, self.max_shift), min(hi, self.max_shift) + 1)
        out = np.full(self.ctx, self.pad, dtype=np.int64)
        out[: len(window)] = window
        return torch.from_numpy(out)


STAT_GROUPS = {"chords": ("Chord_",), "piano_stats": ("PDen_", "PDur_"), "melody_stats": ("MDen_", "MDur_"), "repeat": ("Rep_",)}


class FillDataset(Dataset):
    """Crops of plan-fill songs (hw2.plan): each item starts at <BOS> + header or at a bar's block, then whole bars of
    prefix + Mel/Acc note blocks until ctx tokens.

    Per crop: the Mel/Acc order is random (or fixed by `order`: "melody_first" or "piano_first"), and each group in STAT_GROUPS is removed from every
    prefix with probability drop_p, so the model also works when an attribute is not given.
    Uses its own RNG, so keep num_workers=0.
    """

    def __init__(self, songs, tok2idx, ctx, drop_p=0.15, seed=0, order=None):
        self.ctx, self.drop_p, self.order = ctx, drop_p, order
        self.pad, self.bos, self.eos = tok2idx["<PAD>"], tok2idx["<BOS>"], tok2idx["<EOS>"]
        self.melody_tok, self.piano_tok = tok2idx["Mel"], tok2idx["Acc"]
        self.rng = np.random.default_rng(seed)
        group_of = {i: g for t, i in tok2idx.items() for g, pre in STAT_GROUPS.items() if t.startswith(pre)}
        self.songs = []
        for s in songs:
            blocks = []
            for b in s["blocks"]:
                prefix = np.array([tok2idx[t] for t in b["prefix"]], dtype=np.int64)
                groups = np.array([group_of.get(int(i), "") for i in prefix])
                blocks.append((prefix, groups, np.array([tok2idx[t] for t in b["melody"]], dtype=np.int64),
                               np.array([tok2idx[t] for t in b["piano"]], dtype=np.int64)))
            self.songs.append(([self.bos] + [tok2idx[t] for t in s["header"]], blocks))
        lengths = np.array([sum(len(p) + len(m) + len(a) + 2 for p, _, m, a in bl) for _, bl in self.songs], dtype=np.float64)
        self.probs = lengths / lengths.sum()
        self.length = int(lengths.sum()) // ctx

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        header, blocks = self.songs[self.rng.choice(len(self.songs), p=self.probs)]
        start = int(self.rng.integers(len(blocks)))
        melody_first = self.rng.random() < 0.5 if self.order is None else self.order == "melody_first"
        dropped = [g for g in STAT_GROUPS if self.rng.random() < self.drop_p]
        parts = [np.array(header, dtype=np.int64)] if start == 0 else []
        n = sum(len(x) for x in parts)
        for prefix, groups, melody, piano in blocks[start:]:
            if dropped:
                prefix = prefix[~np.isin(groups, dropped)]
            m = np.concatenate([[self.melody_tok], melody])
            a = np.concatenate([[self.piano_tok], piano])
            parts += [prefix, m, a] if melody_first else [prefix, a, m]
            n += len(prefix) + len(m) + len(a)
            if n >= self.ctx:
                break
        else:
            parts.append(np.array([self.eos], dtype=np.int64))
        window = np.concatenate(parts)[: self.ctx]
        out = np.full(self.ctx, self.pad, dtype=np.int64)
        out[: len(window)] = window
        return torch.from_numpy(out)
