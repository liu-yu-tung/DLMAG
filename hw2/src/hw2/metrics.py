"""H1 / H4 / GS through the TA's eval_basic.py functions, with a fixed seed for GS bar-pair sampling."""

import random
import sys

import numpy as np
import pandas as pd

from hw2.remi import HW2_ROOT

sys.path.insert(0, str(HW2_ROOT / "scripts" / "ta"))
from eval_basic import compute_piece_groove_similarity, compute_piece_pitch_entropy  # noqa: E402

COLUMNS = ["H1", "H4", "GS"]


def vocab_ids(tok2idx: dict[str, int]):
    def ids(prefix):
        keys = sorted((k for k in tok2idx if k.startswith(prefix)), key=lambda k: int(k.split("_")[1]))
        return [tok2idx[k] for k in keys]

    return tok2idx["Bar"], ids("Pos_"), ids("Pitch_")


def evaluate(sequences: list[dict], tok2idx: dict[str, int], seed: int = 0) -> pd.DataFrame:
    """Per-piece metrics, same as eval_basic.py; GS samples at most 1000 bar pairs, seeded here."""
    bar_ev, pos_evs, pitch_evs = vocab_ids(tok2idx)
    rng_state = random.getstate()
    random.seed(seed)
    rows = []
    for item in sequences:
        ids = [tok2idx[t] for t in item["tokens"] if t in tok2idx]
        rows.append(
            {
                "song_name": item["song_name"],
                "H1": compute_piece_pitch_entropy(ids, 1, bar_ev, pitch_evs),
                "H4": compute_piece_pitch_entropy(ids, 4, bar_ev, pitch_evs),
                "GS": compute_piece_groove_similarity(ids, bar_ev, pos_evs, pitch_evs),
            }
        )
    random.setstate(rng_state)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    out = {"n": len(df)}
    for col in COLUMNS:
        out[f"{col}_mean"] = float(np.nanmean(df[col]))
        out[f"{col}_std"] = float(np.nanstd(df[col]))
    return out
