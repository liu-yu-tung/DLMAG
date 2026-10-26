"""
Install the following dependencies before running this script:
- numpy
- pandas
- tqdm
- MusDr (https://github.com/slSeanWU/MusDr)
"""

import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from MusDr.musdr.side_utils import (
    get_bars_crop,
    get_pitch_histogram,
    compute_histogram_entropy,
    get_onset_xor_distance
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok2idx_path', type=str, required=True, help='Path to the token-to-index mapping JSON file.')
    parser.add_argument('--sequences_path', type=str, required=True, help='Path to the tokenized song sequences JSON file.')
    parser.add_argument('--output_csv', type=str, default='unconditional_metrics.csv', help='Output path for the evaluation results.')

    return parser.parse_args()

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id, pitch_evs):
    """
    For calculating H1 and H4
    """
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    n_bars = piece_ev_seq.count(bar_ev_id)
    if window_size > n_bars:
        window_size = n_bars

    pitch_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)
        pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
        if pitch_hist is None:
            continue
        pitch_ents.append(compute_histogram_entropy(pitch_hist))

    return np.mean(pitch_ents) if pitch_ents else np.nan

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id, pos_evs, pitch_evs, max_pairs=1000):
    """
    For calculating GS
    """
    import itertools, random
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    n_bars = piece_ev_seq.count(bar_ev_id)
    bar_seqs = [get_bars_crop(piece_ev_seq, b, b, bar_ev_id) for b in range(n_bars)]

    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    grv_sims = []
    for p in pairs:
        dist = get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
        grv_sims.append(1.0 - dist)

    return np.mean(grv_sims) if grv_sims else np.nan

if __name__ == "__main__":
    args = parse_args()

    # Load the token-to-index mapping
    with open(args.tok2idx_path, 'r') as f:
        tok2idx = json.load(f)

    BAR_EV = tok2idx["Bar"]

    # NOTE: if you have a different tokenization scheme, you may need to adjust the following lines to match your tokenization.
    # =====
    pos_keys = [k for k in tok2idx.keys() if k.startswith("Pos_")]
    pos_keys.sort(key=lambda x: int(x.split("_")[1]))
    POS_EVS = [tok2idx[k] for k in pos_keys]

    pitch_keys = [k for k in tok2idx.keys() if k.startswith("Pitch_")]
    pitch_keys.sort(key=lambda x: int(x.split("_")[1]))
    PITCH_EVS = [tok2idx[k] for k in pitch_keys]
    # =====

    # Load the song sequences
    with open(args.sequences_path, 'r') as f:
        generated_data = json.load(f)

    result_dict = {
        'song_name': [],
        'H1 (Pitch Entropy 1-Bar)': [],
        'H4 (Pitch Entropy 4-Bar)': [],
        'GS (Groove Similarity)': []
    }

    print(f"Evaluating {len(generated_data)} unconditionally generated pieces...")

    for item in tqdm(generated_data):
        song_name = item["song_name"]

        # Convert tokens to indcies
        seq_ids = [tok2idx[tok] for tok in item["tokens"] if tok in tok2idx]

        result_dict['song_name'].append(song_name)

        h1 = compute_piece_pitch_entropy(seq_ids, 1, BAR_EV, PITCH_EVS)
        result_dict['H1 (Pitch Entropy 1-Bar)'].append(h1)

        h4 = compute_piece_pitch_entropy(seq_ids, 4, BAR_EV, PITCH_EVS)
        result_dict['H4 (Pitch Entropy 4-Bar)'].append(h4)

        gs = compute_piece_groove_similarity(seq_ids, BAR_EV, POS_EVS, PITCH_EVS)
        result_dict['GS (Groove Similarity)'].append(gs)

    if len(result_dict['song_name']):
        df = pd.DataFrame.from_dict(result_dict)
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"\nEvaluation complete! Results saved to {args.output_csv}")

        # Output Average Summary
        print(f"Average H1: {df['H1 (Pitch Entropy 1-Bar)'].mean():.4f}")
        print(f"Average H4: {df['H4 (Pitch Entropy 4-Bar)'].mean():.4f}")
        print(f"Average GS: {df['GS (Groove Similarity)'].mean():.4f}")