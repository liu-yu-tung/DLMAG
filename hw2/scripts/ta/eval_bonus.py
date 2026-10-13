"""
Install the following dependencies before running this script:
- numpy
- pandas
- tqdm
- pretty_midi
"""

import os
import argparse
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, namedtuple

# ==========================================
# Constants
# ==========================================
BASS_WEIGHT = 3.0
TIMESTEPS_PER_BAR = 32
N_CLASSES = 32
DEGENERATE_BANDWIDTH = 0.5

NO_CHORD = -1            # ground truth is 'N', or no chord label covers the bar
UNSUPPORTED_CHORD = -2   # ground-truth dim/aug/sus/unparseable: the 24-triad estimator cannot produce it
NO_ESTIMATE = -3         # generated bar has no notes: never equal to any ground-truth label

OA_COLS = {
    "DP": "DP (Pitch-class OA)",
    "DV": "DV (Velocity OA)",
    "DD": "DD (Duration OA)",
    "DIOI": "DIOI (Inter-Onset OA)",
}
METRIC_COLS = ["CA", "CA-ceiling"] + list(OA_COLS.values())

ROOTS = {'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
         'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
         'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

class SkipSong(Exception):
    pass

def make_oa_grids(n_pitch):
    return {
        "DP":   np.arange(n_pitch),           # 12 pitch classes
        "DV":   np.arange(N_CLASSES),         # velocity classes 0..31
        "DD":   np.arange(1, N_CLASSES + 1),  # duration classes 1..32
        "DIOI": np.arange(1, N_CLASSES + 1),  # IOI classes 1..32 (same as duration)
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_dir', type=str, required=True, help='Path to the 16-bar prompts directory.')
    parser.add_argument('--gen_midi_dir', type=str, required=True, help='Path to the generated MIDI files.')
    parser.add_argument('--output_csv', type=str, default='conditional_metrics.csv', help='Output path for the evaluation results.')

    return parser.parse_args()

# === Time / beat utilities ===
def time_to_beats(t, beat_times):
    t = np.asarray(t, dtype=float)
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 2:
        raise SkipSong("fewer than 2 beats")
    b = np.interp(t, beat_times, np.arange(len(beat_times), dtype=float))
    first = beat_times[1] - beat_times[0]
    last = beat_times[-1] - beat_times[-2]
    b = np.where(t > beat_times[-1], (len(beat_times) - 1) + (t - beat_times[-1]) / last, b)
    b = np.where(t < beat_times[0], (t - beat_times[0]) / first, b)
    return b

def load_gt_beats(path):
    times, order = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                times.append(float(p[0]))
                order.append(float(p[2]))

    times, order = np.array(times), np.array(order)
    down = np.flatnonzero(np.isclose(order, 1.0)).tolist()

    if len(down) < 2:
        down = np.arange(0, len(times), 4).tolist()

    if len(down) > 0 and down[0] > 0:
        down.insert(0, 0)

    return times, np.array(down)

class GTBars:
    """
    Bar grid from the ground-truth beat file, optionally restricted to a bar range.
    """
    def __init__(self, beat_times, down_idx, start_bar=0, max_bars=None):
        self.t, self.d = beat_times, down_idx
        total = len(down_idx) - 1
        end = total if max_bars is None else min(total, start_bar + max_bars)
        self.first = start_bar
        self.n_bars = max(0, end - start_bar)

    def bar_start(self, b):
        return self.t[self.d[self.first + b]]

    def bar_end(self, b):
        return self.t[self.d[self.first + b + 1]]

    def to_timesteps(self, t):
        beats = time_to_beats(t, self.t)
        d = np.asarray(self.d, dtype=float)
        bar = np.interp(beats, d, np.arange(len(d), dtype=float))
        bar = np.where(beats < d[0], (beats - d[0]) / (d[1] - d[0]), bar)
        bar = np.where(beats > d[-1], (len(d) - 1) + (beats - d[-1]) / (d[-1] - d[-2]), bar)
        return bar * TIMESTEPS_PER_BAR
# =========================================

def pick_track(midi, name, fallback_idx):
    for inst in midi.instruments:
        if inst.name.strip().upper() == name:
            return inst
    if len(midi.instruments) > fallback_idx:
        return midi.instruments[fallback_idx]
    raise SkipSong(f"no {name} track")

# === Chord parsing and evaluation ===
def parse_pop909_chord(chord_str):
    """
    Returns 0-11 (major triad root), 12-23 (minor triad root + 12),
    NO_CHORD for 'N', UNSUPPORTED_CHORD for qualities the 24-triad estimator cannot represent.
    """
    if chord_str == 'N':
        return NO_CHORD
    parts = chord_str.split(':')
    root = ROOTS.get(parts[0], -1)
    if root == -1:
        return UNSUPPORTED_CHORD
    quality = (parts[1] if len(parts) > 1 else 'maj').split('/')[0].split('(')[0]
    if quality.startswith(('dim', 'hdim', 'aug')) or 'sus' in quality:
        return UNSUPPORTED_CHORD
    return root + 12 if quality.startswith('min') else root

Chords = namedtuple("Chords", "starts ends labels")
NoteArrays = namedtuple("NoteArrays", "starts ends pitches vels")

def load_gt_chords(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), parse_pop909_chord(p[2])))
    if not rows:
        raise SkipSong("empty chord file")
    rows.sort()
    a = np.array(rows)
    return Chords(a[:, 0], a[:, 1], a[:, 2].astype(int))

def dominant_chord(ch, t0, t1):
    i = int(np.searchsorted(ch.ends, t0, side="right"))
    votes = {}
    while i < len(ch.starts) and ch.starts[i] < t1:
        ov = min(ch.ends[i], t1) - max(ch.starts[i], t0)
        if ov > 0:
            votes[int(ch.labels[i])] = votes.get(int(ch.labels[i]), 0.0) + ov
        i += 1
    return max(votes, key=votes.get) if votes else NO_CHORD

def note_arrays(notes):
    return NoteArrays(np.array([n.start for n in notes], dtype=float),
                      np.array([n.end for n in notes], dtype=float),
                      np.array([n.pitch for n in notes], dtype=int),
                      np.array([n.velocity for n in notes], dtype=float))

def _build_templates():
    t = []
    for root in range(12):
        v = np.zeros(12); v[[root, (root + 4) % 12, (root + 7) % 12]] = 1; t.append(v)
    for root in range(12):
        v = np.zeros(12); v[[root, (root + 3) % 12, (root + 7) % 12]] = 1; t.append(v)
    t = np.array(t)
    return t / np.linalg.norm(t, axis=1, keepdims=True)

TEMPLATES = _build_templates()

def estimate_chord(arr, t0, t1):
    """
    Chord of the generated notes in [t0, t1): best-matching of 24 major/minor triads
    (bass-weighted chroma, cosine similarity). A bar without notes returns NO_ESTIMATE,
    which never matches any ground-truth label (including 'N').
    """
    if len(arr.starts) == 0:
        return NO_ESTIMATE
    m = (arr.starts < t1) & (arr.ends > t0)
    if not m.any():
        return NO_ESTIMATE
    p = arr.pitches[m]
    ov = np.minimum(arr.ends[m], t1) - np.maximum(arr.starts[m], t0)
    w = np.maximum(ov, 1e-3) * arr.vels[m]
    w = np.where(p == p.min(), w * BASS_WEIGHT, w)
    chroma = np.bincount(p % 12, weights=w, minlength=12)
    n = np.linalg.norm(chroma)
    if n == 0:
        return NO_ESTIMATE
    return int(np.argmax(TEMPLATES @ chroma / n))

def chord_accuracy(gen_arr, gt_arr, chords, bars):
    """
    CA = (1 / N_bars) * sum_j I{C_j == C^_j}, over ALL bars (none are skipped).
    """
    hits_gen = hits_gt = 0
    for i in range(bars.n_bars):
        b_start, b_end = bars.bar_start(i), bars.bar_end(i)
        ref = dominant_chord(chords, b_start, b_end)
        hits_gen += int(estimate_chord(gen_arr, b_start, b_end) == ref)
        hits_gt += int(estimate_chord(gt_arr, b_start, b_end) == ref)
    return hits_gen / bars.n_bars, hits_gt / bars.n_bars
# ==========================================

# === Overlap Area (OA) of KDE-smoothed histograms ===
def histogram_kde_pdf(samples, grid):
    """
    Histogram of integer class samples over `grid`, smoothed with a Gaussian KDE.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    hist, _ = np.histogram(samples, bins=np.append(grid - 0.5, grid[-1] + 0.5))
    w = hist / hist.sum()

    mean = np.sum(w * grid)
    var = np.sum(w * (grid - mean) ** 2)
    std = np.sqrt(var * n / (n - 1)) if n > 1 else 0.0
    bw = std * n ** (-1.0 / 5.0)
    if bw <= 0:
        bw = DEGENERATE_BANDWIDTH

    # kernel[g, c] = Gaussian bump centred on class c, evaluated at class g
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / bw) ** 2)
    pdf = kernel @ w
    return pdf / pdf.sum()

def overlap_area(gt_samples, gen_samples, grid):
    if len(gt_samples) == 0 and len(gen_samples) == 0:
        return 1.0      # both bars empty for this feature: identical (empty) distributions
    if len(gt_samples) == 0 or len(gen_samples) == 0:
        return 0.0      # one side has no data: nothing overlaps
    pdf_gt = histogram_kde_pdf(gt_samples, grid)
    pdf_gen = histogram_kde_pdf(gen_samples, grid)
    return float(np.sum(np.minimum(pdf_gt, pdf_gen)))

def track_features(notes, bars, n_pitch):
    """
    Per-note features for a whole track, each tagged with the bar it belongs to.
      * onset  = start time quantised to the nearest timestep; the note belongs to the bar of that timestep
      * D      = duration in timesteps, clipped to 1..32 (32 = one full bar)
      * IOI    = timesteps between consecutive distinct onsets, clipped to 1..32 (same classes as D);
                 it can span bar lines and is assigned to the bar of the later onset
      * V      = MIDI velocity // 4  -> 32 classes (0..31)
      * P      = pitch % 12 (12 pitch classes)
    """
    if not notes:
        return None
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    starts = np.array([n.start for n in notes], dtype=float)
    ends = np.array([n.end for n in notes], dtype=float)
    pitches = np.array([n.pitch for n in notes], dtype=int)
    vels = np.array([n.velocity for n in notes], dtype=int)

    s_ts = bars.to_timesteps(starts)
    e_ts = bars.to_timesteps(ends)
    onsets = np.round(s_ts)
    bar = np.floor_divide(onsets, TIMESTEPS_PER_BAR).astype(int) - bars.first

    # distinct onset positions (simultaneous notes share one onset)
    is_new = np.r_[True, np.diff(onsets) != 0]
    u_onsets, u_bar = onsets[is_new], bar[is_new]

    return {
        "P": pitches % 12 if n_pitch == 12 else pitches,
        "V": np.minimum(vels // 4, N_CLASSES - 1),
        "D": np.clip(np.round(e_ts - s_ts), 1, N_CLASSES),
        "bar": bar,
        "IOI": np.clip(np.diff(u_onsets), 1, N_CLASSES),
        "IOI_bar": u_bar[1:],
    }

def _select(feat, key, bar_key, b):
    if feat is None:
        return np.array([])
    return feat[key][feat[bar_key] == b]

def song_oa(gt_notes, gen_notes, bars, n_pitch=12):
    """
    D_A = (1 / N_bars) * sum_j OA(P^A_j, P^^A_j), over ALL bars.
    """
    grids = make_oa_grids(n_pitch)
    f_gt = track_features(gt_notes, bars, n_pitch)
    f_gen = track_features(gen_notes, bars, n_pitch)
    feature_keys = {"DP": ("P", "bar"), "DV": ("V", "bar"), "DD": ("D", "bar"), "DIOI": ("IOI", "IOI_bar")}

    oa_sums = {k: 0.0 for k in grids}
    for b in range(bars.n_bars):
        for k, grid in grids.items():
            key, bar_key = feature_keys[k]
            oa_sums[k] += overlap_area(_select(f_gt, key, bar_key, b),
                                       _select(f_gen, key, bar_key, b), grid)

    return {OA_COLS[k]: oa_sums[k] / bars.n_bars for k in grids}
# =========================================

def evaluate_song(song, args):
    gen_path = os.path.join(args.gen_midi_dir, f"{song}.mid")
    if not os.path.exists(gen_path):
        raise SkipSong("Generated MIDI missing")

    gen_midi = pretty_midi.PrettyMIDI(gen_path)
    if len(gen_midi.instruments) < 2:
        raise SkipSong("Generated MIDI has fewer than 2 tracks")
    gen_piano = pick_track(gen_midi, "PIANO", 1)

    prompt_folder = os.path.join(args.prompt_dir, song)
    beat_times, down = load_gt_beats(os.path.join(prompt_folder, "beat_midi.txt"))
    bars = GTBars(beat_times, down)
    if bars.n_bars <= 0:
        raise SkipSong("no bars in the requested range")
    chords = load_gt_chords(os.path.join(prompt_folder, "chord_midi.txt"))

    gt_piano_path = os.path.join(prompt_folder, "piano.mid")
    if not os.path.exists(gt_piano_path):
        raise SkipSong("GT Piano MIDI missing in prompt folder")
    gt_midi = pretty_midi.PrettyMIDI(gt_piano_path)
    gt_piano = pick_track(gt_midi, "PIANO", 0)

    gen_arr = note_arrays(gen_piano.notes)
    gt_arr = note_arrays(gt_piano.notes)

    res = {"n_bars": bars.n_bars}

    # Chord Accuracy (CA-ceiling = same estimator run on the ground-truth piano)
    ca, ca_ceil = chord_accuracy(gen_arr, gt_arr, chords, bars)
    res["CA"] = ca
    res["CA-ceiling"] = ca_ceil

    # Overlap Area of pitch / velocity / duration / IOI distributions
    res.update(song_oa(gt_piano.notes, gen_piano.notes, bars))

    return res

if __name__ == "__main__":
    args = parse_args()

    gen_files = sorted([f for f in os.listdir(args.gen_midi_dir) if f.endswith(".mid")])
    print(f"Evaluating {len(gen_files)} conditional generation files...")

    rows = []
    skipped = Counter()

    for filename in tqdm(gen_files):
        song_name = filename[:-4]
        try:
            res = evaluate_song(song_name, args)
            rows.append({"song_name": song_name, **res})
        except SkipSong as e:
            skipped[str(e)] += 1
        except Exception as e:
            skipped[f"error: {type(e).__name__}: {e}"] += 1
            tqdm.write(f"Error evaluating {filename}: {e}")

    if sum(skipped.values()) > 0:
        print(f"\nSkipped {sum(skipped.values())} songs:")
        for reason, c in skipped.most_common():
            print(f"  - {c:4d}: {reason}")

    if rows:
        df = pd.DataFrame(rows, columns=["song_name", "n_bars"] + METRIC_COLS)
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"\nEvaluation complete! Results saved to {args.output_csv}")

        # Pooled over all (track, bar) pairs: 1 / (sum of bars) * sum over every bar of every track
        total_bars = df["n_bars"].sum()
        print(f"\n=== Average Summary (pooled over {len(df)} songs, {total_bars} bars) ===")
        for col in METRIC_COLS:
            pooled = (df[col] * df["n_bars"]).sum() / total_bars
            print(f"Average {col}: {pooled:.4f}")
    else:
        print("\nNo valid songs were evaluated. CSV not generated.")