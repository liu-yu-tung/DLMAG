"""Piece features for the listening-grade predictor (`scripts/grade_model.py`), read from the MIDI with all tracks
merged: single-stage pieces and real songs are one piano track while plan-fill has two, so a per-track feature would
only identify the system."""

from pathlib import Path

import numpy as np
import pretty_midi

BARS = 32
SLOTS = 16
HIGH_PITCH = 84
BASS_PITCH = 60
MAJOR = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)

FEATURES = [
    "notes_per_bar", "density_cv", "empty_bars", "longest_onset_gap_beats", "onset_gaps_1bar",
    "bass_dropout", "high_share", "opening_high_share", "opening_density_ratio", "skyline_leap",
    "half_rhythm_sim", "half_register_shift", "half_density_change", "half_pc_sim",
    "loop_run", "one_note_bars", "key_fit", "key_fit_worst8", "mean_chord", "velocity_spike_share",
    "tempo", "tempo_changes", "duration_s",
]


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else np.nan


def key_fit(pc: np.ndarray) -> float:
    total = pc.sum()
    return float(max(pc[np.roll(MAJOR, k)].sum() for k in range(12)) / total) if total > 0 else np.nan


def runs_of(mask: np.ndarray) -> int:
    best = n = 0
    for v in mask:
        n = n + 1 if v else 0
        best = max(best, n)
    return best


def piece_features(path: str | Path, bars: int = BARS) -> dict:
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted((n for inst in pm.instruments if not inst.is_drum for n in inst.notes), key=lambda n: (n.start, n.pitch))
    beats = pm.get_beats()
    downbeats = pm.get_downbeats()
    end = pm.get_end_time()
    step = beats[-1] - beats[-2] if len(beats) > 1 else 0.5

    def to_beats(t):
        t = np.asarray(t, float)
        return np.where(t <= beats[-1], np.interp(t, beats, np.arange(len(beats))), len(beats) - 1 + (t - beats[-1]) / step)

    bar_beats = to_beats(np.append(downbeats, end))[:bars + 1]
    n_bars = len(bar_beats) - 1
    start = to_beats(np.array([n.start for n in notes]))
    stop = to_beats(np.array([n.end for n in notes]))
    pitch = np.array([n.pitch for n in notes])
    velocity = np.array([n.velocity for n in notes])
    keep = (start >= bar_beats[0]) & (start < bar_beats[-1])
    start, stop, pitch, velocity = start[keep], stop[keep], pitch[keep], velocity[keep]
    bar = np.searchsorted(bar_beats, start, side="right") - 1
    bar_len = np.diff(bar_beats)[bar]
    slot = np.clip(((start - bar_beats[bar]) / bar_len * SLOTS).astype(int), 0, SLOTS - 1)
    weight = np.minimum(stop, bar_beats[bar + 1]) - start

    count = np.bincount(bar, minlength=n_bars).astype(float)
    filled = count > 0
    low = np.array([pitch[bar == b].min() if filled[b] else np.nan for b in range(n_bars)])
    onsets = np.unique(np.round(start * SLOTS / 4) * 4 / SLOTS)
    gaps = np.diff(np.concatenate([[bar_beats[0]], onsets, [bar_beats[-1]]]))
    skyline = np.array([pitch[np.isclose(np.round(start * SLOTS / 4) * 4 / SLOTS, t)].max() for t in onsets])

    def pc_hist(mask):
        return np.bincount(pitch[mask] % 12, weight[mask], 12)

    def rhythm_hist(mask):
        return np.bincount(slot[mask], minlength=SLOTS).astype(float)

    half = n_bars // 2
    first, second = bar < half, bar >= half
    opening = bar < 4
    patterns = [frozenset(zip(slot[bar == b], pitch[bar == b])) for b in range(n_bars)]
    loops = np.array([False] + [bool(patterns[b]) and patterns[b] == patterns[b - 1] for b in range(1, n_bars)])
    one_note = [count[b] >= 8 and np.bincount(pitch[bar == b]).max() / count[b] >= 0.6 for b in range(n_bars)]
    worst = [key_fit(pc_hist((bar >= b) & (bar < b + 8))) for b in range(0, n_bars - 7, 4)]
    chord = np.bincount(np.unique(np.round(start * SLOTS / 4), return_inverse=True)[1])
    bar_median = np.array([np.median(velocity[bar == b]) if filled[b] else np.nan for b in range(n_bars)])
    tempos = pm.get_tempo_changes()[1]
    return {
        "notes_per_bar": count.mean(),
        "density_cv": count.std() / count.mean() if count.mean() > 0 else np.nan,
        "empty_bars": int((~filled).sum()),
        "longest_onset_gap_beats": float(gaps.max()),
        "onset_gaps_1bar": int((gaps >= 4).sum()),
        "bass_dropout": float((low[filled] >= BASS_PITCH).mean()) if filled.any() else np.nan,
        "high_share": float((pitch >= HIGH_PITCH).mean()),
        "opening_high_share": float((pitch[opening] >= HIGH_PITCH).mean()) if opening.any() else 0.0,
        "opening_density_ratio": count[:4].mean() / count[4:].mean() if count[4:].mean() > 0 else np.nan,
        "skyline_leap": float(np.abs(np.diff(skyline)).mean()) if len(skyline) > 1 else np.nan,
        "half_rhythm_sim": cosine(rhythm_hist(first), rhythm_hist(second)),
        "half_register_shift": abs(pitch[second].mean() - pitch[first].mean()) if first.any() and second.any() else np.nan,
        "half_density_change": abs(np.log((count[half:].mean() + 1) / (count[:half].mean() + 1))),
        "half_pc_sim": cosine(pc_hist(first), pc_hist(second)),
        "loop_run": runs_of(loops),
        "one_note_bars": int(sum(one_note)),
        "key_fit": key_fit(pc_hist(np.ones_like(bar, bool))),
        "key_fit_worst8": float(np.nanmin(worst)) if worst else np.nan,
        "mean_chord": float(chord.mean()) if len(chord) else np.nan,
        "velocity_spike_share": float((velocity >= bar_median[bar] + 20).mean()),
        "tempo": float(tempos[0]) if len(tempos) else np.nan,
        "tempo_changes": max(0, len(tempos) - 1),
        "duration_s": float(downbeats[bars]) if len(downbeats) > bars else end,
    }
