"""Sudden loud notes: share of notes at least 20 above their bar's median velocity (all tracks merged, the
`velocity_spike_share` rule of hw2.grade_features), split by track role, for real POP909 openings and generated sets,
as written and as the listening build plays them (real-velocity map, with and without the min_gap melody lift).

uv run python scripts/velocity_spikes.py 10
"""

import glob, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pretty_midi as pm
from pathlib import Path
from hw2.render import apply_velocity_map, median_velocity, one_keyboard, real_velocity_counts, velocity_maps

REAL = real_velocity_counts(Path("outputs/stats/real_velocity_counts.json"))
MIN_GAP = median_velocity(REAL["MELODY"]) - median_velocity(REAL["PIANO"])
BARS, SPIKE = 32, 20
SETS = {"single_t1_p1": "outputs/cand_a/single_t1_p1", "pf_oldplan_fill040": "outputs/cand_a/pf_oldplan_fill040",
        "pf_fill60k": "outputs/sweep/fill_rep_60k/last"}


def piece_rows(m: pm.PrettyMIDI) -> dict:
    downbeats = m.get_downbeats()
    end = downbeats[BARS] if len(downbeats) > BARS else m.get_end_time()
    notes = [(n.start, n.velocity, i.name.upper() == "MELODY") for i in m.instruments for n in i.notes if n.start < end]
    start, velocity, melody = (np.array(x) for x in zip(*notes))
    bar = np.searchsorted(downbeats, start, side="right") - 1
    median = {b: np.median(velocity[bar == b]) for b in np.unique(bar)}
    spike = velocity >= np.array([median[b] for b in bar]) + SPIKE
    own = np.zeros_like(spike)
    for role in (True, False):
        for b in np.unique(bar[melody == role]):
            k = (bar == b) & (melody == role)
            own[k] = velocity[k] >= np.median(velocity[k]) + SPIKE
    two_roles = melody.any() and (~melody).any()
    return {"notes": len(velocity), "spikes": int(spike.sum()), "melody_notes": int(melody.sum()),
            "melody_spikes": int((spike & melody).sum()), "own_role_spikes": int(own.sum()),
            "gap": float(np.median(velocity[melody]) - np.median(velocity[~melody])) if two_roles else np.nan,
            "melody_share_of_bar_notes": float(melody.mean())}


def real_job(f):
    m = pm.PrettyMIDI(f)
    for i in m.instruments:
        if i.name.upper() == "BRIDGE":
            i.name = "PIANO"
    one_keyboard(m)
    return piece_rows(m)


def gen_job(args):
    f, maps, lift = args
    m = pm.PrettyMIDI(f)
    if maps is not None:
        apply_velocity_map(m, maps, min_gap=MIN_GAP if lift else 0)
    one_keyboard(m)
    return piece_rows(m)


def summary(name: str, rows: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    return {"source": name, "pieces": len(d), "spike_share": d.spikes.sum() / d.notes.sum(),
            "melody_share_of_notes": d.melody_notes.sum() / d.notes.sum(),
            "melody_share_of_spikes": d.melody_spikes.sum() / max(d.spikes.sum(), 1),
            "melody_spike_share": d.melody_spikes.sum() / max(d.melody_notes.sum(), 1),
            "piano_spike_share": (d.spikes - d.melody_spikes).sum() / max((d.notes - d.melody_notes).sum(), 1),
            "own_role_spike_share": d.own_role_spikes.sum() / d.notes.sum(),
            "gap_median": d.gap.median(), "gap_sd": d.gap.std()}


if __name__ == "__main__":
    step = int(sys.argv[1])
    out = []
    with Pool(5) as pool:
        out.append(summary("real_open32", pool.map(real_job, sorted(glob.glob("data/POP909/training/*/*.mid"))[::step])))
        for name, folder in SETS.items():
            files = sorted(glob.glob(f"{folder}/*.mid"))
            maps = velocity_maps([pm.PrettyMIDI(x) for x in files], REAL)
            for kind, m, lift in (("written", None, False), ("map", maps, False), ("map_lift", maps, True)):
                out.append(summary(f"{name}_{kind}", pool.map(gen_job, [(x, m, lift) for x in files])))
    table = pd.DataFrame(out)
    table.to_csv("results/velocity_spikes.csv", index=False)
    print(table.round(3).to_string(index=False))
