"""Melody-minus-piano stem loudness (LUFS) per piece: 299 real POP909 songs (seconds 8-72) vs plan-fill sets, with and without the min_gap velocity lift.

uv run python scripts/stem_gap.py 3
"""

import copy, glob, sys, tempfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi as pm
from hw2.render import apply_velocity_map, loudness_lufs, median_velocity, midi_to_wav, one_keyboard, real_velocity_counts, velocity_maps

REAL = real_velocity_counts(Path("outputs/stats/real_velocity_counts.json"))
MIN_GAP = median_velocity(REAL["MELODY"]) - median_velocity(REAL["PIANO"])


def stems(m, window=None):
    with tempfile.TemporaryDirectory() as t:
        out = {}
        for role in ("MELODY", "PIANO"):
            c = copy.deepcopy(m)
            c.instruments = [i for i in c.instruments if (i.name.upper() == "MELODY") == (role == "MELODY")]
            if window:
                for i in c.instruments:
                    i.notes = [n for n in i.notes if window[0] < n.start < window[1]]
            if not any(i.notes for i in c.instruments):
                return None
            p = Path(t) / f"{role}.mid"
            c.write(str(p))
            midi_to_wav(p, p.with_suffix(".wav"), trim=False)
            out[role] = loudness_lufs(p.with_suffix(".wav"))
        return out["MELODY"] - out["PIANO"]


def real_job(f):
    m = pm.PrettyMIDI(f)
    for i in m.instruments:
        if i.name.upper() == "BRIDGE":
            i.name = "PIANO"
    mel = [i for i in m.instruments if i.name.upper() == "MELODY" and i.notes]
    pia = [i for i in m.instruments if i.name.upper() == "PIANO" and i.notes]
    if not mel or not pia:
        return None
    m.instruments = mel + pia
    one_keyboard(m)
    return stems(m, (8, 72))


def gen_job(args):
    f, maps, lift = args
    m = pm.PrettyMIDI(f)
    apply_velocity_map(m, maps, min_gap=MIN_GAP if lift else 0)
    one_keyboard(m)
    return stems(m)


if __name__ == "__main__":
    step = int(sys.argv[1])
    rows = []
    with Pool(5) as pool:
        files = sorted(glob.glob("data/POP909/training/*/*.mid"))[::step]
        r = [g for g in pool.map(real_job, files) if g is not None and np.isfinite(g)]
        rows += [{"source": "real", "gap_db": g} for g in r]
        print("real", len(r), np.mean(r).round(2), np.median(r).round(2), np.std(r).round(2), flush=True)
        for d in ["pf_oldplan_fill040", "pf_newplan_fill040", "pf_oldplan_fill060"]:
            fs = sorted(glob.glob(f"outputs/cand_a/{d}/*.mid"))
            maps = velocity_maps([pm.PrettyMIDI(x) for x in fs], REAL)
            for lift in (False, True):
                r = [g for g in pool.map(gen_job, [(x, maps, lift) for x in fs]) if g is not None and np.isfinite(g)]
                rows += [{"source": f"{d}{'_lift' if lift else ''}", "gap_db": g} for g in r]
                print(d, "lift" if lift else "map", len(r), np.mean(r).round(2), np.median(r).round(2), np.std(r).round(2), flush=True)
    pd.DataFrame(rows).to_csv("results/stem_gap.csv", index=False)
