import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HW3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HW3))

import numpy as np
import torch

from text_to_music.eval_common import FrechetReference

PRED = HW3 / "outputs" / "pred_cache"
MC = HW3 / "outputs" / "ref" / "musiccaps_cache"
names = [str(i) for i in range(100)]


def groups(cache_dir):
    cache = torch.load(cache_dir / "vggish_features.pth", weights_only=True, map_location="cpu")
    return [cache[n].numpy() for n in names]


mc = torch.load(MC / "vggish_features.pth", weights_only=True, map_location="cpu")
reference = FrechetReference(np.concatenate([mc[k].numpy() for k in mc]))
cache = {}


def g(run):
    if run not in cache:
        cache[run] = groups(PRED / f"dev__{run}")
    return cache[run]


def paired(a, b, seed, resamples=1000):
    idx = np.random.default_rng(seed).integers(0, 100, size=(resamples, 100))
    ga, gb = g(a), g(b)
    point = reference.distance(np.concatenate(ga)) - reference.distance(np.concatenate(gb))
    diffs = np.array([reference.distance(np.concatenate([ga[j] for j in row])) - reference.distance(np.concatenate([gb[j] for j in row])) for row in idx])
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


pairs = [("cfg12", "cfg7"), ("apg12", "cfg7"), ("apg12_r2", "cfg7"), ("interval_early", "cfg7"), ("rescale12", "cfg7"),
         ("rescale12_best3_c8998", "cfg7"), ("tags", "cfg7"),
         ("rescale18", "rescale12"), ("rescale12_phi1", "rescale12"), ("rescale12_best3_c8998", "rescale18"),
         ("rescale12_best3_c8998", "rescale12_phi1"), ("rescale12_best3_c8998", "rescale12")]
for a, b in pairs:
    results = [paired(a, b, s) for s in range(5)]
    excl = sum(1 for r in results if r[1] > 0 or r[2] < 0)
    lows = [r[1] for r in results]
    highs = [r[2] for r in results]
    print(f"{a} - {b}: point {results[0][0]:+.3f}; seed0 [{results[0][1]:+.3f}, {results[0][2]:+.3f}]; "
          f"low range [{min(lows):+.3f}, {max(lows):+.3f}], high range [{min(highs):+.3f}, {max(highs):+.3f}]; excludes 0 in {excl}/5 seeds", flush=True)
