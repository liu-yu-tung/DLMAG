"""Production features beyond handcrafted.py: mastering/limiting and the 5-12 kHz band shape. Clip-level, mono 24 kHz.

master_*: short-term level spread (400 ms RMS windows, 100 ms hop; unweighted, not LUFS), crest factor per 50 ms
frame, share of samples near full scale and near the clip peak, flat-topped peak runs (limiter/clipping), clip PAPR.
high_*: long-term average spectrum (n_fft 2048) in dB relative to the 1-4 kHz mean; band levels, fitted slopes,
the frequency where it last sits within 30/40/50 dB of the reference, 5-12 kHz flatness, and how much the 7-12 kHz
share of each frame varies over time (steady hiss vs. musical content).
"""

import numpy as np

SR = 24000
EPS = 1e-10
HIGH_BANDS = [(5000, 7000), (7000, 9000), (9000, 10500), (10500, 11500), (11500, 12000)]


def _frames(y: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def _master(y: np.ndarray) -> dict[str, float]:
    f: dict[str, float] = {}
    st = 10 * np.log10(np.mean(_frames(y, int(0.4 * SR), int(0.1 * SR)) ** 2, 1) + EPS)
    st = st[st > -70]
    p10, p50, p95 = np.percentile(st, [10, 50, 95]) if len(st) else (-70.0, -70.0, -70.0)
    f.update(master_st_p10=p10, master_st_p50=p50, master_st_p95=p95, master_lra=p95 - p10,
             master_st_std=float(np.std(st)) if len(st) else 0.0)
    fr = _frames(y, int(0.05 * SR), int(0.05 * SR))
    rms = np.sqrt(np.mean(fr**2, 1))
    keep = rms > 10 ** (-60 / 20)
    crest = 20 * np.log10(np.abs(fr[keep]).max(1) / (rms[keep] + EPS) + EPS) if keep.any() else np.zeros(1)
    f.update(master_crest_mean=float(crest.mean()), master_crest_std=float(crest.std()),
             master_crest_p10=float(np.percentile(crest, 10)))
    a = np.abs(y)
    peak = a.max() + EPS
    f["master_peak_dbfs"] = 20 * np.log10(peak)
    f["master_papr_db"] = 20 * np.log10(peak / (np.sqrt(np.mean(y**2)) + EPS))
    f["master_fullscale_share"] = float(np.mean(a >= 0.98))
    f["master_nearpeak_share"] = float(np.mean(a >= 0.95 * peak))
    at = a >= 0.99 * peak
    same = np.abs(np.diff(y)) < 1e-4
    run = at[1:] & at[:-1] & same
    starts = np.flatnonzero(np.diff(np.r_[0, run.astype(np.int8)]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[run.astype(np.int8), 0]) == -1)
    f["master_flat_runs_per_s"] = float(np.sum(ends - starts + 1 >= 2) / (len(y) / SR))
    return {k: float(v) for k, v in f.items()}


def _high(y: np.ndarray) -> dict[str, float]:
    n_fft, hop = 2048, 512
    fr = _frames(y, n_fft, hop) * np.hanning(n_fft)[None, :]
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    ltas = 10 * np.log10(P.mean(0) + EPS)
    ref = ltas[(freqs >= 1000) & (freqs < 4000)].mean()
    rel = ltas - ref
    f: dict[str, float] = {}
    for lo, hi in HIGH_BANDS:
        f[f"high_band_{lo // 100}_{hi // 100}"] = float(rel[(freqs >= lo) & (freqs < hi)].mean())
    for lo, hi in ((2000, 5000), (5000, 12000), (8000, 12000)):
        m = (freqs >= lo) & (freqs < hi)
        f[f"high_slope_{lo // 1000}_{hi // 1000}k"] = float(np.polyfit(freqs[m] / 1000, rel[m], 1)[0])
    above = freqs >= 1000
    for thr in (30, 40, 50):
        ok = np.flatnonzero(above & (rel >= -thr))
        f[f"high_edge_{thr}db_khz"] = float(freqs[ok[-1]] / 1000) if len(ok) else 1.0
    m = (freqs >= 5000) & (freqs < 12000)
    pm = P.mean(0)[m] + EPS
    f["high_flatness_5_12k"] = float(np.exp(np.mean(np.log(pm))) / np.mean(pm))
    share = P[:, (freqs >= 7000) & (freqs < 12000)].sum(1) / (P.sum(1) + EPS)
    loud = P.sum(1) > np.percentile(P.sum(1), 20)
    s = 10 * np.log10(share[loud] + EPS)
    f["high_share_7_12k_db_std"] = float(s.std())
    f["high_share_7_12k_db_p10"] = float(np.percentile(s, 10))
    return f


def extract(y: np.ndarray) -> dict[str, float]:
    return {**_master(y), **_high(y)}
