import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
R = HW3 / "results"
A = R / "analysis"
OUT = HW3 / "docs/deck_numbers.md"
RESAMPLES = 2000
lines = []
claims = [0]


def sg(x, d=3):
    return f"{x:+.{d}f}"


def ci(m, lo, hi, d=3, signed=True):
    f = sg if signed else (lambda v, k=d: f"{v:.{k}f}")
    return f"{f(m, d)} [{f(lo, d)}, {f(hi, d)}]"


def claim(text, source):
    claims[0] += 1
    lines.append(f"- {text} — `{source}`")


def head(text):
    lines.append("")
    lines.append(text)
    lines.append("")


def note(text):
    if lines and lines[-1].startswith("- "):
        lines.append("")
    lines.append(text)
    lines.append("")


def pv_(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def per_item(run, file, column):
    return pd.read_csv(R / run / file).set_index("id")[column]


def paired(run_a, run_b, file="per_prompt.csv", column="clap_cosine", ids=None):
    a, b = per_item(run_a, file, column), per_item(run_b, file, column)
    d = (a - b).dropna()
    if ids is not None:
        d = d.loc[[i for i in ids if i in d.index]]
    rng = np.random.default_rng(0)
    values = d.to_numpy()
    means = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(values)


def fmt_pair(run_a, run_b, d=3):
    out = []
    for label, file, column in (("CLAP", "per_prompt.csv", "clap_cosine"), ("KL-PaSST", "per_prompt.csv", "kl_passt_softmax"),
                                ("MQAScore", "mqa_per_item.csv", "mqascore")):
        if not (R / run_a / file).exists() or not (R / run_b / file).exists():
            out.append(f"{label} pending (no per-item file)")
            continue
        m, lo, hi, n = paired(run_a, run_b, file, column)
        out.append(f"{label} {ci(m, lo, hi, d if label != 'KL-PaSST' else 3)} (n={n})")
    return "; ".join(out)


knowledge = (A / "knowledge.md").read_text().splitlines()
verdicts = []
for ln in knowledge:
    m = re.match(r"^(met|MISSED)\s+(\S+) vs (\S+), (\S+) in \[([^\]]+)\] \(written ([^)]+)\): got (.*)$", ln)
    if m:
        verdicts.append(dict(zip(("v", "run", "base", "metric", "range", "written", "got"), m.groups())))


def verdict(run, base, metric):
    for v in verdicts:
        if v["run"] == run and v["base"] == base and v["metric"] == metric:
            return f"prediction [{v['range']}] {v['v'].upper()}, knowledge.md got {v['got']}"
    return None


table = pd.read_csv(A / "findings_table.csv").set_index("run")
fd = json.load(open(A / "paired_fd_dev.json"))["runs"]


def metrics_json(run):
    return json.load(open(R / run / "metrics.json"))


head("# HW3 deck numbers (checked sheet)")
note("Written by `scripts/analysis/deck_numbers.py` from files in `results/`. Intervals are 95% bootstrap over the 100 prompts. "
     "Values from `findings_table.csv`, `knowledge.md` and the analysis CSVs use their own bootstrap (1000 resamples, seed 0); "
     "values marked 'computed' use 2000 resamples, seed 0, on the per-item files. Test appears only as aggregate numbers.")

head("## 1. Main metric table")
note("Baseline `cfg7` (CFG 7, seed 0); `rescale12` (CFG 12, rescale 0.7, seed 0); `rescale12 best-of-3 (c8998 ranker)` is the submission "
     "(best of 3 seeds per prompt, ranked by the CLAP 89.98 checkpoint, scored by the 90.14 checkpoint).")
rows = [("dev", "cfg7", "baseline"), ("dev", "rescale12", "rescale12"), ("dev", "rescale12_best3_c8998", "rescale12 best-of-3 c8998 (submission)"),
        ("test", "cfg7", "baseline"), ("test", "rescale12", "rescale12"), ("test", "rescale12_best3_c8998", "rescale12 best-of-3 c8998 (submission)")]
for set_name, run, label in rows:
    full = f"{set_name}__{run}"
    mj = metrics_json(full)
    t = table.loc[full]
    clap, kl = mj["LAION-CLAP-Score"], mj["KL-PASST-softmax"]
    base = f"{set_name}__cfg7"
    if run == "cfg7":
        paired_text = "reference"
    else:
        mq = paired(full, base, "mqa_per_item.csv", "mqascore")
        paired_text = (f"vs cfg7 paired: dCLAP {ci(t.clap_difference, t.clap_difference_low, t.clap_difference_high)}, "
                       f"dKL {ci(t.kl_difference, t.kl_difference_low, t.kl_difference_high)}, "
                       f"dMQAScore (computed, n={mq[3]}) {ci(*mq[:3])}")
        if set_name == "dev":
            f_ = fd[run]["VGG"]
            paired_text += f", dFD-VGG {ci(f_['difference'], *f_['difference_ci95'], d=2)}"
        else:
            paired_text += ", dFD-VGG not computed for test (paired FD file is dev only)"
    mqa_abs = f"{t.mqascore:.3f} [{t.mqascore_low:.3f}, {t.mqascore_high:.3f}]"
    claim(f"**{set_name} {label}**: CLAP {clap['value']:.3f} [{clap['ci95'][0]:.3f}, {clap['ci95'][1]:.3f}]; "
          f"KL-PaSST {kl['value']:.3f} [{kl['ci95'][0]:.3f}, {kl['ci95'][1]:.3f}]; FD-VGG {mj['FD-VGG_musiccaps']['value']:.2f} (vs all MusicCaps, point value); "
          f"FD-MERT {mj['fd_mert_musiccaps']['value']:.1f}; MQAScore {mqa_abs}; {paired_text} (n=100 prompts)",
          f"results/{full}/metrics.json, results/analysis/findings_table.csv" + (", results/analysis/paired_fd_dev.json" if set_name == "dev" and run != "cfg7" else ""))
note("Notes: MQAScore is over the 98 dev captions that have at least one tag (100 on test). FD-MERT has no paired interval (per-run MERT embeddings are not cached; `paired_fd_dev.json`, key `fd_mert`). "
     "The FD interval in `metrics.json` is a per-run bootstrap, not a difference. dev FD floor for 100 real clips: FD-VGG 1.9, FD-MERT 21 (`docs/findings.md`).")

head("## 2. Caption-word study")
head("### 2a. Probes and detectors")
pv = pd.read_csv(A / "probe_validity.csv", header=[0, 1], index_col=[0, 1])
mert = pv[("mert", "real")].droplevel(1).dropna()
clap = pv[("clap90", "real")].droplevel(1).dropna()
valid = mert[mert >= 0.75]
claim(f"Probe groups tested on held-out real dev clips: {len(mert)}; valid (MERT AUC >= 0.75, rule in `valid_groups`): **{len(valid)}**; "
      f"valid MERT AUC range {valid.min():.2f} to {valid.max():.2f}, median {valid.median():.2f}",
      "results/analysis/probe_validity.csv, scripts/selection/probe_rank.py (valid_groups)")
claim(f"CLAP 90.14 probe AUC >= 0.75 on {int((clap >= 0.75).sum())} of {len(clap)} groups; median {clap.median():.2f}",
      "results/analysis/probe_validity.csv")
best = mert.sort_values(ascending=False)
claim("Highest MERT AUCs: " + "; ".join(f"{g} {v:.2f}" for g, v in best.head(6).items()), "results/analysis/probe_validity.csv")
claim("Invalid groups (MERT AUC < 0.75): " + "; ".join(f"{g} {v:.2f}" for g, v in mert[mert < 0.75].sort_values().items()), "results/analysis/probe_validity.csv")
claim("Validity table, MERT / CLAP AUC per valid group: " + "; ".join(f"{g} {mert[g]:.2f}/{clap[g]:.2f}" for g in valid.index), "results/analysis/probe_validity.csv")
cv = pd.read_csv(A / "attribute_probes_cv.csv")
claim(f"Probe cross-validated AUC on the training pool: {len(cv)} groups, MERT median {cv['mert'].median():.2f}, min {cv['mert'].min():.2f}; "
      f"CLAP median {cv['clap90'].median():.2f}", "results/analysis/attribute_probes_cv.csv")

det = pd.read_csv(A / "dsp_detectors.csv")
for _, r in det[det.check.isin(["tempo", "loudness", "quality"])].iterrows():
    fresh = f", fresh sample AUC {r.auc_b_fresh:.2f} (n={int(r.n_positive_b)}+{int(r.n_negative_b)})" if pd.notna(r.auc_b_fresh) else ""
    claim(f"Detector check `{r.check}`, reader `{r.reader}` (features: {int(r.features)}): CV AUC {r.auc_a_cv:.3f} "
          f"(n={int(r.n_positive_a)}+{int(r.n_negative_a)}){fresh}", "results/analysis/dsp_detectors.csv")

head("### 2b. Word-swap edits: which axes are followed")
note("`moved` = pairs where the detector moved in the predicted direction out of n pairs. Registered = the 12 pre-registered pairs, "
     "Holm over the registered axes; pooled = registered plus the extension pairs (no Holm). Probe = linear probe on MERT or CLAP 90.14 "
     "embeddings of the generated clips, the group of the target word.")
ee = pd.read_csv(A / "edit_effects.csv")
sel = ee[(ee.detector == "probe_mert") | (ee.detector == "probe_clap90")]
for axis in ["guitar_to_piano", "voice_male_to_female", "acoustic_to_electric", "insert_jazz", "insert_jazz_vs_country", "insert_country",
             "soft_to_loud", "happy_to_sad", "fast_to_slow", "low_quality_to_studio"]:
    parts = []
    for det_name in ("probe_mert", "probe_clap90"):
        reg = ee[(ee.axis == axis) & (ee.detector == det_name) & (ee.subset == "registered")].iloc[0]
        pool = ee[(ee.axis == axis) & (ee.detector == det_name) & (ee.subset == "pooled")].iloc[0]
        holm = f", Holm p={reg.p_binomial_holm:.3f}" if pd.notna(reg.p_binomial_holm) else ""
        parts.append(f"{det_name.replace('probe_', '').upper()}: registered {int(reg.moved)}/{int(reg.n)} (p={pv_(reg.p_binomial)}{holm}), "
                     f"pooled {int(pool.moved)}/{int(pool.n)} (p={pv_(pool.p_binomial)}), mean shift {pool['mean']:.2f} [{pool.ci_low:.2f}, {pool.ci_high:.2f}]")
    claim(f"**{axis}**: " + "; ".join(parts), "results/analysis/edit_effects.csv")
note("Hand-crafted detectors on the same pairs (pooled moved/n, p binomial):")
for axis, detector in [("fast_to_slow", "dsp: rhythm_tempo_bpm"), ("fast_to_slow", "reader: all features"), ("fast_to_slow", "reader: interpretable"),
                       ("fast_to_slow", "reader: chroma entropy"), ("soft_to_loud", "dsp: energy_rms_db_mean"), ("soft_to_loud", "reader: interpretable"),
                       ("voice_male_to_female", "dsp stem: pitch_f0_median_energy_gated"), ("low_quality_to_studio", "reader: interpretable"),
                       ("low_quality_to_studio", "reader: single"), ("low_quality_to_studio", "reader: all features"),
                       ("low_quality_to_studio", "dsp: energy_band_6000_12000hz_ratio")]:
    reg = ee[(ee.axis == axis) & (ee.detector == detector) & (ee.subset == "registered")].iloc[0]
    pool = ee[(ee.axis == axis) & (ee.detector == detector) & (ee.subset == "pooled")].iloc[0]
    claim(f"{axis}, `{detector}`: registered {int(reg.moved)}/{int(reg.n)} (p={pv_(reg.p_binomial)}), pooled {int(pool.moved)}/{int(pool.n)} (p={pv_(pool.p_binomial)})",
          "results/analysis/edit_effects.csv")
ev = json.load(open(A / "edit_effects.json"))["verdicts"]
done = [v for v in ev if "moved" in v]
claim(f"Registered predictions with a result: {len(done)}; met {sum(v['met'] for v in done)}, missed {sum(not v['met'] for v in done)} "
      f"(missed: " + ", ".join(f"{v['axis']} / {v['detector']} moved {v['moved']}/{v['n']}, predicted {v['predicted'][0]}-{v['predicted'][1]}" for v in done if not v["met"]) + ")",
      "results/analysis/edit_effects.json")
claim(f"Registered pairs not run (result null): {sum('moved' not in v for v in ev)} ("
      + ", ".join(f"{v['axis']} / {v['detector']}" for v in ev if "moved" not in v) + ")", "results/analysis/edit_effects.json")

head("### 2c. Meaning-free controls and word vs seed (re-roll share)")
note("`profile_change_vs_seed` = change of the 22-group probe profile under the edit divided by the change under a new seed; "
     "`groups_clear_of_zero` = groups whose 95% interval excludes 0 (of 22). Controls: `control_song_to_track` (word swap without meaning) and `control_neutral_insert`.")
ec = pd.read_csv(A / "edit_controls.csv")
for kind in ("mert", "clap90"):
    sub = ec[ec.kind == kind]
    ctrl, real = sub[sub.control], sub[~sub.control]
    claim(f"{kind.upper()}: meaning-free controls change the profile {ctrl.profile_change_vs_seed.min():.2f} to {ctrl.profile_change_vs_seed.max():.2f} of a seed change "
          f"(signed shifts clear of zero: {', '.join(str(x) for x in ctrl.groups_clear_of_zero)} of 22); real edits {real.profile_change_vs_seed.min():.2f} to "
          f"{real.profile_change_vs_seed.max():.2f} (clear groups up to {int(real.groups_clear_of_zero.max())})", "results/analysis/edit_controls.csv")
    claim(f"{kind.upper()} per edit (change vs seed / clear groups, n): " + "; ".join(
        f"{r.edit} {r.profile_change_vs_seed:.2f}/{int(r.groups_clear_of_zero)}, n={int(r.n)}" for r in sub.itertuples()), "results/analysis/edit_controls.csv")
for axis in ["guitar_to_piano", "voice_male_to_female", "acoustic_to_electric", "insert_jazz", "insert_country", "fast_to_slow", "low_quality_to_studio"]:
    r = ee[(ee.axis == axis) & (ee.detector == "probe_mert") & (ee.subset == "pooled")].iloc[0]
    c = ee[(ee.axis == axis) & (ee.detector == "probe_clap90") & (ee.subset == "pooled")].iloc[0]
    claim(f"{axis}: target-probe edit change over seed change (`edit_abs_vs_seed_abs`) MERT {r.edit_abs_vs_seed_abs:.2f}, CLAP {c.edit_abs_vs_seed_abs:.2f} (n={int(r.n)})",
          "results/analysis/edit_effects.csv")

head("### 2d. Spillover bundles")
sp = pd.read_csv(A / "edit_spillover.csv")
sp["clear"] = (sp.ci_low > 0) | (sp.ci_high < 0)
claim(f"Spillover table: {len(sp)} axis x group cells; {int(sp.clear.sum())} have a 95% interval that excludes 0 (about {len(sp) * 0.05:.0f} expected by chance); "
      f"{sp.axis.nunique()} axes, {sp.group.nunique()} groups", "results/analysis/edit_spillover.csv")
for axis, group, label in [("soft_to_loud", "timbre: groovy, punchy", "loud adds groove"), ("soft_to_loud", "instrument: drums, percussion", "loud adds drums"),
                           ("soft_to_loud", "vocals: male voice", "loud adds male voice"), ("soft_to_loud", "mood: happy, bright", "loud adds happy"),
                           ("happy_to_sad", "instrument: drums, percussion", "sad removes drums"), ("happy_to_sad", "timbre: groovy, punchy", "sad removes groovy"),
                           ("acoustic_to_electric", "mood: sad, emotional", "acoustic to electric removes sad"),
                           ("acoustic_to_electric", "instrument: synth, electronic sounds", "acoustic to electric adds synth"),
                           ("guitar_to_piano", "timbre: acoustic", "guitar to piano removes acoustic"), ("guitar_to_piano", "instrument: bass", "guitar to piano removes bass"),
                           ("guitar_to_piano", "timbre: electric, distorted", "guitar to piano removes electric"),
                           ("guitar_to_piano", "genre: latin, reggae, world", "guitar to piano removes latin"),
                           ("insert_country", "timbre: acoustic", "country adds acoustic"), ("insert_country", "genre: hip hop, r&b, funk", "country adds hip hop"),
                           ("insert_country", "genre: latin, reggae, world", "country adds latin")]:
    r = sp[(sp.axis == axis) & (sp.group == group)].iloc[0]
    claim(f"{label} ({group}): {r.change_vs_seed_sd:+.2f} seed SD, raw change {ci(r.mean_change, r.ci_low, r.ci_high, 2)}, up in {r.share_up * r.n:.0f} of {int(r.n)} pairs; "
          f"interval {'excludes' if r.clear else 'includes'} 0", "results/analysis/edit_spillover.csv")
for axis in ("soft_to_loud", "happy_to_sad"):
    r = ee[(ee.axis == axis) & (ee.detector == "probe_mert") & (ee.subset == "pooled")].iloc[0]
    claim(f"{axis}: its own target probe on MERT, pooled {int(r.moved)}/{int(r.n)}, mean {ci(r['mean'], r.ci_low, r.ci_high, 2)} (interval includes 0)", "results/analysis/edit_effects.csv")
r = ee[(ee.axis == "insert_country") & (ee.detector == "probe_mert") & (ee.subset == "pooled")].iloc[0]
claim(f"country inserted: country not rendered as country, MERT probe moved {int(r.moved)}/{int(r.n)} (p={r.p_binomial:.3f})", "results/analysis/edit_effects.csv")

head("### 2e. Bundle origin: caption vs real-audio co-occurrence")
bo = json.load(open(A / "bundle_origin.json"))
claim(f"Spillover vs caption co-occurrence Spearman {bo['caption_cooccurrence']['spearman']:.2f} (p={bo['caption_cooccurrence']['p']:.1e}); "
      f"vs real-audio co-occurrence {bo['audio_cooccurrence']['spearman']:.2f} (p={bo['audio_cooccurrence']['p']:.1e}); n={bo['pairs']} axis x group pairs",
      "results/analysis/bundle_origin.json")
claim(f"The two co-occurrence measures correlate {bo['inter_association_spearman']:.2f} with each other", "results/analysis/bundle_origin.json")
claim(f"Sign agreement on the {bo['clear_pairs']} pairs with a clear spillover: {bo['caption_cooccurrence_sign_agreement_on_clear']:.2f} (caption), "
      f"{bo['audio_cooccurrence_sign_agreement_on_clear']:.2f} (audio); linear fit R2 {bo['linear_fit']['r2']:.2f}", "results/analysis/bundle_origin.json")
bc = pd.read_csv(A / "bundle_origin.csv")
for g in ("genre: hip hop, r&b, funk", "genre: latin, reggae, world", "timbre: acoustic"):
    r = bc[(bc.axis == "insert_country") & (bc.group == g)].iloc[0]
    claim(f"Exception check, country insert -> {g}: spillover {r.spillover:+.2f} SD, caption co-occurrence {r.caption_cooccurrence:+.2f}, audio co-occurrence {r.audio_cooccurrence:+.2f} "
          f"(target clips {int(r.n_target_clips)}, source clips {int(r.n_source_clips)})", "results/analysis/bundle_origin.csv")
note("Caveat in the file: the 'audio' side uses the same probes (scripts/analysis/bundle_origin.py help text).")

head("### 2f. Probe leakage (drums probe vs level)")
pl = json.load(open(A / "probe_leakage.json"))
claim(f"Drums probe score on real-clip level alone: {pl['drums_score_on_level_alone']:.2f} SD per SD of RMS; given percussion features {pl['drums_score_on_level_given_percussion']:.2f} (n={pl['real_clips']} real clips)",
      "results/analysis/probe_leakage.json")
for axis, label in (("soft_to_loud", "loud"), ("happy_to_sad", "sad"), ("voice_male_to_female", "voice swap")):
    a = pl["axes"][axis]
    claim(f"{label}: drums probe change {a['drums_probe_change_sd']:+.2f} SD; predicted from level alone {a['drums_change_predicted_from_level_alone_sd']:+.2f} "
          f"({a['drums_change_predicted_from_level_alone_sd'] / a['drums_probe_change_sd'] * 100:.0f}% of it); level change {a['level_change_db']:+.2f} dB; "
          f"percussive share change {a['percussive_share_change_sd']:+.2f} SD (up in {a['percussive_share_share_up'] * a['pairs']:.0f} of {a['pairs']} pairs); "
          f"percussive onset rate {a['onset_rate_percussive_change_sd']:+.2f} SD (n={a['pairs']} pairs)", "results/analysis/probe_leakage.json")

head("### 2g. Recording-quality words")
claim(f"Quality reader (`interpretable`, 5 features) CV AUC {det[(det.check == 'quality') & (det.reader == 'interpretable')].auc_a_cv.iloc[0]:.3f} on real audio "
      f"(83 low quality vs 150 others); single-feature reader {det[(det.check == 'quality') & (det.reader == 'single')].auc_a_cv.iloc[0]:.3f}; all 110 features "
      f"{det[(det.check == 'quality') & (det.reader == 'all features')].auc_a_cv.iloc[0]:.3f}", "results/analysis/dsp_detectors.csv")
for det_name, label in (("reader: interpretable", "quality reader"), ("probe_mert", "MERT probe"), ("probe_clap90", "CLAP probe")):
    reg = ee[(ee.axis == "low_quality_to_studio") & (ee.detector == det_name) & (ee.subset == "registered")].iloc[0]
    pool = ee[(ee.axis == "low_quality_to_studio") & (ee.detector == det_name) & (ee.subset == "pooled")].iloc[0]
    claim(f"Swap low quality to studio, {label}: registered {int(reg.moved)}/{int(reg.n)}, pooled {int(pool.moved)}/{int(pool.n)} (pooled p={pool.p_binomial:.3f})", "results/analysis/edit_effects.csv")
ab = json.load(open(A / "ablate_dev_scores.json"))
cl = ab["clean recording words"]
claim(f"Removing the recording and setting sentences ({cl['n']} dev captions), scored against the original caption: CLAP {ci(cl['clap_cosine']['mean'], cl['clap_cosine']['ci_low'], cl['clap_cosine']['ci_high'])}; "
      f"MQAScore {ci(cl['mqascore']['mean'], cl['mqascore']['ci_low'], cl['mqascore']['ci_high'])}; KL-PaSST {ci(cl['kl_passt_softmax']['mean'], cl['kl_passt_softmax']['ci_low'], cl['kl_passt_softmax']['ci_high'], 2)} (n={cl['n']})",
      "results/analysis/ablate_dev_scores.json")
cj = json.load(open(A / "clean_dev.json"))
ids = [int(i) for i in cj["ids"]]
seed_abs = np.mean([(per_item(f"dev__rescale12_seed{k}", "per_prompt.csv", "clap_cosine") - per_item("dev__rescale12", "per_prompt.csv", "clap_cosine")).loc[ids].abs().mean() for k in (1, 2)])
claim(f"Mean |dCLAP| on the same {len(ids)} ids: cleaning {cl['clap_cosine']['mean_abs']:.3f} vs a new seed {seed_abs:.3f} (computed, mean over seeds 1 and 2 vs seed 0 of rescale12; same recipe as `scripts/analysis/ablate_followed.py`)",
      "results/analysis/ablate_dev_scores.json, results/dev__rescale12*/per_prompt.csv, results/analysis/clean_dev.json")
claim(f"Power limit: the CLAP interval half-width is {(cl['clap_cosine']['ci_high'] - cl['clap_cosine']['ci_low']) / 2:.3f} (about +-0.04)", "results/analysis/ablate_dev_scores.json")
claim(f"Full-100 run with cleaned captions vs rescale12: " + (verdict("dev__rescale12_clean", "dev__rescale12", "clap") or "n/a") + "; "
      + (verdict("dev__rescale12_clean", "dev__rescale12", "mqa") or "n/a") + "; " + (verdict("dev__rescale12_clean", "dev__rescale12", "kl") or "n/a"), "results/analysis/knowledge.md")
claim(f"Cleaning drops {cj['changed']} captions' sentences (groups: {', '.join(cj['drop_groups'])}); median words {cj['median_words_before']:.0f} to {cj['median_words_after']:.0f}",
      "results/analysis/clean_dev.json")
dg = pd.read_csv(A / "caption_vocab_groups.csv")
for r in dg[dg.group.isin(["low quality", "high quality"])].itertuples():
    claim(f"Caption words for `{r.group}`: MusicCaps {r.musiccaps_pct}% of captions, dev {r.dev_captions}, test {r.test_captions} captions (of 100 each); `hear` flag in file: {r.hear}", "results/analysis/caption_vocab_groups.csv")

head("### 2h. Chroma check")
cc = pd.read_csv(A / "chroma_check.csv")
note("Features are computed on the HPSS harmonic part only (percussive part removed), so the plan's 'drums removed' means 'percussive part removed'.")
for check, feature in [("tempo", "harmony_chroma_entropy_mean"), ("loudness", "harmonic_frame_entropy"), ("jazz vs country", "harmonic_profile_entropy"),
                       ("jazz vs country", "harmony_chroma_entropy_mean"), ("electric vs acoustic", "harmonic_profile_entropy"), ("piano vs guitar", "harmony_chroma_entropy_mean"),
                       ("happy vs sad", "harmony_chroma_entropy_mean")]:
    r = cc[(cc.check == check) & (cc.feature == feature)].iloc[0]
    claim(f"`{check}`, `{feature}`: AUC {r.auc:.3f} (n={int(r.n)}); control-only CV AUC {r.auc_controls_cv:.3f}, with the feature {r.auc_controls_plus_feature_cv:.3f}, gain {r.gain:+.3f}",
          "results/analysis/chroma_check.csv")
for axis in ("insert_jazz", "insert_country", "guitar_to_piano", "fast_to_slow"):
    parts = []
    for d_ in ("chroma: harmonic_profile_entropy", "chroma: harmonic_frame_entropy"):
        r = ee[(ee.axis == axis) & (ee.detector == d_) & (ee.subset == "pooled")].iloc[0]
        parts.append(f"{d_.split(': ')[1]} {int(r.moved)}/{int(r.n)} (p={r.p_binomial:.4f})")
    claim(f"Word-swap {axis}, chroma moved (pooled): " + "; ".join(parts), "results/analysis/edit_effects.csv")

head("### 2i. Specificity (target moves, unrelated groups move like seed noise)")
note("`target_rise`, `source_drop`: mean change of the target and source group probe scores (probe score units of the file); `unrelated_mean_abs` and `unrelated_seed_abs`: mean absolute change of the other probe groups under the edit and under a new seed; ratio = `unrelated_edit_vs_seed`.")
for axis in ["guitar_to_piano", "voice_male_to_female", "acoustic_to_electric", "insert_jazz", "insert_country", "soft_to_loud", "happy_to_sad", "fast_to_slow", "low_quality_to_studio"]:
    r = ee[(ee.axis == axis) & (ee.detector == "probe_mert") & (ee.subset == "pooled")].iloc[0]
    claim(f"{axis} (MERT, pooled n={int(r.n)}): target_rise {r.target_rise:.2f}, source_drop {r.source_drop:.2f}, unrelated mean |change| {r.unrelated_mean_abs:.2f} vs seed {r.unrelated_seed_abs:.2f} "
          f"(ratio {r.unrelated_edit_vs_seed:.2f})", "results/analysis/edit_effects.csv")

head("## 3. Positive control (ablate) and why CLAP, MQAScore and KL differ")
fw = ab["ablate followed words"]
claim(f"Dropping a sentence with a followed word (guitar, piano, voice, acoustic, electric), {fw['n']} captions, generated clip scored against the original caption: CLAP {ci(fw['clap_cosine']['mean'], fw['clap_cosine']['ci_low'], fw['clap_cosine']['ci_high'])} "
      f"(mean |change| {fw['clap_cosine']['mean_abs']:.3f}); MQAScore {ci(fw['mqascore']['mean'], fw['mqascore']['ci_low'], fw['mqascore']['ci_high'])}; "
      f"KL-PaSST {ci(fw['kl_passt_softmax']['mean'], fw['kl_passt_softmax']['ci_low'], fw['kl_passt_softmax']['ci_high'], 2)} (mean |change| {fw['kl_passt_softmax']['mean_abs']:.2f})",
      "results/analysis/ablate_dev_scores.json")
rg = ab["removed group probe change (seed SD)"]
claim(f"Removed-group probe {ci(rg['mean'], rg['ci_low'], rg['ci_high'], 2)} seed SD; {rg['share_down'] * 100:.0f}% of {rg['n']} removed groups go down", "results/analysis/ablate_dev_scores.json")
claim(f"Seed change mean |dCLAP| on the ablated ids: {ab['seed change mean_abs clap on ablated ids']:.4f}", "results/analysis/ablate_dev_scores.json")
claim("Full 100-caption run `dev__rescale12_ablate` vs `dev__rescale12`: " + "; ".join(verdict("dev__rescale12_ablate", "dev__rescale12", m) or "n/a" for m in ("clap", "mqa")), "results/analysis/knowledge.md")
tg = pd.read_csv(A / "ablate_mqa_tags.csv")
for r in tg.itertuples():
    claim(f"MQAScore per-tag, `{r.tag_kind}` tags: change {ci(r.mean, r.ci_low, r.ci_high, 2)} (n={int(r.n)} tags, mean before {r.before_mean:.2f})", "results/analysis/ablate_mqa_tags.csv")
at = pd.read_csv(A / "ablate_mqa_attributes.csv")
for r in at[at.measure.isin(["mqascore", "instrument_mean", "vocal_mean"])].itertuples():
    claim(f"After removing the {r.removed} sentence, caption `{r.measure}` {ci(r.mean, r.ci_low, r.ci_high, 2)} (n={int(r.n)} captions; `tag_share` {r.tag_share})", "results/analysis/ablate_mqa_attributes.csv")
mq = pd.read_csv(R / "dev__rescale12/mqa_per_item.csv").set_index("id")
ab_ids = json.load(open(A / "ablate_dev.json"))["ids"]
claim(f"Tag count per caption: mean {mq.loc[ab_ids].n_tags.mean():.1f} on the {len(ab_ids)} ablated captions ({mq.n_tags.mean():.1f} over all 100), so one removed tag is about 1 of {mq.loc[ab_ids].n_tags.mean():.0f}",
      "results/dev__rescale12/mqa_per_item.csv, results/analysis/ablate_dev.json")
md = pd.read_csv(A / "mqa_drivers.csv")
for _, r in md.iterrows():
    claim(f"MQAScore driver `{r['attribute']}`: {r['tag_share'] * 100:.0f}% of tags, mean tag score {r['mean_score']:.2f}, share of tags above 0.5 {r['share_yes_above_0.5'] * 100:.0f}%, "
          f"score variance share across all clips {r['variance_share_all_clips'] * 100:.0f}%, across seeds of one caption {r['variance_share_across_seeds'] * 100:.0f}% (300 clips: dev rescale12 seeds 0-2)",
          "results/analysis/mqa_drivers.csv")
claim("MQAScore asks only genre, instrument, mood_theme, vocal and context tags; tempo and production-quality tags are not asked (`attribute` column lists exactly these five)", "results/analysis/mqa_drivers.csv")

head("## 4. Overnight runs, each vs its baseline")
note("All paired differences are computed from per-item files (2000 resamples, seed 0, over prompt ids; MQAScore over the prompts with at least one tag). "
     "`pending` = the run has no per-item MQAScore file.")
comps = [("dev__rescale12_followed", "dev__rescale12", "followed words push (caption weight 12, extra term)"),
         ("dev__rescale12_followed_sum12", "dev__rescale12", "followed words push, weights summed to 12"),
         ("dev__rescale16", "dev__rescale12", "caption weight 16, no change of words"),
         ("dev__rescale12_followed", "dev__rescale16", "followed push vs equal total weight"),
         ("dev__rescale12_loop2", "dev__rescale12_best3_probe", "round 2 emphasis on the same seed vs round 1 pick")]
for run, base, label in comps:
    sources = f"results/{run}/per_prompt.csv, results/{run}/mqa_per_item.csv, results/{base}/per_prompt.csv, results/analysis/knowledge.md"
    claim(f"**{run} vs {base}** ({label}): " + fmt_pair(run, base), sources)
    for metric in ("clap", "kl", "mqa"):
        v = verdict(run, base, metric)
        if v:
            claim(f"verdict line ({metric}): {v}", "results/analysis/knowledge.md")
for run, base in [("dev__rescale12_followed", "dev__cfg7"), ("dev__rescale12_followed_sum12", "dev__cfg7"), ("dev__rescale16", "dev__cfg7"), ("dev__rescale12_loop2", "dev__cfg7")]:
    f_ = fd[run.replace("dev__", "")]["VGG"]
    claim(f"{run} vs {base}: dFD-VGG {ci(f_['difference'], *f_['difference_ci95'], d=2)} (paired FD is only available vs cfg7)", "results/analysis/paired_fd_dev.json")
lj = json.load(open(A / "loop_dev.json"))
ls = pd.read_csv(R / "dev__rescale12_loop2/loop_selection.csv")
claim(f"Round 2 changed {lj['changed']} of 100 captions ({lj['groups']} probe groups checked); emphasis not added for token limit in {lj['emphasis_not_added_token_limit']}; "
      f"round-2 clips kept (score above round 1) for {int(ls.kept_round2.sum())} of {lj['changed']} changed prompts = {ls.kept_round2.sum() / lj['changed']:.3f} "
      f"(knowledge.md `kept_fraction` 0.303); {int(ls.kept_round2.sum())} of all {len(ls)} prompts = {ls.kept_round2.mean():.3f}",
      "results/analysis/loop_dev.json, results/dev__rescale12_loop2/loop_selection.csv, results/analysis/knowledge.md")
claim("Most-missing groups in round 1 (count of captions): " + "; ".join(f"{g} {n}" for g, n in list(lj["missing_per_group"].items())[:5]), "results/analysis/loop_dev.json")
fj = json.load(open(A / "followed_dev.json"))
claim(f"Followed-word push: {fj['with_phrase']} of 100 dev captions have a followed-word phrase; median {fj['median_phrases']:.0f} phrases per caption", "results/analysis/followed_dev.json")
note("**Bundle cut** (negation formula, caption weight kept at 12; `bundle` arm = extra term away from (soft_to_loud) or toward (happy_to_sad) the bundle text, `neutral` arm = meaning-free extra text at the same weight). "
     "Effects in seed-SD units against the plain edited clip; `edit_effect_sd` = the edit's own effect in the same units.")
bg = pd.read_csv(A / "bundle_guidance.csv")
for (kind, axis), grp in bg.groupby(["kind", "axis"], sort=False):
    parts = []
    for r in grp.itertuples():
        parts.append(f"{r.arm} / {r.measure}: {ci(r.change_vs_edited_sd, r.ci_low_sd, r.ci_high_sd, 2)} (edit effect {r.edit_effect_sd:+.2f})")
    claim(f"{kind.upper()} {axis} (n={int(grp.n.iloc[0])} pairs, extra sign {int(grp.extra_sign.iloc[0]):+d}): " + "; ".join(parts), "results/analysis/bundle_guidance.csv")
for (kind, axis), grp in bg.groupby(["kind", "axis"], sort=False):
    tm = grp[grp.measure == "target minus source"].set_index("arm")
    other = grp[grp.measure != "target minus source"]
    bundle_clear = other[(other.arm == "bundle") & ((other.ci_low_sd > 0) | (other.ci_high_sd < 0))]
    neutral_clear = other[(other.arm == "neutral") & ((other.ci_low_sd > 0) | (other.ci_high_sd < 0))]
    b, n_ = tm.loc["bundle"], tm.loc["neutral"]
    claim(f"Reading, {kind.upper()} {axis}: bundle arm moves the bundle probes clearly in {len(bundle_clear)} of {len(other[other.arm == 'bundle'])} groups, neutral arm in {len(neutral_clear)}; "
          f"target-minus-source (the word's own attribute) in the bundle arm {'excludes' if (b.ci_low_sd > 0 or b.ci_high_sd < 0) else 'includes'} 0, in the neutral arm {'excludes' if (n_.ci_low_sd > 0 or n_.ci_high_sd < 0) else 'includes'} 0",
          "results/analysis/bundle_guidance.csv")

head("## 5. Rankers and best-of-N")
rc = pd.read_csv(A / "ranker_compare.csv")
note("Each ranker's best-of-3 vs the seed-0 run of the same setting (rescale12): paired change in CLAP (90.14 scorer), KL-PaSST and MQAScore; `same pick` = share of prompts where two rankers pick the same seed (chance 1/3).")
for r in rc.itertuples():
    claim(f"{r.set} `{r.ranker}`: dCLAP {ci(r.clap, r.clap_ci_low, r.clap_ci_high)} (n={r.clap_n}); dKL {ci(r.kl, r.kl_ci_low, r.kl_ci_high)} (n={r.kl_n}); "
          f"dMQAScore {ci(r.mqa, r.mqa_ci_low, r.mqa_ci_high)} (n={r.mqa_n}); same pick as c8998 {r.same_pick_as_best3_c8998:.2f}, as probe {r.same_pick_as_best3_probe:.2f}, as MQAScore {r.same_pick_as_best3_mqa:.2f}",
          "results/analysis/ranker_compare.csv")
same = rc[["same_pick_as_best3_c8998", "same_pick_as_best3_probe", "same_pick_as_best3_mqa"]].to_numpy().ravel()
same = same[same < 1.0]
claim(f"Between-ranker agreement on the same pick: {same.min():.2f} to {same.max():.2f} over all pairs and both sets (chance 0.33)", "results/analysis/ranker_compare.csv")
re_ = pd.read_csv(A / "ranker_explain.csv")
note("Within-caption rank correlations (scores centred per caption over 3 seeds; Spearman; dev n=98 captions = 294 clips, test n=100 = 300 clips; |rho| below about 0.12 is noise):")
for set_name in ("dev", "test"):
    s = re_[re_.set == set_name].set_index("score")
    pairs = [("clap8998", "eval_clap"), ("clap8998", "mqa"), ("mqa", "eval_clap"), ("probe", "clap8998"), ("probe", "mqa"), ("probe", "eval_clap")]
    txt = "; ".join(f"{a} vs {b} {s.loc[a, b]:+.2f}" for a, b in pairs)
    claim(f"{set_name} rank agreement: {txt} (`eval_clap` = the 90.14 scoring CLAP, `clap8998` = the 89.98 ranker CLAP, `probe` = probe ranker)", "results/analysis/ranker_explain.csv")
    feats = ["rms_db", "centroid_hz", "percussive_share", "onset_rate_mix", "flatness", "share_above_12khz", "harmony_chroma_entropy_mean", "crest_factor_db"]
    for score in ("mqa", "eval_clap", "probe", "clap8998"):
        vals = {f: s.loc[score, f] for f in feats if f in s.columns and pd.notna(s.loc[score, f])}
        top = sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:3]
        claim(f"{set_name} `{score}` ranker, strongest feature links: " + "; ".join(f"{k} {v:+.2f}" for k, v in top), "results/analysis/ranker_explain.csv")
allf = re_[["rms_db", "centroid_hz", "percussive_share", "onset_rate_mix", "flatness", "share_above_12khz", "harmony_chroma_entropy_mean", "crest_factor_db"]].abs().to_numpy()
claim(f"Largest |rho| of any judge with any acoustic feature (both sets): {np.nanmax(allf):.2f}", "results/analysis/ranker_explain.csv")
bn = pd.read_csv(A / "best_of_n_vs_seed_mean.csv")
note("Best-of-3 vs the mean of the 3 seeds per prompt (a random pick), paired; `dclap9014` = scoring checkpoint, `dclap8998` = other checkpoint:")
for r in bn.itertuples():
    claim(f"{r.set} {r.setting}, ranker {r.ranker}: dCLAP 90.14 {ci(r.dclap9014, r.dclap9014_low, r.dclap9014_high)}; dCLAP 89.98 {ci(r.dclap8998, r.dclap8998_low, r.dclap8998_high)}; "
          f"dKL-PaSST {ci(r.dkl_passt, r.dkl_passt_low, r.dkl_passt_high, 2)}; picks per seed index {r.picks_seed0}/{r.picks_seed1}/{r.picks_seed2}", "results/analysis/best_of_n_vs_seed_mean.csv")
cm = pd.read_csv(A / "clap_margin.csv")
note("CLAP gain split (`down` = own-caption cosine change, `dother` = change of mean cosine to the other 99 captions, `dmargin` = own minus other; definitions in `scripts/analysis/review_checks/clap_margin.py`):")
for r in cm.itertuples():
    claim(f"{r.set} {r.run} vs {r.baseline}: down {ci(r.down, r.down_low, r.down_high)}; dother {ci(r.dother, r.dother_low, r.dother_high)}; dmargin {ci(r.dmargin, r.dmargin_low, r.dmargin_high)}", "results/analysis/clap_margin.csv")
sj = json.load(open(R / "test__rescale12_best3_c8998/selection.json"))
cc_ = sj["chosen_counts"]
claim(f"Submission picks on test (seed 0 / 1 / 2): {cc_['rescale12']} / {cc_['rescale12_seed1']} / {cc_['rescale12_seed2']} of {sj['n_items']}; ranker score `{sj['score']}`", "results/test__rescale12_best3_c8998/selection.json")

head("## 6. Prediction tally")
met = [v for v in verdicts if v["v"] == "met"]
missed = [v for v in verdicts if v["v"] == "MISSED"]
pending = [ln for ln in knowledge[knowledge.index("Pending:") + 1:] if ln.startswith("  dev__") or ln.startswith("  test__")]
claim(f"Predictions with a result: {len(verdicts)}; **met {len(met)}, missed {len(missed)}**; pending {len(pending)}", "results/analysis/knowledge.md")
for v in missed:
    claim(f"MISSED `{v['run']}` vs `{v['base']}`, {v['metric']} predicted [{v['range']}] (written {v['written']}): got {v['got']}", "results/analysis/knowledge.md")


head("## Mismatches")
note("Plan value vs file value, plus plan items whose file does not exist yet. None of these changes a conclusion.")
for text in [
    "Cleaning (recording and setting sentences removed, 29 captions) CLAP interval: plan `-0.012 [-0.054, +0.026]`; file `-0.012 [-0.052, +0.027]`; `results/analysis/ablate_dev_scores.json` (key `clean recording words`).",
    "Cleaning MQAScore interval: plan `+0.015 [-0.022, +0.052]`; file `+0.015 [-0.024, +0.051]`; `results/analysis/ablate_dev_scores.json`.",
    "Ranker agreement 'MQAScore and CLAP agree only weakly (0.19 / 0.24)': those are MQAScore vs the 89.98 ranker CLAP (`clap8998`). Against the 90.14 scoring CLAP (`eval_clap`) the values are 0.07 dev and 0.16 test; `results/analysis/ranker_explain.csv`. Say which CLAP on the slide.",
    "Loop round 2 `kept_fraction`: knowledge.md 0.303 is 27 of the 89 changed captions; 27 of all 100 prompts is 0.270; `results/dev__rescale12_loop2/loop_selection.csv`, `results/analysis/loop_dev.json`.",
    "Slide 8 cites `results/dev__rescale12_best5_*`: no such result folders exist (best-of-5 runs are still pending in knowledge.md); `dev__rescale12_followed_sentence` is also missing. No numbers can be given for them.",
    "Test paired FD-VGG: `paired_fd_dev.json` covers dev only, so the test rows have no paired dFD-VGG.",
    "dev__rescale12_best3_probe has no `mqa_per_item.csv` and no `metrics.json`, so loop2 vs best3_probe has no MQAScore and `findings_table.csv` shows its CLAP as 0.000 (pending); the CLAP and KL differences above come from `per_prompt.csv`.",
    "Caveat, 'followed' axes: with Holm over the 12 registered pairs only acoustic_to_electric on MERT is significant (Holm p=0.025). Guitar to piano, voice and jazz are followed on the pooled counts (extension pairs added, no Holm), so label them 'pooled'. Probe-based mood and dynamics results are weak on MERT (happy_to_sad pooled interval includes 0; soft_to_loud interval includes 0) but soft_to_loud is clear on CLAP (15/16).",
    "Caveat, plan wording 'drums removed' for the chroma check: the file removes the percussive part by HPSS (harmonic part only), not by a drum stem; `scripts/analysis/chroma_check.py`.",
]:
    lines.append(f"- {text}")

OUT.write_text("\n".join(lines) + "\n")
print(claims[0], "claims")
