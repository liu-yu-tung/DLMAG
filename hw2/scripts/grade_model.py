"""Listening-grade predictor: a linear score over piano-roll features (`hw2.grade_features`), fit on the blind A/B
pairs and the pilot 1-5 scores, checked by leave-one-out and on each new session before it is fit.

uv run python scripts/grade_model.py check --session t3_continuations
uv run python scripts/grade_model.py fit
uv run python scripts/grade_model.py predict outputs/cand_a/* --out results/grade_pred_cand_a.csv
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from hw2.grade_features import FEATURES, piece_features
from hw2.remi import HW2_ROOT

LISTENING = HW2_ROOT / "results" / "listening"
RESULTS = HW2_ROOT / "results"
REFS = ["outputs/ref/heldout_open32", "outputs/ref/heldout_crop32"]
# v1: chosen from the piano rolls before any fit; v2: v1 after the single-feature screen of 2026-10-17 (T1, T2, pilot);
# v3: v1 plus the harmony checks from the T3 comments (key drift from the opening or prompt key, semitone clashes),
# built 2026-10-17 after T3, so only sessions graded later test it. tempo_changes and velocity_spike_share stay out:
# they mostly tell real songs and systems apart.
VARIANTS = {
    "v1": ["longest_onset_gap_beats", "bass_dropout", "high_share", "opening_high_share", "half_rhythm_sim",
           "half_register_shift", "loop_run", "key_fit", "density_cv"],
    "v2": ["longest_onset_gap_beats", "bass_dropout", "high_share", "half_pc_sim", "half_register_shift",
           "one_note_bars", "key_fit", "density_cv"],
    "v3": ["longest_onset_gap_beats", "bass_dropout", "high_share", "opening_high_share", "half_rhythm_sim",
           "half_register_shift", "loop_run", "key_fit", "density_cv", "opening_key_fit_min4", "semitone_clash_share"],
}
TAG_FEATURES = {"silent_gap": "longest_onset_gap_beats", "too_repetitive": "loop_run", "abrupt_start": "opening_high_share",
                "wrong_notes": "key_fit", "melody_missing": "bass_dropout", "odd_tempo": "tempo", "cut_off_ending": "density_cv",
                "timing_off": "density_cv"}
LAMBDAS = [0.1, 1.0, 10.0]


def model_path(variant: str) -> Path:
    return RESULTS / f"grade_model_{variant}.json"


def midi_path(source: str, file: str) -> str:
    return str(Path(source) / Path(file).with_suffix(".mid"))


def load_grades() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A/B pairs (one row per pair, `a_better` 1/0, NaN for same) and scored clips (one row per clip)."""
    pairs, scored = [], []
    for csv in sorted(LISTENING.glob("*.csv")):
        df = pd.read_csv(csv)
        if {"source_a", "source_b", "pref"} <= set(df.columns):
            for _, r in df.iterrows():
                pairs.append({"session": r["session"], "item": r["pair"], "a": midi_path(r["source_a"], r["file_a"]),
                              "b": midi_path(r["source_b"], r["file_b"]), "pref": r["pref"],
                              "a_better": np.nan if r["pref"] == 0 else float(r["pref"] < 0),
                              "tags_a": r.get("tags_a"), "tags_b": r.get("tags_b")})
        elif {"source", "file", "overall"} <= set(df.columns):
            for _, r in df.iterrows():
                scored.append({"session": r["session"], "item": r["clip"], "path": midi_path(r["source"], r["file"]),
                               "overall": r["overall"], "tags": r.get("tags")})
    return pd.DataFrame(pairs), pd.DataFrame(scored).dropna(subset=["overall"])


def first_bar_of(folder: Path) -> int:
    """TASK B sets (a prompt file or folder in gen_info.json) start generating after the prompt bars."""
    info = folder / "gen_info.json"
    args = json.loads(info.read_text()).get("args", {}) if info.exists() else {}
    return int(args.get("prompt_bars") or 8) if args.get("prompts") or args.get("prompt_dir") else 0


def features_of(path: Path) -> dict:
    return piece_features(path, first_bar=first_bar_of(path.parent))


def feature_table(paths) -> pd.DataFrame:
    return pd.DataFrame([{"path": p, **features_of(HW2_ROOT / p)} for p in sorted(set(paths))]).set_index("path")


def training_pairs(pairs: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """A/B pairs with weight 1; pilot clips paired within their session where scores differ by 1 or more, weighted so
    each clip counts about once."""
    rows = [{"a": r.a, "b": r.b, "y": r.a_better, "w": 1.0, "group": f"{r.session}/{r.item}"}
            for r in pairs.dropna(subset=["a_better"]).itertuples()]
    for session, g in scored.groupby("session"):
        w = 1.0 / max(1, len(g) - 1)
        for x, z in combinations(g.itertuples(), 2):
            if abs(x.overall - z.overall) >= 1:
                rows.append({"a": x.path, "b": z.path, "y": float(x.overall > z.overall), "w": w, "group": f"pilot/{session}"})
    return pd.DataFrame(rows)


def fit_weights(d: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    def loss(beta):
        m = d @ beta
        return (w * (np.logaddexp(0, m) - y * m)).sum() + lam * beta @ beta

    def grad(beta):
        return d.T @ (w * (1 / (1 + np.exp(-(d @ beta))) - y)) + 2 * lam * beta

    return minimize(loss, np.zeros(d.shape[1]), jac=grad, method="L-BFGS-B").x


def standardize(feats: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean, std = feats[cols].mean(), feats[cols].std().replace(0, 1)
    return ((feats[cols] - mean) / std).fillna(0), mean, std


def cross_validate(train: pd.DataFrame, z: pd.DataFrame, scored: pd.DataFrame, cols: list[str], lam: float) -> list[dict]:
    rows = []
    diffs = z.loc[train["a"], cols].to_numpy() - z.loc[train["b"], cols].to_numpy()
    for group in train["group"].unique():
        test = (train["group"] == group).to_numpy()
        beta = fit_weights(diffs[~test], train["y"].to_numpy()[~test], train["w"].to_numpy()[~test], lam)
        if group.startswith("pilot/"):
            g = scored[scored["session"] == group.split("/", 1)[1]]
            rows.append({"lambda": lam, "test": group, "kind": "pilot_spearman",
                         "value": spearmanr(z.loc[g["path"], cols].to_numpy() @ beta, g["overall"]).statistic, "n": len(g)})
        else:
            p = 1 / (1 + np.exp(-(diffs[test] @ beta)))[0]
            rows.append({"lambda": lam, "test": group, "kind": "ab_correct", "value": float((p > 0.5) == bool(train["y"].to_numpy()[test][0])),
                         "p_a_better": p, "n": 1})
    return rows


def single_feature_table(pairs: pd.DataFrame, scored: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    decisive = pairs.dropna(subset=["a_better"])
    rows = []
    for c in FEATURES:
        da = feats.loc[decisive["a"], c].to_numpy() - feats.loc[decisive["b"], c].to_numpy()
        winner_higher = np.where(decisive["a_better"] == 1, da, -da)
        sessions = [spearmanr(feats.loc[g["path"], c], g["overall"]).statistic for _, g in scored.groupby("session")]
        rows.append({"feature": c, "ab_winner_higher": float((winner_higher > 0).mean()), "ab_ties": int((da == 0).sum()),
                     "pilot_spearman_mean": float(np.nanmean(sessions)),
                     "in_model": ",".join(v for v, cols in VARIANTS.items() if c in cols)})
    return pd.DataFrame(rows)


def tag_table(pairs: pd.DataFrame, scored: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Per tag and feature: AUC of the feature for clips with the tag against clips without (clips where tags were
    recorded: A/B clips and the pilot session that had tags)."""
    clips = [(r.a, r.tags_a) for r in pairs.itertuples()] + [(r.b, r.tags_b) for r in pairs.itertuples()]
    clips += [(r.path, r.tags) for r in scored.itertuples() if r.session == "candidates"]
    tags = pd.DataFrame([{"path": p, "tags": set(str(t).split(";")) if isinstance(t, str) else set()} for p, t in clips])
    rows = []
    for tag in sorted(set().union(*tags["tags"]) - {""}):
        has = tags["tags"].map(lambda s: tag in s).to_numpy()
        if has.sum() < 3 or (~has).sum() < 3:
            continue
        for c in FEATURES:
            v = feats.loc[tags["path"], c].to_numpy()
            x, z = v[has], v[~has]
            auc = np.mean([(a > b) + 0.5 * (a == b) for a in x for b in z])
            rows.append({"tag": tag, "feature": c, "auc": auc, "n_tag": int(has.sum()), "n_other": int((~has).sum()),
                         "mapped": TAG_FEATURES.get(tag) == c})
    return pd.DataFrame(rows)


def fit(args):
    pairs, scored = load_grades()
    paths = list(pairs["a"]) + list(pairs["b"]) + list(scored["path"])
    feats = feature_table(paths)
    train = training_pairs(pairs, scored)
    base = pairs.dropna(subset=["a_better"])
    more_notes = ((feats.loc[base["a"], "notes_per_bar"].to_numpy() > feats.loc[base["b"], "notes_per_bar"].to_numpy()) == (base["a_better"] == 1)).mean()
    real = feature_table([str(p.relative_to(HW2_ROOT)) for r in REFS for p in sorted((HW2_ROOT / r).glob("*.mid"))])
    cvs = []
    for variant, cols in VARIANTS.items():
        z, mean, std = standardize(feats, cols)
        cvs += [{"variant": variant, **r} for lam in LAMBDAS for r in cross_validate(train, z, scored, cols, lam)]
        d = z.loc[train["a"]].to_numpy() - z.loc[train["b"]].to_numpy()
        beta = fit_weights(d, train["y"].to_numpy(), train["w"].to_numpy(), args.lam)
        real_score = float((((real[cols] - mean) / std).fillna(0).to_numpy() @ beta).mean())
        model_path(variant).write_text(json.dumps({
            "features": cols, "mean": mean.to_dict(), "std": std.to_dict(), "weights": dict(zip(cols, beta)),
            "lambda": args.lam, "real_score": real_score, "n_ab_pairs": int(len(base)), "n_scored_clips": int(len(scored)),
            "sessions": sorted(set(pairs["session"]) | set(scored["session"]))}, indent=1))
        print(variant, "weights:", {k: round(v, 3) for k, v in zip(cols, beta)})
    cv = pd.DataFrame(cvs)
    out = RESULTS
    labels = pd.DataFrame([{"path": getattr(r, side), "session": r.session, "item": f"{r.item}_{side}", "pref": r.pref}
                           for r in pairs.itertuples() for side in "ab"] + scored[["path", "session", "item", "overall"]].to_dict("records"))
    labels.merge(feats, left_on="path", right_index=True).to_csv(out / "grade_features.csv", index=False)
    cv.to_csv(out / "grade_model_cv.csv", index=False)
    single_feature_table(pairs, scored, feats).to_csv(out / "grade_single_features.csv", index=False)
    tag_table(pairs, scored, feats).to_csv(out / "grade_tags.csv", index=False)
    summary = cv.groupby(["variant", "lambda", "kind"])["value"].agg(["mean", "count"]).round(3)
    print(summary.to_string())
    print(f"baseline 'more notes wins': {more_notes:.3f} of {len(base)} decisive A/B pairs; coin flip 0.5")
    print("wrote grade_model_<variant>.json, grade_features.csv, grade_model_cv.csv, grade_single_features.csv, grade_tags.csv")


def load_model(variant: str) -> tuple[dict, list[str], np.ndarray, pd.Series, pd.Series]:
    model = json.loads(model_path(variant).read_text())
    cols = model["features"]
    return model, cols, np.array([model["weights"][c] for c in cols]), pd.Series(model["mean"]), pd.Series(model["std"])


def score_features(f: dict, cols, beta, mean, std) -> float:
    return float((((pd.Series(f)[cols] - mean[cols]) / std[cols]).fillna(0).to_numpy()) @ beta)


def check(args):
    """Score a newly graded session with the models fit before it, then append the result to grade_model_check.csv."""
    pairs, _ = load_grades()
    g = pairs[pairs["session"] == args.session].dropna(subset=["a_better"])
    if g.empty:
        sys.exit(f"no decisive graded pairs for {args.session}")
    rows = []
    for variant in VARIANTS:
        model, cols, beta, mean, std = load_model(variant)
        if args.session in model["sessions"]:
            print(f"{variant}: already fit on {args.session}, skipped")
            continue
        correct = [(score_features(features_of(HW2_ROOT / r.a), cols, beta, mean, std)
                    > score_features(features_of(HW2_ROOT / r.b), cols, beta, mean, std)) == (r.a_better == 1) for r in g.itertuples()]
        rows.append({"session": args.session, "variant": variant, "pairs": len(correct), "correct": int(sum(correct)),
                     "accuracy": float(np.mean(correct)), "fit_on": ";".join(model["sessions"])})
    if rows:
        out = RESULTS / "grade_model_check.csv"
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        old = old[old["session"] != args.session] if len(old) else old
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
        print(pd.DataFrame(rows).drop(columns="fit_on").round(3).to_string(index=False))


def checkpoint_info(folder: Path, key: str) -> dict:
    info = folder / "gen_info.json"
    if not info.exists():
        return {}
    args = json.loads(info.read_text()).get("args", {})
    ckpt = HW2_ROOT / (args.get(key) or args.get("fill_ckpt") or args.get("ckpt") or "")
    state = ckpt / "train_state.json"
    if not state.exists():
        return {"checkpoint": str(ckpt)}
    st = json.loads(state.read_text())
    row = {"checkpoint": f"{ckpt.parent.name}/{ckpt.name}", "step": st["step"], "train_loss_ema": st["train_loss_ema"]}
    log = HW2_ROOT / "runs" / ckpt.parent.name / "log.csv"
    if log.exists():
        val = pd.read_csv(log).dropna(subset=["val_loss"])
        val = val[val["step"] <= st["step"]]
        if len(val):
            row["val_loss"] = float(val["val_loss"].iloc[-1])
    return row


def plot_vs_loss(sets: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = sets.dropna(subset=["step"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run, g in sets.groupby(sets["checkpoint"].str.split("/").str[0]):
        g = g.sort_values("step")
        axes[0].errorbar(g["step"], g["win_vs_real_mean"], yerr=g["win_vs_real_sd"], marker="o", capsize=3, label=run)
        axes[1].plot(g["train_loss_ema"], g["win_vs_real_mean"], "o-", label=f"{run} train loss")
        if "val_loss" in g:
            axes[1].plot(g["val_loss"], g["win_vs_real_mean"], "s--", label=f"{run} held-out loss")
    axes[0].axhline(0.5, color="gray", lw=0.8)
    axes[1].axhline(0.5, color="gray", lw=0.8)
    axes[0].set(xlabel="training step", ylabel="predicted win chance vs real")
    axes[1].set(xlabel="loss at the checkpoint", ylabel="predicted win chance vs real")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def predict(args):
    model, cols, beta, mean, std = load_model(args.variant)
    rows = []
    for folder in args.folders:
        folder = folder.resolve()
        for mid in sorted(folder.glob("*.mid")):
            f = features_of(mid)
            score = score_features(f, cols, beta, mean, std)
            rows.append({"source": str(folder.relative_to(HW2_ROOT)), "piece": mid.stem, "score": score,
                         "win_vs_real": 1 / (1 + np.exp(-(score - model["real_score"]))), **f})
    df = pd.DataFrame(rows)
    per_folder = pd.DataFrame([{"source": s, "variant": args.variant, "pieces": len(g), "win_vs_real_mean": g["win_vs_real"].mean(),
                                "win_vs_real_sd": g["win_vs_real"].std(), "win_vs_real_min": g["win_vs_real"].min(),
                                **checkpoint_info(HW2_ROOT / s, args.ckpt_key)} for s, g in df.groupby("source", sort=False)])
    df.to_csv(args.out, index=False)
    per_folder.to_csv(args.out.with_name(args.out.stem + "_sets.csv"), index=False)
    print(per_folder.drop(columns="variant").round(3).to_string(index=False))
    if args.plot:
        plot_vs_loss(per_folder, args.plot)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit on every graded session in results/listening, write results/grade_*.csv")
    f.add_argument("--lam", type=float, default=10.0, help="L2 penalty (10 had the best leave-one-out on 2026-10-17)")
    c = sub.add_parser("check", help="score a newly graded session with the models fit before it (run before fit)")
    c.add_argument("--session", required=True)
    g = sub.add_parser("predict", help="score every MIDI in the folders with results/grade_model_<variant>.json")
    g.add_argument("folders", nargs="+", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--variant", choices=list(VARIANTS), default="v3", help="v3 since T4 (6 of 7 before refit, v1 5 of 7)")
    g.add_argument("--plot", type=Path, help="figure of predicted grade vs checkpoint step and loss")
    g.add_argument("--ckpt_key", default="fill_ckpt", choices=["fill_ckpt", "plan_ckpt", "ckpt"],
                   help="gen_info.json checkpoint the step and loss columns come from")
    args = p.parse_args()
    {"fit": fit, "check": check, "predict": predict}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
