"""Blind A/B listening test for two runs: prepare clips and the phone page, import ratings, reveal the runs and check metric agreement.

Flow: prepare -> copy outputs/listening/<session>/site/ to the pages site -> rate on the phone -> import -> reveal.
The key (which run is A or B) stays in outputs/listening/<session>/key.json, never in the page.
"""
import argparse
import filecmp
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

HW3 = Path(__file__).resolve().parents[2]
DATA = HW3 / "data"
OUTPUTS = HW3 / "outputs"
RESULTS = HW3 / "results"
PAGE = HW3 / "scripts" / "listening_page.html"
SITE_STACK = Path.home() / "Developments" / "Claude" / "self-hosting" / "stack"
SITE_PAGES = SITE_STACK / "pages-site"
SITE_RATINGS = SITE_STACK / "data" / "ratings"
PREFS = {-2: "A clearly", -1: "A a bit", 0: "same", 1: "B a bit", 2: "B clearly"}
# per-clip metric columns: prefix -> True when higher is better
METRIC_DIRECTION = {"clap": True, "kl": False, "mqa": True}


def listening_dir(args) -> Path:
    return Path(args.listening_dir) / args.session


def prompt_ids(set_name: str, runs: list[str]) -> list[str]:
    stems = [{w.stem for w in (OUTPUTS / set_name / run).glob("*.wav")} for run in runs]
    return sorted(set.intersection(*stems), key=int)


def captions(set_name: str) -> dict[str, str]:
    table = pd.read_csv(DATA / f"{set_name}.csv", dtype=str)
    return dict(zip(table.iloc[:, 0], table["caption"]))


def loudness_lufs(wav: Path) -> float:
    stderr = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(wav), "-af", "loudnorm=I=-18:print_format=json", "-f", "null", "-"],
                            capture_output=True, text=True, check=True).stderr
    return float(re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"', stderr).group(1))


def encode(wav: Path, out_stem: Path, lufs: float, codec: str) -> Path:
    gain = [] if not lufs else ["-af", f"volume={lufs - loudness_lufs(wav):.2f}dB,alimiter=limit=0.89:level=false"]
    if codec == "wav":
        out = out_stem.with_suffix(".wav")
        if gain:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav)] + gain + [str(out)], check=True)
        else:
            shutil.copy(wav, out)
        return out
    out = out_stem.with_suffix(".mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav)] + gain + ["-c:a", "libmp3lame", "-b:a", "128k", str(out)], check=True)
    return out


def pick_codec() -> str:
    if not shutil.which("ffmpeg"):
        return "wav"
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "mp3" if "libmp3lame" in encoders else "wav"


def prepare(args):
    runs = [args.a, args.b]
    assert args.a != args.b, "--a and --b must differ"
    for run in runs:
        assert (OUTPUTS / args.set / run).is_dir(), f"no audio in outputs/{args.set}/{run}"
    ids = prompt_ids(args.set, runs)
    # Best-of-N keeps seed 0 for some prompts; identical pairs carry no preference.
    ids = [i for i in ids if not filecmp.cmp(OUTPUTS / args.set / args.a / f"{i}.wav", OUTPUTS / args.set / args.b / f"{i}.wav", shallow=False)]
    assert len(ids) >= args.n, f"only {len(ids)} prompts have audio in both runs"
    rng = random.Random(args.seed)
    chosen = rng.sample(ids, args.n)
    a_first = [True] * (args.n // 2) + [False] * (args.n - args.n // 2)
    rng.shuffle(a_first)

    root = listening_dir(args)
    assert not root.exists(), f"{root} exists; pick another --session"
    clips = root / "site" / "clips"
    clips.mkdir(parents=True)
    codec = pick_codec()
    text = captions(args.set)
    key, pairs = {"set": args.set, "run_first": args.a, "run_second": args.b, "seed": args.seed, "codec": codec, "pairs": {}}, []
    for number, (prompt, first_is_a) in enumerate(zip(chosen, a_first), 1):
        pair = f"pair_{number:02d}"
        side_runs = {"a": args.a, "b": args.b} if first_is_a else {"a": args.b, "b": args.a}
        files = {side: encode(OUTPUTS / args.set / run / f"{prompt}.wav", clips / f"{pair}_{side}", args.lufs, codec).name
                 for side, run in side_runs.items()}
        key["pairs"][pair] = {"prompt_id": prompt, "a": side_runs["a"], "b": side_runs["b"]}
        pairs.append({"id": pair, "caption": text[prompt], "a": f"clips/{files['a']}", "b": f"clips/{files['b']}"})
    (root / "key.json").write_text(json.dumps(key, indent=1))
    manifest = {"session": args.session, "pairs": pairs}
    html = re.sub(r'(<script type="application/json" id="manifest">).*?(</script>)', lambda m: m.group(1) + json.dumps(manifest) + m.group(2),
                  PAGE.read_text(), count=1, flags=re.S)
    (root / "site" / "index.html").write_text(html)
    size = sum(f.stat().st_size for f in (root / "site").rglob("*") if f.is_file())
    print(f"{args.n} pairs ({codec}, {size / 1e6:.1f} MB) in {root / 'site'}; A is run --a in {sum(a_first)} pairs")
    print(f"copy to the site with: rsync -a --delete {root / 'site'}/ {SITE_PAGES / ('hw3-' + args.session)}/")


def read_ratings(source: Path, session: str) -> pd.DataFrame:
    if source.is_dir():
        docs = [json.loads(f.read_text()) for f in sorted(source.glob(f"{session}__*.json"))]
        table = pd.DataFrame([{"pair": d.get("pair") or d.get("clip"), "pref": d.get("pref"), "note": d.get("note", "")} for d in docs])
    else:
        table = pd.read_csv(source)
        if "session" in table:
            table = table[table["session"] == session]
    assert {"pair", "pref"} <= set(table.columns), "ratings need columns pair, pref"
    return table.dropna(subset=["pref"]).assign(pref=lambda t: t["pref"].astype(int)).drop_duplicates("pair", keep="last")


def import_ratings(args):
    root = listening_dir(args)
    key = json.loads((root / "key.json").read_text())
    table = read_ratings(Path(args.ratings).expanduser(), args.session)
    assert table["pref"].isin(PREFS).all(), "pref must be in -2..2 (negative = A)"
    unknown = set(table["pair"]) - set(key["pairs"])
    assert not unknown, f"pairs not in this session: {sorted(unknown)}"
    table.sort_values("pair").to_csv(root / "ratings.csv", index=False)
    print(f"{len(table)} of {len(key['pairs'])} pairs rated -> {root / 'ratings.csv'}")


def metric_columns(table: pd.DataFrame) -> dict[str, bool]:
    return {c: METRIC_DIRECTION[next(p for p in METRIC_DIRECTION if c.startswith(p))] for c in table.columns
            if c != "id" and c.startswith(tuple(METRIC_DIRECTION))}


def per_clip_metrics(set_name: str, run: str) -> pd.DataFrame:
    folder = RESULTS / f"{set_name}__{run}"
    tables = []
    if (folder / "per_prompt.csv").exists():
        tables.append(pd.read_csv(folder / "per_prompt.csv"))
    if (folder / "mqa_per_item.csv").exists():
        tables.append(pd.read_csv(folder / "mqa_per_item.csv", usecols=["id", "mqascore"]))
    if not tables:
        return pd.DataFrame(columns=["id"])
    merged = tables[0]
    for extra in tables[1:]:
        merged = merged.merge(extra, on="id", how="outer")
    return merged.assign(id=lambda t: t["id"].astype(str)).set_index("id")


def agreement(rated: pd.DataFrame, set_name: str, run_first: str, run_second: str) -> pd.DataFrame:
    first, second = per_clip_metrics(set_name, run_first), per_clip_metrics(set_name, run_second)
    shared = {**metric_columns(first), **metric_columns(second)}
    shared = {c: higher for c, higher in shared.items() if c in first and c in second}
    decided = rated[rated["strength"] > 0]
    rows = []
    for column, higher in shared.items():
        delta = first[column].reindex(decided["prompt_id"]).to_numpy() - second[column].reindex(decided["prompt_id"]).to_numpy()
        metric_first = (delta > 0) if higher else (delta < 0)
        usable = ~pd.isna(delta) & (delta != 0)
        ear_first = (decided["chosen_run"] == run_first).to_numpy()
        agree = (metric_first == ear_first) & usable
        clearly = usable & (decided["strength"] == 2).to_numpy()
        n = int(usable.sum())
        rows.append({"metric": column, "better": "higher" if higher else "lower", "pairs": n, "agree": int(agree.sum()),
                     "agreement_rate": round(agree.sum() / n, 3) if n else float("nan"),
                     "sign_test_p": round(binomtest(int(agree.sum()), n).pvalue, 3) if n else float("nan"),
                     "clearly_pairs": int(clearly.sum()), "clearly_agree": int((agree & clearly).sum())})
    return pd.DataFrame(rows)


def reveal(args):
    root = listening_dir(args)
    key = json.loads((root / "key.json").read_text())
    ratings = pd.read_csv(root / "ratings.csv", dtype={"pair": str})
    rows = []
    for _, r in ratings.iterrows():
        entry = key["pairs"][r["pair"]]
        pref = int(r["pref"])
        chosen_side = "a" if pref < 0 else "b" if pref > 0 else "same"
        chosen_run = entry[chosen_side] if pref else "same"
        rows.append({"pair": r["pair"], "prompt_id": entry["prompt_id"], "run_a": entry["a"], "run_b": entry["b"], "rating": pref,
                     "chosen_side": chosen_side.upper() if pref else "same", "chosen_run": chosen_run, "strength": abs(pref)})
    table = pd.DataFrame(rows)
    out = Path(args.results_dir) / f"{args.session}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    first, second = key["run_first"], key["run_second"]
    wins = {run: int((table["chosen_run"] == run).sum()) for run in (first, second)}
    ties = int((table["chosen_run"] == "same").sum())
    decided = sum(wins.values())
    p = binomtest(wins[first], decided).pvalue if decided else float("nan")
    print(f"{key['set']}: {first} wins {wins[first]}, {second} wins {wins[second]}, same {ties} (sign test p={p:.3f}, ties excluded)")
    print(f"A chosen {int((table['chosen_side'] == 'A').sum())}, B chosen {int((table['chosen_side'] == 'B').sum())} (side bias check)")
    check = agreement(table, key["set"], first, second)
    check.to_csv(Path(args.results_dir) / f"{args.session}_agreement.csv", index=False)
    print(check.to_string(index=False))
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listening_dir", default=str(OUTPUTS / "listening"), help="sessions live here, git-ignored")
    p.add_argument("--results_dir", default=str(RESULTS / "listening"))
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="pick prompts, randomize sides, encode clips, build the page")
    a.add_argument("--set", choices=["dev", "test"], required=True)
    a.add_argument("--a", required=True, help="first run, e.g. cfg7 (audio in outputs/<set>/<run>/)")
    a.add_argument("--b", required=True, help="second run")
    a.add_argument("--n", type=int, default=16)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--session", required=True)
    a.add_argument("--lufs", type=float, default=-18.0, help="loudness target per clip (gain plus peak limiter); 0 = off")
    a.set_defaults(func=prepare)
    b = sub.add_parser("import", help="read ratings from the page's CSV export or the site's ratings folder")
    b.add_argument("--session", required=True)
    b.add_argument("--ratings", required=True, help=f"CSV with columns pair, pref (-2..2, negative = A), or a folder such as {SITE_RATINGS}")
    b.set_defaults(func=import_ratings)
    c = sub.add_parser("reveal", help="map ratings to runs, write results/listening/<session>.csv, check metric agreement")
    c.add_argument("--session", required=True)
    c.set_defaults(func=reveal)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
