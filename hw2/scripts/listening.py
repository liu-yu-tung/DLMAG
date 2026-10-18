"""Blind listening sessions: prepare clips, build the phone rating page, import ratings, reveal sources."""

import argparse
import base64
import collections
import csv
import json
import time
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from scipy.io import wavfile
from scipy.stats import binomtest

from hw2.remi import HW2_ROOT
from hw2.render import (apply_velocity_map, first_sound, loudness_lufs, median_velocity, midi_to_wav, one_keyboard, real_velocity_counts,
                         velocity_maps)

RESULTS = HW2_ROOT / "results" / "listening"
SHEET = RESULTS / "ratings.csv"
AB_SHEET = RESULTS / "ratings_ab.csv"
LISTENING = HW2_ROOT / "outputs" / "listening"
REAL_VELOCITY = HW2_ROOT / "outputs" / "stats" / "real_velocity_counts.json"
RUBRIC = HW2_ROOT / "docs" / "listening_rubric.md"
ASPECTS_V1 = ["coherence", "correctness", "structureness", "richness", "overall"]
SITE_STACK = Path.home() / "Developments" / "Claude" / "self-hosting" / "stack"
SITE_PAGES = SITE_STACK / "pages-site"
SITE_RATINGS = SITE_STACK / "data" / "ratings"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


RUBRIC_LISTS = {"Issue tags": "tags", "Section flaws": "flaws"}


def parse_rubric(path=RUBRIC) -> dict:
    """docs/listening_rubric.md -> {"rules", "criteria", "tags", "flaws"} for the page and the sheet."""
    rules, criteria, lists, cur, items = [], [], {v: [] for v in RUBRIC_LISTS.values()}, None, None
    for line in path.read_text().splitlines():
        if line.startswith("## ") and line[3:].strip() in RUBRIC_LISTS:
            cur, items = None, lists[RUBRIC_LISTS[line[3:].strip()]]
            continue
        if line.startswith("## ") and items is not None:
            break
        if line.startswith("## "):
            cur = {"key": slug(line[3:]), "label": line[3:].strip(), "question": "", "scope": "", "listen": "", "not": "", "anchors": {}}
            criteria.append(cur)
            continue
        m = re.match(r"- \*\*(.+?):\*\* (.*)", line)
        if not criteria and not line.startswith(("#", "-")) and line.strip():
            rules.append(line.strip())
        elif m and items is not None:
            items.append({"key": slug(m[1]), "label": m[1], "desc": m[2]})
        elif m and cur is not None:
            field = {"Scope": "scope", "Listen for": "listen", "Not here": "not"}.get(m[1])
            if field:
                cur[field] = m[2]
            elif m[1].isdigit():
                cur["anchors"][m[1]] = m[2]
        elif cur is not None and line.strip() and not line.startswith("-") and not cur["question"]:
            cur["question"] = line.strip()
    return {"rules": rules, "criteria": criteria, **lists}


ASPECTS_V2 = [c["key"] for c in parse_rubric()["criteria"]]
AB_FIELDS = ["session", "pair", "pref", "reasons", "sections", "note_a", "note_b", "tags_a", "tags_b", "elem_a", "elem_b", "note"]
SECTIONS = ["1-8", "9-16", "17-24", "25-32"]
FIELDS = ["session", "clip"] + ASPECTS_V1 + [a for a in ASPECTS_V2 if a not in ASPECTS_V1] + ["tags", "note"]


def read_sheet() -> list[dict]:
    if not SHEET.exists():
        return []
    with open(SHEET, newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(rows: list[dict]):
    with open(SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def used_clips() -> set:
    used = set()
    for key_file in LISTENING.glob("*/key.json"):
        for v in json.loads(key_file.read_text()).values():
            used |= {(e["source"], e["file"]) for e in ([v["a"], v["b"]] if "a" in v else [v])}
    return used


def prepare(args):
    rng = random.Random(args.seed)
    used = used_clips()
    clips = []
    for src in args.sources:
        wavs = [w for w in sorted(Path(src).glob("*.wav")) if (str(Path(src)), w.name) not in used]
        if len(wavs) < args.k:
            raise SystemExit(f"{src}: only {len(wavs)} wav files not used in earlier sessions")
        clips += [(str(Path(src)), w) for w in rng.sample(wavs, args.k)]
    rng.shuffle(clips)

    out = LISTENING / args.session
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --session")
    out.mkdir(parents=True)
    key = {}
    for i, (src, wav) in enumerate(clips, 1):
        name = f"clip_{i:02d}"
        shutil.copy(wav, out / f"{name}.wav")
        key[name] = {"source": src, "file": wav.name}
    (out / "key.json").write_text(json.dumps(key, indent=1))

    write_sheet(read_sheet() + [{"session": args.session, "clip": name} for name in key])
    print(f"{len(key)} clips in {out}; build the page with: listening.py web --session {args.session}")


def pairs(args):
    rng = random.Random(args.seed)
    used = used_clips()
    pools = {}
    for spec in args.systems + [f"ref={args.ref}"]:
        name, dirs = spec.split("=", 1)
        pools[name] = [(str(Path(d)), w) for d in dirs.split(",") for w in sorted(Path(d).glob("*.wav"))
                       if (str(Path(d)), w.name) not in used]
        rng.shuffle(pools[name])
    names = [s.split("=", 1)[0] for s in args.systems]
    chosen = [tuple(m.split(":")) for m in args.matchups.split(",") if m] or [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
    same_file = {tuple(m.split(":")) for m in args.same_file.split(",") if m}
    matchups = [m for m in chosen for _ in range(args.k)]
    matchups += [(x, "ref") for x in names for _ in range(args.anchor)]
    for system, n in collections.Counter(s for m in matchups for s in m).items():
        if len(pools[system]) < n:
            raise SystemExit(f"{system}: needs {n} clips not used in earlier sessions, has {len(pools[system])}")
    rng.shuffle(matchups)

    out = LISTENING / args.session
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --session")
    out.mkdir(parents=True)
    key, flips = {}, collections.Counter()
    for i, (x, y) in enumerate(matchups, 1):
        # alternate which system plays first within each matchup, so order bias cancels
        sides = (x, y) if flips[(x, y)] % 2 == 0 else (y, x)
        flips[(x, y)] += 1
        name = f"pair_{i:02d}"
        key[name] = {}
        if (x, y) in same_file:
            names_y = {w.name for _, w in pools[y]}
            clip_x = next((c for c in pools[x] if c[1].name in names_y), None)
            if clip_x is None:
                raise SystemExit(f"{x}:{y}: no unused file name left in both")
            clip_y = next(c for c in pools[y] if c[1].name == clip_x[1].name)
            pools[x].remove(clip_x)
            pools[y].remove(clip_y)
            clips = {x: clip_x, y: clip_y}
        else:
            clips = {x: pools[x].pop(), y: pools[y].pop()}
        for side, system in zip("ab", sides):
            src, wav = clips[system]
            shutil.copy(wav, out / f"{name}_{side}.wav")
            key[name][side] = {"system": system, "source": src, "file": wav.name}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} pairs in {out}; build the page with: listening.py web --session {args.session} --page ab")


def prompt_pairs(args):
    rng = random.Random(args.seed)
    dirs = {name: Path(d) for name, d in (spec.split("=", 1) for spec in args.systems)}
    names = list(dirs)
    prompts = sorted(set.intersection(*[{w.name for w in d.glob("*.wav")} for d in dirs.values()]))
    matchups = [(x, y, w) for w in prompts for i, x in enumerate(names) for y in names[i + 1:]]
    rng.shuffle(matchups)
    out = LISTENING / args.session
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --session")
    out.mkdir(parents=True)
    key, flips = {}, collections.Counter()
    for i, (x, y, wav) in enumerate(matchups, 1):
        sides = (x, y) if flips[(x, y)] % 2 == 0 else (y, x)
        flips[(x, y)] += 1
        name = f"pair_{i:02d}"
        key[name] = {}
        for side, system in zip("ab", sides):
            shutil.copy(dirs[system] / wav, out / f"{name}_{side}.wav")
            key[name][side] = {"system": system, "source": str(dirs[system]), "file": wav}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} pairs in {out}; build the page with: listening.py web --session {args.session} --page ab --velocity generated")


def source_midi(entry) -> Path:
    return HW2_ROOT / entry["source"] / Path(entry["file"]).with_suffix(".mid")


def key_entry(key, stem):
    if stem in key:
        return key[stem]
    pair, side = stem.rsplit("_", 1)
    return key[pair][side]


def roll_data(midi: pretty_midi.PrettyMIDI, wav: Path):
    """Notes [start, dur, pitch, velocity, is_melody], bar start times and end time in the clip's audio time, for the piano roll."""
    first_onset = min(n.start for inst in midi.instruments for n in inst.notes)
    # midi_to_wav trims leading silence from the audio but not the MIDI; bars before 0 keep the bar numbers right
    shift = max(0.0, first_onset - first_sound(wav))
    notes = [[round(n.start - shift, 3), round(n.end - n.start, 3), n.pitch, n.velocity, int(inst.name.upper() == "MELODY")]
             for inst in midi.instruments for n in inst.notes]
    bars = [float(b) - shift for b in midi.get_downbeats()]
    last_onset = max(n[0] for n in notes)
    while len(bars) > 1 and bars[-1] > last_onset:
        bars.pop()
    rate, x = wavfile.read(wav)
    return {"notes": sorted(notes), "bars": [round(b, 3) for b in bars], "end": round(len(x) / rate, 3)}


SITE_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
             '<meta name="hw2-backend" content="site"><style>body{margin:0}</style></head><body>')


AB_INTRO = [
    "One pair at a time: Back and Next at the bottom (or the arrow keys); the dots show which pairs are picked.",
    "Top card: which piece is better, what decided it, and section by section (bars 1-8, 9-16, 17-24, 25-32) which is better and what was wrong with the worse one. Below it, A and B are tabs with a piano roll and a comment each: tap a letter to fold that column. The bar at the top switches the columns (A, A+B, B) and the rows (Roll, Comment).",
    "Clips are full length; tap a piano roll to jump. Every tap saves to this page.",
]
AB_RULES = [
    "Judge the music, not the sound quality: every clip uses the same piano sound, and the volume is matched.",
    "Clearly: you would pick it every time. A bit: you lean that way but it is close. Same: no preference.",
    "Pick a bit whenever you lean at all; same only when you cannot choose. Wins are counted; how much is kept for the report.",
]


def web(args):
    """claude.ai pages cannot load files published next to them, so that target embeds the audio as
    base64 AAC sized to the 16 MB page limit. The site target serves audio files next to the page."""
    site = args.target == "site"
    src = LISTENING / args.session
    root = LISTENING / "site" / args.page if site else LISTENING / "web"
    cache = LISTENING / "web" / "audio" / args.session
    cache.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.glob("*.wav"))
    key = json.loads((src / "key.json").read_text())
    seconds = sum(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", w],
                                       capture_output=True, text=True, check=True).stdout) for w in wavs)
    kbps = args.kbps if site else min(96, int(args.budget_mb * 1e6 * 0.75 * 8 / seconds / 1000))
    real = real_velocity_counts(REAL_VELOCITY) if args.velocity == "real" else None
    maps, clips, blobs = {}, [], []
    for wav in wavs:
        entry = key_entry(key, wav.stem)
        midi = pretty_midi.PrettyMIDI(str(source_midi(entry)))
        audio = wav
        if real is not None:
            if entry["source"] not in maps:
                maps[entry["source"]] = velocity_maps(
                    [pretty_midi.PrettyMIDI(str(m)) for m in sorted((HW2_ROOT / entry["source"]).glob("*.mid"))], real)
            apply_velocity_map(midi, maps[entry["source"]], min_gap=median_velocity(real["MELODY"]) - median_velocity(real["PIANO"]))
            one_keyboard(midi)
            audio = cache / f"{wav.stem}_realvel_keyboard.wav"
            if not audio.exists():
                midi.write(str(audio.with_suffix(".mid")))
                midi_to_wav(audio.with_suffix(".mid"), audio)
        m4a = cache / f"{audio.stem}{f'_{-args.lufs:g}lufs' if args.lufs else ''}_{kbps}k{'_mono' if args.mono else ''}.m4a"
        if not m4a.exists():
            gain = ["-af", f"volume={args.lufs - loudness_lufs(audio):.2f}dB,alimiter=limit=0.89:level=false:latency=true"] if args.lufs else []
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio] + gain + (["-ac", "1"] if args.mono else [])
                           + ["-c:a", "aac", "-b:a", f"{kbps}k", "-movflags", "+faststart", m4a], check=True)
        clip = {"id": wav.stem, "mime": "audio/mp4", **roll_data(midi, audio)}
        if site:
            served = root / "clips" / args.session / f"{wav.stem}.m4a"
            served.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(m4a, served)
            clip["src"] = served.relative_to(root).as_posix()
        else:
            b64 = base64.b64encode(m4a.read_bytes()).decode()
            blobs.append(f'<script type="text/plain" id="audio-{args.session}-{wav.stem}">{b64}</script>')
        clips.append(clip)
    session = {"id": args.session, "title": args.title or args.session, "clips": clips, "rubric": parse_rubric(),
               "built": time.strftime("%a %d %b %H:%M")}
    if any("a" in v for v in key.values()):
        by_id = {c["id"]: c for c in clips}
        session.update(kind="ab", intro=AB_INTRO, rules=AB_RULES,
                       clips=[{"id": p, "a": by_id[f"{p}_a"], "b": by_id[f"{p}_b"]} for p in sorted(key)])
    html = (HW2_ROOT / "scripts" / "listening_page.html").read_text()
    html = re.sub(r'(<script type="application/json" id="manifest">).*?(</script>)',
                  lambda m: m.group(1) + json.dumps([session]) + m.group(2), html, count=1, flags=re.S)
    html = html.replace("<!--AUDIO-->", "\n".join(blobs), 1)
    if site:
        html = SITE_HEAD + html + "</body></html>"
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(html)
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"{len(clips)} clips, {seconds:.0f} s at {kbps} kbps -> {root} ({size / 1e6:.1f} MB)")
    if site:
        print(f"copy to the site with: rsync -a --delete {root}/ {SITE_PAGES / args.page}/")


def import_ratings(args):
    path = Path(args.json).expanduser()
    docs = [json.loads(f.read_text()) for f in sorted(path.glob("*.json"))] if path.is_dir() else json.loads(path.read_text())
    pair_docs = [d for d in docs if str(d.get("clip", "")).startswith("pair_")]
    if pair_docs:
        old = pd.read_csv(AB_SHEET, dtype=str) if AB_SHEET.exists() else pd.DataFrame(columns=AB_FIELDS)
        new = pd.DataFrame([{"session": d["session"], "pair": d["clip"], "pref": d.get("pref"), "reasons": ";".join(d.get("reasons") or []),
                             "tags_a": ";".join(d.get("tags_a") or []), "tags_b": ";".join(d.get("tags_b") or []),
                             "elem_a": json.dumps(d.get("elem_a") or {}), "elem_b": json.dumps(d.get("elem_b") or {}),
                             "note": d.get("note") or "", "sections": json.dumps(d["sections"]) if d.get("sections") else "",
                             "note_a": d.get("note_a") or "", "note_b": d.get("note_b") or ""} for d in pair_docs])
        keep = ~old.set_index(["session", "pair"]).index.isin(new.set_index(["session", "pair"]).index)
        pd.concat([old[keep], new]).reindex(columns=AB_FIELDS).sort_values(["session", "pair"]).to_csv(AB_SHEET, index=False)
        print(f"{len(new)} rated pairs written to {AB_SHEET}")
    docs = [d for d in docs if d not in pair_docs]
    rated = {(d["session"], d["clip"]): d for d in docs if "session" in d and "clip" in d}
    rows = read_sheet()
    seen = {(r["session"], r["clip"]) for r in rows}
    rows += [{"session": s, "clip": c} for s, c in rated if (s, c) not in seen]
    n = 0
    for r in rows:
        d = rated.get((r["session"], r["clip"]))
        if d:
            r.update({k: d.get(k, "") if d.get(k) is not None else "" for k in FIELDS[2:-2] + ["note"]})
            r["tags"] = ";".join(d.get("tags") or [])
            n += 1
    write_sheet(rows)
    print(f"{n} rated clips written to {SHEET}")


def reveal_ab(session: str, key: dict):
    df = pd.read_csv(AB_SHEET)
    df = df[(df["session"] == session) & df["pref"].notna()].copy()
    for side in "ab":
        for field in ["system", "source", "file"]:
            df[f"{field}_{side}"] = df["pair"].map(lambda p: key[p][side][field])
    out = RESULTS / f"{session}.csv"
    df.to_csv(out, index=False)
    df["winner"] = np.where(df["pref"] < 0, df["system_a"], np.where(df["pref"] > 0, df["system_b"], "same"))
    rows = []
    for (x, y), g in df.groupby(df.apply(lambda r: tuple(sorted([r["system_a"], r["system_b"]])), axis=1)):
        # pref < 0 means A is better; turn it into x's advantage
        adv = np.where(g["system_a"] == x, -g["pref"], g["pref"])
        wins_x, wins_y = int((adv > 0).sum()), int((adv < 0).sum())
        p = binomtest(wins_x, wins_x + wins_y).pvalue if wins_x + wins_y else float("nan")
        rows.append({"x": x, "y": y, "pairs": len(g), "x_wins": wins_x, "y_wins": wins_y, "same": len(g) - wins_x - wins_y,
                     "x_mean_advantage": round(float(adv.mean()), 2), "sign_test_p": round(p, 3),
                     "x_first_wins": int(((g["system_a"] == x) & (adv > 0)).sum()), "x_first": int((g["system_a"] == x).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print("A wins %d, B wins %d, same %d (order bias check)" % ((df["pref"] < 0).sum(), (df["pref"] > 0).sum(), (df["pref"] == 0).sum()))
    if (df["file_a"] == df["file_b"]).all():
        wins = df[df["winner"] != "same"].assign(prompt=lambda d: d["file_a"].str.replace(".wav", "", regex=False))
        print("TASK B wins per prompt and config:")
        print(pd.crosstab(wins["prompt"], wins["winner"]).to_string())
    reasons = df.assign(reason=df["reasons"].fillna("").str.split(";")).explode("reason", ignore_index=True).query("reason != ''")
    if len(reasons):
        print("what decided it, by winner:")
        print(pd.crosstab(reasons["winner"], reasons["reason"]).to_string())
    tags = pd.concat([df.assign(system=df[f"system_{s}"], tag=df[f"tags_{s}"].fillna("").str.split(";")) for s in "ab"])
    tags = tags.explode("tag", ignore_index=True).query("tag != ''")
    if len(tags):
        print("issue tags per system:")
        print(pd.crosstab(tags["system"], tags["tag"]).to_string())
    elems = []
    for side in "ab":
        for _, r in df.iterrows():
            for element, e in json.loads(r.get(f"elem_{side}") or "{}").items() if isinstance(r.get(f"elem_{side}"), str) else []:
                elems.append({"system": r[f"system_{side}"], "element": element, "level": e["v"], "bars": ";".join(str(i * 8 + 1) + "-" + str(i * 8 + 8) for i in e.get("bars", [])) or "all"})
    if elems:
        e = pd.DataFrame(elems)
        print("element levels per system (-1 weak, 0 ok, 1 strong; mean and count):")
        print(e.groupby(["system", "element"])["level"].agg(["mean", "size"]).round(2).unstack("system").to_string())
        e.to_csv(RESULTS / f"{session}_elements.csv", index=False)
    secs = []
    for _, r in df.iterrows():
        raw = r.get("sections")
        for bars, e in zip(SECTIONS, json.loads(raw) if isinstance(raw, str) else []):
            if e:
                secs.append({"pair": r["pair"], "section": bars, "system_a": r["system_a"], "system_b": r["system_b"],
                             "better": {-1: r["system_a"], 1: r["system_b"]}.get(e["w"], "same"),
                             "worse": {-1: r["system_b"], 1: r["system_a"]}.get(e["w"], "both"), "flaws": ";".join(e.get("why") or [])})
    if secs:
        s = pd.DataFrame(secs)
        s.to_csv(RESULTS / f"{session}_sections.csv", index=False)
        per = pd.concat([s.assign(system=s[f"system_{x}"]) for x in "ab"], ignore_index=True)
        per["result"] = np.where(per["better"] == per["system"], "won", np.where(per["worse"] == per["system"], "lost", "same"))
        print("sections won / same / lost per system:")
        print(per.groupby(["system", "section"], sort=False)["result"].value_counts().unstack(fill_value=0)
              .reindex(columns=["won", "same", "lost"], fill_value=0).unstack("section", fill_value=0).to_string())
        flaws = per[per["result"] != "won"].assign(flaw=lambda d: d["flaws"].str.split(";")).explode("flaw", ignore_index=True).query("flaw != ''")
        if len(flaws):
            print("flaws of the worse piece (both when same), per system and section:")
            print(pd.crosstab([flaws["system"], flaws["section"]], flaws["flaw"]).to_string())
    print(f"-> {out}")


def reveal(args):
    key = json.loads((LISTENING / args.session / "key.json").read_text())
    if any("a" in v for v in key.values()):
        return reveal_ab(args.session, key)
    df = pd.read_csv(SHEET)
    df = df[df["session"] == args.session].copy()
    df["source"] = df["clip"].map(lambda c: key[c]["source"])
    df["file"] = df["clip"].map(lambda c: key[c]["file"])
    out = RESULTS / f"{args.session}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    aspects = [a for a in FIELDS[2:-2] if a in df and df[a].notna().any()]
    df = df.dropna(axis=1, how="all")
    df.to_csv(out, index=False)
    print(df.groupby("source")[aspects].mean().round(2).to_string())
    if "tags" in df:
        tags = df.assign(tag=df["tags"].fillna("").str.split(";")).explode("tag", ignore_index=True).query("tag != ''")
        if len(tags):
            print(pd.crosstab(tags["source"], tags["tag"]).to_string())
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="copy k random unused WAVs per source into a shuffled, blind session")
    a.add_argument("--session", required=True)
    a.add_argument("--k", type=int, default=3, help="clips per source")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("sources", nargs="+")
    e = sub.add_parser("pairs", help="blind A/B session: k pairs for each two systems, plus anchor pairs of each system vs real crops")
    e.add_argument("--session", required=True)
    e.add_argument("--k", type=int, default=8, help="pairs per two systems")
    e.add_argument("--anchor", type=int, default=2, help="pairs of each system against a real crop")
    e.add_argument("--ref", default="outputs/ref/heldout_crop32")
    e.add_argument("--matchups", default="", help="x:y,x:y system pairs to compare (default: every two systems); x plays first in the first pair of each matchup, then the sides alternate")
    e.add_argument("--same_file", default="", help="x:y,x:y matchups whose two clips share the file name (same plan or seed)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("systems", nargs="+", help="name=dir[,dir...]")
    f = sub.add_parser("prompt_pairs", help="blind A/B session for TASK B: each prompt's continuation from every two systems (same file name in each dir)")
    f.add_argument("--session", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("systems", nargs="+", help="name=dir")
    b = sub.add_parser("reveal", help="join ratings with the hidden key -> results/listening/<session>.csv; "
                                      "means per source, or win counts per system pair for an A/B session")
    b.add_argument("--session", required=True)
    c = sub.add_parser("web", help="build the rating page: claude.ai (outputs/listening/web, audio embedded) "
                                   "or the self-hosted site (outputs/listening/site, audio files next to the page)")
    c.add_argument("--session", required=True)
    c.add_argument("--title", default="")
    c.add_argument("--target", choices=["claude", "site"], default="site")
    c.add_argument("--page", default="audio", help="site target: folder under outputs/listening/site/ and pages-site/")
    c.add_argument("--velocity", choices=["real", "generated"], default="real",
                   help="real: quantile-map each source's velocities per track onto POP909's, play it as one keyboard "
                        "(no key sounding twice), and re-render; generated: the session WAVs as they are")
    c.add_argument("--lufs", type=float, default=-18.0, help="loudness target per clip (fixed gain plus peak limiter); 0 = off")
    c.add_argument("--budget_mb", type=float, default=13.5, help="claude target: embedded audio size limit")
    c.add_argument("--kbps", type=int, default=128, help="site target: AAC bitrate")
    c.add_argument("--mono", action="store_true", help="mono AAC: twice the bitrate per channel, so more clips fit")
    d = sub.add_parser("import", help="copy ratings into results/listening/ratings.csv (pairs: ratings_ab.csv): a JSON list exported from the claude.ai page, "
                                      f"or a folder of rating files such as the site's {SITE_RATINGS}")
    d.add_argument("json")
    args = p.parse_args()
    {"prepare": prepare, "pairs": pairs, "prompt_pairs": prompt_pairs, "reveal": reveal, "web": web, "import": import_ratings}[args.cmd](args)


if __name__ == "__main__":
    main()
