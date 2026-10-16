"""Blind listening sessions: prepare clips, build the phone rating page, import ratings, reveal sources."""

import argparse
import base64
import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pretty_midi

from hw2.remi import HW2_ROOT

SHEET = HW2_ROOT / "listening.csv"
RUBRIC = HW2_ROOT / "docs" / "listening_rubric.md"
ASPECTS_V1 = ["coherence", "correctness", "structureness", "richness", "overall"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_rubric(path=RUBRIC) -> dict:
    """docs/listening_rubric.md -> {"rules": [...], "criteria": [...], "tags": [...]} for the page and the sheet."""
    rules, criteria, tags, cur = [], [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            cur = None if line[3:].strip() == "Issue tags" else {"key": slug(line[3:]), "label": line[3:].strip(), "question": "",
                                                                  "scope": "", "listen": "", "not": "", "anchors": {}}
            if cur:
                criteria.append(cur)
            continue
        m = re.match(r"- \*\*(.+?):\*\* (.*)", line)
        if not criteria and not line.startswith(("#", "-")) and line.strip():
            rules.append(line.strip())
        elif m and cur is None and criteria:
            tags.append({"key": slug(m[1]), "label": m[1], "desc": m[2]})
        elif m and cur is not None:
            field = {"Scope": "scope", "Listen for": "listen", "Not here": "not"}.get(m[1])
            if field:
                cur[field] = m[2]
            elif m[1].isdigit():
                cur["anchors"][m[1]] = m[2]
        elif cur is not None and line.strip() and not line.startswith("-") and not cur["question"]:
            cur["question"] = line.strip()
    return {"rules": rules, "criteria": criteria, "tags": tags}


ASPECTS_V2 = [c["key"] for c in parse_rubric()["criteria"]]
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


def prepare(args):
    rng = random.Random(args.seed)
    used = set()
    for key_file in (HW2_ROOT / "outputs" / "listening").glob("*/key.json"):
        used |= {(v["source"], v["file"]) for v in json.loads(key_file.read_text()).values()}
    clips = []
    for src in args.sources:
        wavs = [w for w in sorted(Path(src).glob("*.wav")) if (str(Path(src)), w.name) not in used]
        if len(wavs) < args.k:
            raise SystemExit(f"{src}: only {len(wavs)} wav files not used in earlier sessions")
        clips += [(str(Path(src)), w) for w in rng.sample(wavs, args.k)]
    rng.shuffle(clips)

    out = HW2_ROOT / "outputs" / "listening" / args.session
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


def roll_data(entry):
    """Notes [start, dur, pitch, velocity, is_melody], bar start times and end time of a clip's MIDI, for the piano roll."""
    midi = pretty_midi.PrettyMIDI(str(HW2_ROOT / entry["source"] / Path(entry["file"]).with_suffix(".mid")))
    notes = [[round(n.start, 3), round(n.end - n.start, 3), n.pitch, n.velocity, int(inst.name.upper() == "MELODY")]
             for inst in midi.instruments for n in inst.notes]
    bars = [float(b) for b in midi.get_downbeats()]
    last_onset = max(n[0] for n in notes)
    while len(bars) > 1 and bars[-1] > last_onset:
        bars.pop()
    return {"notes": sorted(notes), "bars": [round(b, 3) for b in bars], "end": round(midi.get_end_time(), 3)}


def web(args):
    """The phone page cannot load files published next to it, so the audio is embedded in the page
    (base64 AAC). The bitrate is picked so the session fits the 16 MB page limit."""
    src = HW2_ROOT / "outputs" / "listening" / args.session
    root = HW2_ROOT / "outputs" / "listening" / "web"
    audio = root / "audio" / args.session
    audio.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.glob("clip_*.wav"))
    key = json.loads((src / "key.json").read_text())
    seconds = sum(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", w],
                                       capture_output=True, text=True, check=True).stdout) for w in wavs)
    kbps = min(96, int(args.budget_mb * 1e6 * 0.75 * 8 / seconds / 1000))
    clips, blobs = [], []
    for wav in wavs:
        m4a = audio / f"{wav.stem}_{kbps}k{'_mono' if args.mono else ''}.m4a"
        if not m4a.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav] + (["-ac", "1"] if args.mono else []) + ["-c:a", "aac", "-b:a", f"{kbps}k", m4a], check=True)
        clips.append({"id": wav.stem, "mime": "audio/mp4", **roll_data(key[wav.stem])})
        b64 = base64.b64encode(m4a.read_bytes()).decode()
        blobs.append(f'<script type="text/plain" id="audio-{args.session}-{wav.stem}">{b64}</script>')
    session = {"id": args.session, "title": args.title or args.session, "clips": clips, "rubric": parse_rubric()}
    html = (HW2_ROOT / "scripts" / "listening_page.html").read_text()
    html = re.sub(r'(<script type="application/json" id="manifest">).*?(</script>)',
                  lambda m: m.group(1) + json.dumps([session]) + m.group(2), html, count=1, flags=re.S)
    html = html.replace("<!--AUDIO-->", "\n".join(blobs), 1)
    out = root / "index.html"
    out.write_text(html)
    print(f"{len(clips)} clips, {seconds:.0f} s at {kbps} kbps -> {out} ({out.stat().st_size / 1e6:.1f} MB); publish it to the page's URL")


def import_ratings(args):
    docs = json.loads(Path(args.json).read_text())
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


def reveal(args):
    key = json.loads((HW2_ROOT / "outputs" / "listening" / args.session / "key.json").read_text())
    df = pd.read_csv(SHEET)
    df = df[df["session"] == args.session].copy()
    df["source"] = df["clip"].map(lambda c: key[c]["source"])
    df["file"] = df["clip"].map(lambda c: key[c]["file"])
    out = HW2_ROOT / "results" / f"listening_{args.session}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    aspects = [a for a in FIELDS[2:-2] if a in df and df[a].notna().any()]
    df = df.dropna(axis=1, how="all")
    df.to_csv(out, index=False)
    print(df.groupby("source")[aspects].mean().round(2).to_string())
    if "tags" in df:
        tags = df.assign(tag=df["tags"].fillna("").str.split(";")).explode("tag").query("tag != ''")
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
    b = sub.add_parser("reveal", help="join ratings with the hidden key; means per source -> results/listening_<session>.csv")
    b.add_argument("--session", required=True)
    c = sub.add_parser("web", help="build outputs/listening/web/index.html with the session's audio and the rubric embedded")
    c.add_argument("--session", required=True)
    c.add_argument("--title", default="")
    c.add_argument("--budget_mb", type=float, default=13.5, help="embedded audio size limit")
    c.add_argument("--mono", action="store_true", help="mono AAC: twice the bitrate per channel, so more clips fit")
    d = sub.add_parser("import", help="copy ratings exported from the page (JSON list of rating documents) into listening.csv")
    d.add_argument("json")
    args = p.parse_args()
    {"prepare": prepare, "reveal": reveal, "web": web, "import": import_ratings}[args.cmd](args)


if __name__ == "__main__":
    main()
