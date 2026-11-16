import argparse
import csv
import difflib
import html
import json
import re
import shutil
import subprocess
import tomllib
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HW3 = Path(__file__).resolve().parents[2]
ANALYSIS = HW3 / "results" / "analysis"
HOP = 512
N_FFT = 2048
N_MELS = 128
# One range for every image so colours compare across clips; generated and real clips both sit between these.
MEL_DB_MIN, MEL_DB_MAX = -90.0, -10.0

DEV_RUNS = ["cfg3", "cfg7", "cfg12", "apg12", "rescale12", "rescale12_seed1", "rescale12_best3_c8998", "rescale12_phi1",
            "interval_late", "tags", "rescale12_followed", "rescale16", "rescale12_clean", "rescale12_ablate", "rescale12_degraded"]
TEST_RUNS = ["cfg7", "rescale12", "rescale12_best3_c8998"]
BASE_RUN = "rescale12"

RUN_TEXT = {
    "cfg3": "CFG scale 3.",
    "cfg7": "CFG scale 7, the baseline.",
    "cfg12": "CFG scale 12.",
    "apg12": "Adaptive projected guidance (APG) at scale 12 in place of plain CFG.",
    "rescale12": "CFG 12 with rescale phi 0.7; the single-run pick.",
    "rescale12_seed1": "rescale12 with seed index 1 (a different noise draw, same caption).",
    "rescale12_seed2": "rescale12 with seed index 2.",
    "rescale12_best3_c8998": "Submission: per caption, the best of the three rescale12 seeds (0, 1, 2), ranked by a second CLAP checkpoint (89.98).",
    "rescale12_phi1": "rescale12 with phi 1.0 (full rescale).",
    "interval_late": "CFG 7 switched off for the late steps (noise level sigma <= 2.6).",
    "tags": "A tag list built from the caption's MusicCaps tags as the prompt, CFG 7.",
    "rescale12_followed": "rescale12 plus an extra guidance term (weight 4) on the caption's own instrument, voice, timbre and genre phrases.",
    "rescale16": "CFG 16 with rescale phi 0.7.",
    "rescale12_clean": "rescale12 with the recording and setting sentences removed from the caption; only captions that had such sentences.",
    "rescale12_ablate": "rescale12 with one sentence that carries a followed word (instrument or voice) removed; a subset of captions.",
    "rescale12_degraded": "rescale12 clips of captions that mention recording quality, low-passed at 4 kHz and noised at 20 dB SNR; other clips are copies of rescale12.",
}
EDIT_GLOSSARY = [
    ("edit_<axis>", "Word-swap test: the rescale12 settings and seed, with one word group of the caption swapped, so a pair differs only in the words. "
                    "The original clip is the rescale12 clip of the same id."),
    ("edit_<axis>_ext", "The same swap on every other eligible caption (the extended set)."),
    ("edit_null", "Unchanged captions sent through the edit path; a gate that must reproduce rescale12."),
    ("edit_control_*", "Meaningless edits (a synonym swap, or one neutral sentence appended) as controls for how much any text change moves a clip."),
    ("bundle_<axis>_bundle", "The edited caption at weight 12 plus an extra guidance term on the text of the drum and groove bundle (away from it for loud, toward it for sad)."),
    ("bundle_<axis>_neutral", "Control arm: the same extra term weight and sign, on a meaning-free sentence."),
]
FIGURE_TEXT = {
    "compute_dev.png": "Dev set: quality against compute; dCLAP bars are paired bootstrap intervals against cfg7, and best-of-3 is picked by CLAP.",
    "genre_confusion.png": "Genre map learned from audio: 5-fold genre probe confusion on MERT and CLAP embeddings, genres ordered by clustering.",
    "ltas_dev.png": "Long-term average spectrum of dev clips per run, and the difference from the real dev clips.",
    "roadmap_dev.png": "How each dev decision moved the score: paired change against cfg7 with 95% bootstrap intervals.",
    "probe_auc.png": "Cross-validated probe AUC per caption word group, next to how many dev captions name the group.",
    "moved_axes.png": "Word-swap test: in how many pairs the target probe moved as predicted, per axis, for the MERT and CLAP probes (dashed line: chance).",
    "spillover.png": "Spillover of each word swap onto other attribute groups, in seed SD; dots mark intervals that exclude zero.",
    "bundle_cut.png": "Bundle cut: change against the plain edited clip for the bundle arm and the neutral arm (loud: drums, groove, loud probe; sad: drums, sad score).",
    "metric_grid.png": "Metric by intervention: followed word removed, recording sentences removed, recording degraded, read by CLAP, MQAScore and KL-PaSST.",
    "word_forest.png": "Followed-word guidance and probe-in-the-loop rows against rescale 12 and scale 16: dCLAP, dKL-PaSST, dMQAScore.",
}
AXIS_TEXT = {
    "voice_male_to_female": "vocals: male voice to female voice",
    "guitar_to_piano": "instrument: guitar to piano, keys",
    "acoustic_to_electric": "timbre: acoustic to electric, distorted",
    "soft_to_loud": "dynamics: soft, calm to loud, intense",
    "happy_to_sad": "mood: happy, bright to sad, emotional",
    "fast_to_slow": "tempo: fast to slow",
    "low_quality_to_studio": "recording: low quality to high quality",
    "insert_jazz": "one sentence appended: the genre is jazz",
    "insert_country": "one sentence appended: the genre is country",
    "insert_1960s": "one sentence appended: from the 1960s",
    "insert_2010s": "one sentence appended: from the 2010s",
    "insert_brazil": "one sentence appended: from Brazil",
    "insert_germany": "one sentence appended: from Germany",
    "insert_latin": "one sentence appended: the genre is Latin",
    "insert_techno": "one sentence appended: the genre is techno",
    "control_song_to_track": "control: the word song replaced by track",
    "control_neutral_insert": "control: the sentence 'The clip is ten seconds long.' appended",
    "null": "gate: caption unchanged",
}
KIND_LABEL = {
    "asks_for_vocals": "asks for vocals",
    "says_no_vocals": "says no vocals",
    "mentions_recording_quality": "low-quality recording words",
    "mentions_non_music_sounds": "non-music sounds",
    "mostly_mood_words": "mostly mood words",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the non-blind HW3 samples page: dev and test clips with mel spectrograms and per-clip scores, the word-swap and "
        "bundle-cut pairs, and every figure under results/analysis/figures and report/deck/figs_out (globbed at build time, so new "
        "figures appear on a rebuild). Needs numpy and matplotlib; run with: uv run --no-project --with numpy,matplotlib python -I "
        "scripts/listening/samples_page.py. Writes index.html, audio/, mel/ and fig/ under --out; test captions are never written.")
    parser.add_argument("--out", type=Path, default=HW3 / "outputs/listening/site/hw3-samples")
    parser.add_argument("--bitrate", default="128k")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dev-ids", type=int, nargs="+", help="override the 8 dev ids (default: picked by caption kind, see the page)")
    parser.add_argument("--test-ids", type=int, nargs="+", help="override the 6 test ids (default: evenly spaced ids)")
    parser.add_argument("--edit-ids", type=int, default=3, help="pairs shown per edit run and per bundle axis")
    parser.add_argument("--force", action="store_true", help="re-encode audio and re-render mels that already exist")
    return parser.parse_args()


def read_csv(path: Path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Scores:
    def __init__(self):
        self.cache = {}

    def table(self, split, run):
        key = (split, run)
        if key not in self.cache:
            folder = HW3 / "results" / f"{split}__{run}"
            table = {}
            if (folder / "per_prompt.csv").is_file():
                for row in read_csv(folder / "per_prompt.csv"):
                    table[row["id"]] = {"clap": to_float(row["clap_cosine"]), "kl": to_float(row.get("kl_passt_softmax"))}
            if (folder / "mqa_per_item.csv").is_file():
                for row in read_csv(folder / "mqa_per_item.csv"):
                    table.setdefault(row["id"], {})["mqa"] = to_float(row["mqascore"])
            self.cache[key] = table
        return self.cache[key]

    def get(self, split, run, clip_id):
        return self.table(split, run).get(str(clip_id), {})


class Assets:
    def __init__(self, out: Path):
        self.out = out
        self.jobs = {}

    def clip(self, folder: str, wav: Path):
        if not wav.is_file():
            return None
        audio = Path("audio") / folder / f"{wav.stem}.mp3"
        mel = Path("mel") / folder / f"{wav.stem}.png"
        self.jobs[audio.as_posix()] = (wav, audio, mel)
        return {"audio": audio.as_posix(), "mel": mel.as_posix()}

    def build(self, bitrate, workers, force):
        todo = list(self.jobs.values())
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda job: self.make(*job, bitrate, force), todo))
        return len(todo)

    def make(self, wav, audio, mel, bitrate, force):
        audio_path, mel_path = self.out / audio, self.out / mel
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not audio_path.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate, str(audio_path)],
                           check=True)
        if force or not mel_path.exists():
            plt.imsave(mel_path, log_mel(wav).T[::-1], vmin=MEL_DB_MIN, vmax=MEL_DB_MAX, cmap="magma")


def mel_filterbank(rate):
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges = to_hz(np.linspace(0, to_mel(rate / 2), N_MELS + 2))
    freqs = np.fft.rfftfreq(N_FFT, 1 / rate)
    bank = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        low, mid, high = edges[i:i + 3]
        bank[i] = np.maximum(0, np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)))
    return bank


def log_mel(wav: Path):
    with wave.open(str(wav)) as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        assert handle.getsampwidth() == 2, wav
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, channels).mean(axis=1) / 32768
    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT)
    frames = padded[np.arange(0, len(padded) - N_FFT + 1, HOP)[:, None] + np.arange(N_FFT)]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 / window.sum() ** 2
    return 10 * np.log10(power @ mel_filterbank(rate).T + 1e-10)


def prompts_of(run_dir: Path):
    path = run_dir / "prompts.jsonl"
    table = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            table[str(row["id"])] = row
    return table


def ids_with_wav(run_dir: Path):
    return {p.stem for p in run_dir.glob("*.wav")}


def pick_dev_ids(count_per_kind=(("asks_for_vocals", 2), ("says_no_vocals", 2), ("mentions_recording_quality", 2),
                                  ("mostly_mood_words", 1), ("mentions_non_music_sounds", 1))):
    rows = {r["id"]: r for r in read_csv(ANALYSIS / "caption_classes_dev.csv")}
    present = {run: ids_with_wav(HW3 / "outputs/dev" / run) for run in DEV_RUNS}
    median_words = float(np.median([int(r["words"]) for r in rows.values()]))
    flag = lambda r, k: r[k] == "True"
    picked, reasons = [], {}
    for kind, count in count_per_kind:
        if kind == "mentions_recording_quality":
            ok = lambda r: flag(r, kind) and not flag(r, "asks_for_vocals")
        elif kind in ("asks_for_vocals", "says_no_vocals"):
            ok = lambda r: flag(r, kind) and not flag(r, "mentions_recording_quality")
        else:
            ok = lambda r: flag(r, kind) and not flag(r, "asks_for_vocals") and not flag(r, "mentions_recording_quality")
        pool = [i for i, r in rows.items() if ok(r) and i not in picked and (HW3 / f"outputs/ref/dev_real/{i}.wav").is_file()]
        pool.sort(key=lambda i: (-sum(i in present[run] for run in DEV_RUNS), abs(int(rows[i]["words"]) - median_words), int(i)))
        for i in pool[:count]:
            picked.append(i)
            reasons[i] = KIND_LABEL[kind]
    return picked, reasons, median_words


def pick_spread(ids, count):
    ids = sorted(ids, key=int)
    if len(ids) <= count:
        return ids
    return [ids[round(x)] for x in np.linspace(0, len(ids) - 1, count)]


def fmt(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def esc(text):
    return html.escape(str(text), quote=True)


def marked(text, other, side):
    a, b = text.split(), other.split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        chunk = esc(" ".join(a[i1:i2]))
        if not chunk:
            continue
        parts.append(chunk if tag == "equal" else f'<mark class="{side}">{chunk}</mark>')
    return " ".join(parts)


def score_line(scores, with_kl=True):
    bits = []
    for key, name, digits in (("clap", "CLAP", 3), ("kl", "KL", 2), ("mqa", "MQAScore", 3)):
        if key == "kl" and not with_kl:
            continue
        value = scores.get(key)
        bits.append(f"<span><b>{name}</b> {fmt(value, digits) or '&ndash;'}</span>")
    return f'<div class="scores">{" ".join(bits)}</div>'


def card(label, clip, scores=None, notes=(), with_kl=True):
    body = [f'<div class="label">{esc(label)}</div>']
    body.append(f'<div class="mel live" data-sec="10"><img loading="lazy" alt="mel spectrogram" src="{esc(clip["mel"])}"><div class="head"></div></div>')
    body.append(f'<audio controls preload="none" src="{esc(clip["audio"])}"></audio>')
    if scores is not None:
        body.append(score_line(scores, with_kl))
    body += [f'<div class="note">{note}</div>' for note in notes]
    return f'<figure class="clip">{"".join(body)}</figure>'


def run_settings(run):
    path = HW3 / "results" / f"dev__{run}" / "gen_info.json"
    if not path.is_file():
        return ""
    settings = json.loads(path.read_text())["settings"]
    guidance = settings["guidance"]
    bits = [f"{guidance['mode'].upper()} {guidance['scale']:g}"]
    if "rescale_phi" in guidance:
        bits.append(f"rescale phi {guidance['rescale_phi']:g}")
    if "interval_sigma_low" in guidance:
        bits.append(f"guidance off for sigma <= {guidance['interval_sigma_low']:g}")
    if guidance.get("extra_scale"):
        bits.append(f"extra term weight {guidance['extra_scale']:g}")
    bits.append(f"seed index {settings['seed_index']}")
    bits.append(f"{settings['steps']} steps")
    return ", ".join(bits)


def findings_table_html():
    lines = [l for l in (ANALYSIS / "findings_table.md").read_text().splitlines() if l.startswith("|")]
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    header, rows = cells(lines[0]), [cells(l) for l in lines[2:]]
    out = ['<div class="scroll"><table class="metrics"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    group = None
    for row in rows:
        split = row[0].split("__")[0]
        if split != group:
            out.append(f'<tr class="group"><td colspan="{len(header)}">{esc(split)} set</td></tr>')
            group = split
        cls = ' class="submission"' if row[0].endswith("rescale12_best3_c8998") else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out), len(rows)


def figures_html(out: Path):
    sources = [("analysis", ANALYSIS / "figures"), ("deck", HW3 / "report/deck/figs_out")]
    items, count = [], 0
    for group, folder in sources:
        for png in sorted(folder.glob("*.png")):
            target = Path("fig") / group / png.name
            (out / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, out / target)
            where = "results/analysis/figures" if group == "analysis" else "report/deck/figs_out"
            note = FIGURE_TEXT.get(png.name, "")
            items.append(f'<figure class="fig"><a href="{esc(target.as_posix())}"><img loading="lazy" alt="{esc(png.name)}" src="{esc(target.as_posix())}"></a>'
                         f'<figcaption><code>{esc(png.name)}</code> <span class="where">{where}</span>'
                         + (f"<br>{esc(note)}" if note else "") + "</figcaption></figure>")
            count += 1
    return '<div class="figs">' + "".join(items) + "</div>", count


def dev_section(assets, scores, dev_ids, reasons):
    classes = {r["id"]: r for r in read_csv(ANALYSIS / "caption_classes_dev.csv")}
    base_prompts = prompts_of(HW3 / "outputs/dev" / BASE_RUN)
    degraded = set(json.loads((HW3 / "results/dev__rescale12_degraded/degrade.json").read_text())["degraded_ids"])
    selection = {r["id"]: r["chosen_run"] for r in read_csv(HW3 / "results/dev__rescale12_best3_c8998/selection.csv")}
    followed_extra = {r["id"]: r["text"] for r in read_csv(HW3 / "data/followed/dev_extra.csv")}
    run_prompts = {run: prompts_of(HW3 / "outputs/dev" / run) for run in DEV_RUNS}
    blocks, skipped = [], {}
    for clip_id in dev_ids:
        row = classes[clip_id]
        tags = [label for key, label in KIND_LABEL.items() if row[key] == "True"]
        caption = base_prompts[clip_id]["prompt"]
        cards = []
        real = assets.clip("dev_real", HW3 / f"outputs/ref/dev_real/{clip_id}.wav")
        if real:
            cards.append(card("real clip (MusicCaps)", real, {**scores.get("dev", "real", clip_id)}, with_kl=False))
        for run in DEV_RUNS:
            clip = assets.clip(f"dev/{run}", HW3 / f"outputs/dev/{run}/{clip_id}.wav")
            if not clip:
                skipped.setdefault(clip_id, []).append(run)
                continue
            notes = []
            prompt = run_prompts[run].get(clip_id, {})
            if prompt.get("prompt") and prompt["prompt"] != caption:
                notes.append("prompt used: " + (esc(prompt["prompt"]) if run == "tags" else marked(prompt["prompt"], caption, "ins")))
            if run == "rescale12_followed" and clip_id in followed_extra:
                notes.append("extra guidance text: " + esc(followed_extra[clip_id]))
            if run == "rescale12_best3_c8998":
                notes.append(f"picked seed: {esc(selection[clip_id])}")
            if run == "rescale12_degraded" and clip_id not in degraded:
                notes.append("not degraded for this id (a copy of rescale12)")
            cards.append(card(run, clip, scores.get("dev", run, clip_id), notes))
        kind = f'<span class="kind">{esc(reasons.get(clip_id, "chosen by id"))}</span>'
        tag_text = ", ".join(tags) or "none of the five flags"
        blocks.append(f'<section class="block" id="dev-{clip_id}"><h3>dev id {clip_id} {kind}</h3>'
                      f'<p class="meta">caption flags: {esc(tag_text)}; {row["words"]} words</p>'
                      f'<blockquote>{esc(caption)}</blockquote><div class="grid">{"".join(cards)}</div></section>')
    return "".join(blocks), skipped


def edit_pairs_index():
    index = {}
    for row in read_csv(ANALYSIS / "edit_pairs.csv"):
        if row["detector"] in ("probe_mert", "probe_clap90"):
            axis = row["axis"]
            index.setdefault((axis, row["id"]), {})[row["detector"]] = float(row["delta"])
    return index


def edit_section(assets, n_ids):
    manifest = json.loads((ANALYSIS / "edits_manifest.json").read_text())
    pairs = edit_pairs_index()
    base_dir = HW3 / "outputs/dev" / BASE_RUN
    base_prompts = prompts_of(base_dir)
    blocks = []
    for run_dir in sorted((HW3 / "outputs/dev").glob("edit_*")):
        name = run_dir.name.removeprefix("edit_")
        axis = name.removesuffix("_ext")
        prompts = prompts_of(run_dir)
        ids = pick_spread([i for i in ids_with_wav(run_dir) if i in prompts and i in base_prompts], n_ids)
        if not ids:
            continue
        what = AXIS_TEXT.get(axis, manifest.get(axis, {}).get("target", axis))
        rows = []
        for clip_id in ids:
            original = assets.clip(f"dev/{BASE_RUN}", base_dir / f"{clip_id}.wav")
            edited = assets.clip(f"dev/{run_dir.name}", run_dir / f"{clip_id}.wav")
            original_text, edited_text = base_prompts[clip_id]["prompt"], prompts[clip_id]["prompt"]
            delta = pairs.get((axis, clip_id), {})
            delta_note = ""
            if delta:
                delta_note = ("probe change, edited minus original, target minus source (positive = as predicted): "
                              + ", ".join(f"{k.removeprefix('probe_')} {v:+.2f}" for k, v in sorted(delta.items())))
            rows.append(
                f'<div class="pair"><h4>id {clip_id}</h4><div class="two">'
                f'<div><div class="cap">original caption</div><blockquote>{marked(original_text, edited_text, "del")}</blockquote>'
                f'{card(BASE_RUN + " (original)", original)}</div>'
                f'<div><div class="cap">edited caption</div><blockquote>{marked(edited_text, original_text, "ins")}</blockquote>'
                f'{card(run_dir.name + " (edited)", edited, notes=[esc(delta_note)] if delta_note else ())}</div></div></div>')
        blocks.append(f'<section class="block" id="{esc(run_dir.name)}"><h3>{esc(run_dir.name)} <span class="kind">{esc(what)}</span></h3>{"".join(rows)}</section>')
    return "".join(blocks), len(blocks)


def bundle_section(assets, n_ids):
    base_dir = HW3 / "outputs/dev" / BASE_RUN
    base_prompts = prompts_of(base_dir)
    guidance = read_csv(ANALYSIS / "bundle_guidance.csv")
    with open(HW3 / "docs/edit_predictions.toml", "rb") as handle:
        predictions = {p["axis"]: p["verdict"] for p in tomllib.load(handle).get("bundle_prediction", [])}
    axes = sorted({p.name.split("_", 1)[1].rsplit("_", 1)[0] for p in (HW3 / "outputs/dev").glob("bundle_*_bundle")})
    blocks = []
    for axis in axes:
        arms = {arm: HW3 / "outputs/dev" / f"bundle_{axis}_{arm}" for arm in ("bundle", "neutral")}
        plain_dirs = [HW3 / "outputs/dev" / f"edit_{axis}", HW3 / "outputs/dev" / f"edit_{axis}_ext"]
        arm_info = {}
        for arm, folder in arms.items():
            info = json.loads((HW3 / "results" / f"dev__bundle_{axis}_{arm}" / "gen_info.json").read_text())["settings"]
            weight = info["guidance"]["extra_scale"]
            arm_info[arm] = (f"extra guidance term, weight {weight:+g} ({'away from' if weight < 0 else 'toward'}) the text "
                             f"\"{info['extra_prompt']}\"")
        prompts = prompts_of(arms["bundle"])
        usable = [i for i in prompts if all((arms[a] / f"{i}.wav").is_file() for a in arms) and i in base_prompts
                  and any((d / f"{i}.wav").is_file() for d in plain_dirs)]
        ids = pick_spread(usable, n_ids)
        rows = []
        for clip_id in ids:
            plain = next(d for d in plain_dirs if (d / f"{clip_id}.wav").is_file())
            edited_text = prompts[clip_id]["prompt"]
            cards = [
                card(BASE_RUN + " (original caption)", assets.clip(f"dev/{BASE_RUN}", base_dir / f"{clip_id}.wav")),
                card(plain.name + " (edited caption, plain)", assets.clip(f"dev/{plain.name}", plain / f"{clip_id}.wav")),
                card(f"bundle_{axis}_neutral", assets.clip(f"dev/bundle_{axis}_neutral", arms["neutral"] / f"{clip_id}.wav"),
                     notes=["control arm: " + esc(arm_info["neutral"])]),
                card(f"bundle_{axis}_bundle", assets.clip(f"dev/bundle_{axis}_bundle", arms["bundle"] / f"{clip_id}.wav"),
                     notes=["bundle arm: " + esc(arm_info["bundle"])]),
            ]
            rows.append(f'<div class="pair"><h4>id {clip_id}</h4><div class="cap">original caption</div>'
                        f'<blockquote>{marked(base_prompts[clip_id]["prompt"], edited_text, "del")}</blockquote>'
                        f'<div class="cap">edited caption (used by all three edited clips)</div><blockquote>{marked(edited_text, base_prompts[clip_id]["prompt"], "ins")}</blockquote>'
                        f'<div class="grid">{"".join(cards)}</div></div>')
        table = ['<details><summary>measured change of each arm against the plain edited clip (seed SD, 95% interval)</summary><div class="scroll"><table class="metrics"><thead>'
                 "<tr><th>probe</th><th>arm</th><th>measure</th><th>n</th><th>change [95%]</th><th>edit effect</th></tr></thead><tbody>"]
        for r in guidance:
            if r["axis"] == axis:
                table.append(f"<tr><td>{esc(r['kind'])}</td><td>{esc(r['arm'])}</td><td>{esc(r['measure'])}</td><td>{r['n']}</td>"
                             f"<td>{float(r['change_vs_edited_sd']):+.2f} [{float(r['ci_low_sd']):+.2f}, {float(r['ci_high_sd']):+.2f}]</td>"
                             f"<td>{float(r['edit_effect_sd']):+.2f}</td></tr>")
        table.append("</tbody></table></div></details>")
        verdict = predictions.get(axis)
        blocks.append(f'<section class="block" id="bundle-{esc(axis)}"><h3>bundle cut: {esc(AXIS_TEXT.get(axis, axis))}'
                      + (f' <span class="kind">prediction written before the clips: {esc(verdict)}</span>' if verdict else "")
                      + f"</h3>{''.join(table)}{''.join(rows)}</section>")
    return "".join(blocks)


def test_section(assets, scores, test_ids):
    cards_by_id = []
    for clip_id in test_ids:
        cards = []
        ref = assets.clip("test_retrieved", HW3 / f"outputs/ref/test_retrieved/{clip_id}.wav")
        if ref:
            cards.append(card("retrieved reference clip", ref))
        for run in TEST_RUNS:
            clip = assets.clip(f"test/{run}", HW3 / f"outputs/test/{run}/{clip_id}.wav")
            if clip:
                cards.append(card(run, clip, scores.get("test", run, clip_id)))
        cards_by_id.append(f'<section class="block" id="test-{clip_id}"><h3>test id {clip_id}</h3><div class="grid">{"".join(cards)}</div></section>')
    return "".join(cards_by_id)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HW3 samples: text-to-music with Stable Audio Open</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 1rem; line-height: 1.45; color: #1d2026; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
h2 {{ margin-top: 2.2rem; font-size: 1.25rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }}
h3 {{ font-size: 1.05rem; margin: 0 0 0.3rem; }}
h4 {{ font-size: 0.95rem; margin: 0.8rem 0 0.2rem; }}
nav {{ position: sticky; top: 0; background: #fff; padding: 0.4rem 0; border-bottom: 1px solid #ddd; z-index: 2; font-size: 0.9rem; }}
nav a {{ margin-right: 0.9rem; }}
code {{ font-size: 0.85em; background: #f1f2f4; padding: 0 0.25em; border-radius: 3px; }}
.meta, .where, .count {{ color: #666; font-size: 0.88rem; }}
.kind {{ font-weight: 400; font-size: 0.85rem; background: #eef2fb; color: #2d4a8a; padding: 0.05rem 0.5rem; border-radius: 10px; margin-left: 0.4rem; }}
blockquote {{ margin: 0.3rem 0 0.6rem; padding: 0.35rem 0.8rem; background: #f6f7f9; border-left: 3px solid #bbc; font-size: 0.92rem; }}
.cap {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #667; margin-top: 0.5rem; }}
mark.ins {{ background: #cdeccd; }}
mark.del {{ background: #f5cccc; }}
.block {{ margin: 1.2rem 0 2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 0.7rem; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }}
.clip {{ margin: 0; border: 1px solid #e3e5e9; border-radius: 6px; padding: 0.45rem 0.55rem; background: #fff; }}
.label {{ font-weight: 600; font-size: 0.88rem; word-break: break-all; }}
.scores {{ font-size: 0.82rem; color: #333; display: flex; gap: 0.8rem; flex-wrap: wrap; }}
.note {{ font-size: 0.78rem; color: #666; margin-top: 0.2rem; }}
audio {{ width: 100%; height: 34px; margin: 0.25rem 0; }}
.mel {{ position: relative; line-height: 0; margin-top: 0.3rem; cursor: pointer; }}
.mel img {{ width: 100%; height: 72px; display: block; }}
.mel .head {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #3cf; left: 0; display: none; }}
.scroll {{ overflow-x: auto; }}
table.metrics {{ border-collapse: collapse; font-size: 0.8rem; white-space: nowrap; }}
table.metrics th, table.metrics td {{ border-bottom: 1px solid #e5e7eb; padding: 0.2rem 0.55rem; text-align: right; }}
table.metrics th:first-child, table.metrics td:first-child {{ text-align: left; }}
table.metrics tr.group td {{ background: #eef2fb; font-weight: 600; text-align: left; }}
table.metrics tr.submission td {{ background: #fff6dd; font-weight: 600; }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; }}
.fig {{ margin: 0; }}
.fig img {{ width: 100%; border: 1px solid #e3e5e9; }}
.fig figcaption {{ font-size: 0.85rem; }}
dl.runs {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; font-size: 0.9rem; }}
dl.runs dt {{ font-weight: 600; }}
dl.runs dd {{ margin: 0; }}
dl.runs .set {{ color: #666; font-size: 0.8rem; }}
summary {{ cursor: pointer; font-size: 0.9rem; margin: 0.4rem 0; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #14161a; color: #dcdfe4; }}
  nav {{ background: #14161a; border-color: #333; }}
  h2 {{ border-color: #333; }}
  code {{ background: #262a31; }}
  a {{ color: #8ab4f8; }}
  .meta, .where, .count, .note, dl.runs .set {{ color: #9aa; }}
  blockquote {{ background: #1c1f25; border-color: #445; }}
  .clip {{ background: #181b20; border-color: #2c3038; }}
  .scores {{ color: #ccd; }}
  .kind {{ background: #222c44; color: #aac4f5; }}
  mark.ins {{ background: #24502b; color: #e8f5e8; }}
  mark.del {{ background: #5a2a2a; color: #f5e5e5; }}
  table.metrics tr.group td {{ background: #1f2638; }}
  table.metrics tr.submission td {{ background: #3a3418; }}
  table.metrics th, table.metrics td {{ border-color: #2c3038; }}
  .fig img {{ background: #fff; }}
}}
</style>
</head>
<body>
<header>
<h1>HW3 samples: text-to-music with Stable Audio Open</h1>
<p class="meta">Built {built}. Dev and test clips, non-blind: run names, settings and scores are shown next to every clip.</p>
<p>Every comparison is paired by caption and seed. Dev has 100 MusicCaps captions used for tuning; test has 100 held-out captions. Each clip is 10 s, 44.1 kHz stereo (here as MP3, {bitrate}).</p>
<p><b>Scores per clip:</b> <b>CLAP</b> is the cosine between the clip and its caption (scoring checkpoint, higher is better). <b>KL</b> is the PaSST class-distribution KL against the reference clip (lower is better; dev reference: the real MusicCaps clip, test reference: the retrieved clip). <b>MQAScore</b> is the music question-answering judge's mean over caption tags (higher is better; blank where it was not run). Per-clip scores are noisy; the paired intervals in the table are the evidence.</p>
<p><b>Mel spectrograms</b> are log-mel (128 bins, hop 512, low frequencies at the bottom), computed from the audio, on one fixed colour range ({mel_min:g} to {mel_max:g} dB) for every image. Click a mel to jump to that point and play.</p>
<h3>Run names</h3>
<dl class="runs">{run_names}</dl>
</header>
<nav><a href="#table">Metric table</a><a href="#graphs">Graphs ({n_figs})</a><a href="#dev">Dev samples</a><a href="#edits">Word-swap edits</a><a href="#bundles">Bundle cut</a><a href="#test">Test samples</a><a href="#skipped">Skipped</a></nav>
<main>
<h2 id="table">Metric table</h2>
<p class="meta">From <code>results/analysis/findings_table.md</code>. dCLAP and dKL are paired differences against the cfg7 run of the same set with 95% bootstrap intervals over prompts; FD-VGG mc and FD-MERT mc are Frechet distances against MusicCaps; clip % is the share of clipped samples; s/clip is generation time. {n_rows} rows. The submission row is highlighted.</p>
{table}
<h2 id="graphs">Graphs</h2>
<p class="meta">Every PNG in <code>results/analysis/figures</code> and <code>report/deck/figs_out</code> at build time ({n_figs}). Click to open full size.</p>
{figures}
<h2 id="dev">Dev samples</h2>
<p>{dev_intro}</p>
{dev}
<h2 id="edits">Word-swap edits</h2>
<p>Each pair has the same seed and settings; only the caption words differ (green: added or changed words, red: words in the original). The original clip is the <code>{base}</code> clip. Per-clip scoring was not run for the edit runs; the probe change line comes from <code>results/analysis/edit_pairs.csv</code> (MERT and CLAP-based attribute probes, raw probe units) where the id was analysed. Three ids per run, spread over the ids that were generated. {n_edit_runs} edit runs.</p>
{edits}
<h2 id="bundles">Bundle cut</h2>
<p>Mood and loudness words act through a bundle of drum and groove cues. Here the edited caption keeps its weight and one extra guidance term is added on the bundle text (the bundle arm) or on a meaning-free sentence (the neutral control). Predictions and measured changes: <code>docs/edit_predictions.toml</code>, <code>results/analysis/bundle_guidance.csv</code>. Three ids per axis.</p>
{bundles}
<h2 id="test">Test samples</h2>
<p>Six test ids, spaced evenly over the id range (ids {test_ids}). Test caption text is course material and is not shown here; clips are labelled by id only. The retrieved reference clip is the real clip retrieved for that caption, used as the KL reference.</p>
{test}
<h2 id="skipped">Skipped</h2>
{skipped}
</main>
<script>
document.addEventListener('play', e => document.querySelectorAll('audio').forEach(a => a !== e.target && a.pause()), true);
const melOf = a => a.closest('.clip').querySelector('.mel');
function place(a) {{
  const mel = melOf(a), head = mel.querySelector('.head');
  head.style.display = 'block';
  head.style.left = Math.min(100, 100 * a.currentTime / +mel.dataset.sec) + '%';
}}
function track(a) {{
  const step = () => {{ place(a); if (!a.paused) requestAnimationFrame(step); }};
  step();
}}
['play', 'playing', 'seeked', 'pause'].forEach(ev => document.addEventListener(ev, e => e.target.tagName === 'AUDIO' && track(e.target), true));
document.addEventListener('click', e => {{
  const mel = e.target.closest('.mel');
  if (!mel) return;
  const a = mel.closest('.clip').querySelector('audio');
  const r = mel.getBoundingClientRect(), f = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
  const seek = () => {{ a.currentTime = f * +mel.dataset.sec; track(a); }};
  if (a.readyState >= 1) {{ seek(); a.play(); }} else {{ a.addEventListener('loadedmetadata', seek, {{once: true}}); a.play(); }}
}});
</script>
</body>
</html>
"""


def run_names_html():
    items = []
    for run in DEV_RUNS:
        settings = run_settings(run)
        text = RUN_TEXT[run] + (f' <span class="set">[{esc(settings)}]</span>' if settings else "")
        items.append(f"<dt>{esc(run)}</dt><dd>{text}</dd>")
    for name, text in EDIT_GLOSSARY:
        items.append(f"<dt>{esc(name)}</dt><dd>{esc(text)}</dd>")
    return "".join(items)


def main():
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assets, scores = Assets(out), Scores()

    if args.dev_ids:
        dev_ids, reasons = [str(i) for i in args.dev_ids], {}
        how = "ids given on the command line."
    else:
        dev_ids, reasons, median_words = pick_dev_ids()
        how = ("picked by caption kind from <code>results/analysis/caption_classes_dev.csv</code>: two that ask for vocals, two that say there are no vocals, "
               "two with low-quality recording words, one mostly mood words, one with non-music sounds. Within a kind the id with the most runs present "
               f"wins, then the caption length closest to the dev median ({median_words:.0f} words). The kind of each id is shown next to its heading.")
    dev_html, skipped = dev_section(assets, scores, dev_ids, reasons)
    edits_html, n_edit_runs = edit_section(assets, args.edit_ids)
    bundles_html = bundle_section(assets, args.edit_ids)
    if args.test_ids:
        test_ids = [str(i) for i in args.test_ids]
    else:
        common = set.intersection(*(ids_with_wav(HW3 / "outputs/test" / run) for run in TEST_RUNS))
        test_ids = pick_spread(common, 6)
    test_html = test_section(assets, scores, test_ids)
    table_html, n_rows = findings_table_html()
    figures, n_figs = figures_html(out)
    n_clips = assets.build(args.bitrate, args.workers, args.force)

    skipped_html = ("<p>Runs with no file for a dev id are left out of that id's grid:</p><ul>"
                    + "".join(f"<li>dev id {i}: {esc(', '.join(runs))}</li>" for i, runs in skipped.items()) + "</ul>"
                    if skipped else "<p>No dev clip was skipped.</p>")
    skipped_html += ("<p>The edit runs <code>edit_*</code> and <code>bundle_*</code> have no per-clip CLAP, KL or MQAScore here because they were not scored per clip. "
                     "Runs that exist under <code>outputs/dev</code> but are not on this page (other seeds, scales and ranker variants) are in the metric table only.</p>")
    page = PAGE.format(
        built=datetime.now().strftime("%Y-%m-%d %H:%M"), bitrate=args.bitrate, mel_min=MEL_DB_MIN, mel_max=MEL_DB_MAX,
        run_names=run_names_html(), n_figs=n_figs, n_rows=n_rows, table=table_html, figures=figures,
        dev_intro=f"Eight dev ids ({', '.join(dev_ids)}), {how} Dev captions are public MusicCaps text.", dev=dev_html,
        base=BASE_RUN, n_edit_runs=n_edit_runs, edits=edits_html, bundles=bundles_html,
        test_ids=", ".join(test_ids), test=test_html, skipped=skipped_html)
    (out / "index.html").write_text(page)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{n_clips} clips, {n_figs} figures, dev ids {dev_ids}, test ids {test_ids}, {size / 1e6:.1f} MB -> {out / 'index.html'}")
    for clip_id, runs in skipped.items():
        print(f"skipped dev id {clip_id}: {', '.join(runs)}")


if __name__ == "__main__":
    main()
