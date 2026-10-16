"""MIDI -> WAV with the FluidSynth CLI (pacman: fluidsynth, soundfont-fluid), and the listening-build helpers
(velocity matching to POP909, loudness)."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.io import wavfile

# Arch: soundfont-fluid; Debian/Ubuntu: fluid-soundfont-gm puts it in /usr/share/sounds/sf2/. Override with $SOUNDFONT.
CANDIDATES = ["/usr/share/soundfonts/FluidR3_GM.sf2", "/usr/share/sounds/sf2/FluidR3_GM.sf2"]


def default_soundfont() -> str:
    if os.environ.get("SOUNDFONT"):
        return os.environ["SOUNDFONT"]
    found = next((p for p in CANDIDATES if Path(p).exists()), None)
    if found is None:
        raise RuntimeError("no soundfont found; set SOUNDFONT=/path/to/file.sf2")
    return found


def first_sound(wav_path: str | Path, floor_db: float = -50.0) -> float:
    rate, x = wavfile.read(wav_path)
    full = np.iinfo(x.dtype).max if np.issubdtype(x.dtype, np.integer) else 1.0
    loud = np.flatnonzero(np.abs(x.reshape(len(x), -1)).max(1) > full * 10 ** (floor_db / 20))
    return loud[0] / rate if len(loud) else 0.0


def trim_leading_silence(wav_path: str | Path, lead_s: float = 0.2) -> float:
    cut = max(0.0, first_sound(wav_path) - lead_s)
    if cut:
        rate, x = wavfile.read(wav_path)
        wavfile.write(wav_path, rate, x[int(cut * rate):])
    return cut


def midi_to_wav(midi_path: str | Path, wav_path: str | Path, soundfont: str | None = None, sample_rate: int = 44100,
                trim: bool = True):
    """Renders the MIDI played as one keyboard (one_keyboard); the MIDI file itself is not changed."""
    soundfont = soundfont or default_soundfont()
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth not found; install it (Arch: pacman -S fluidsynth soundfont-fluid) or skip --wav")
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    one_keyboard(midi)
    with tempfile.TemporaryDirectory() as tmp:
        played = Path(tmp) / "played.mid"
        midi.write(str(played))
        subprocess.run(
            ["fluidsynth", "-ni", "-q", "-F", str(wav_path), "-r", str(sample_rate), soundfont, str(played)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    # POP909-style pickup bars start with seconds of silence; the MIDI keeps its 32 bars, only the audio is trimmed
    if trim:
        trim_leading_silence(wav_path)


# Plan-fill PIANO is POP909 PIANO + BRIDGE (hw2.plan); single-track pieces merge every POP909 track.
REAL_PARTS = {"MELODY": ["MELODY"], "PIANO": ["PIANO", "BRIDGE"], "ALL": ["MELODY", "BRIDGE", "PIANO"]}


def track_role(midi: pretty_midi.PrettyMIDI, inst: pretty_midi.Instrument) -> str:
    return inst.name.upper() if any(i.name.upper() == "MELODY" for i in midi.instruments) else "ALL"


def real_velocity_counts(cache: Path) -> dict[str, np.ndarray]:
    if not cache.exists():
        from hw2.remi import DATA_ROOT
        counts = {name: np.zeros(128) for name in ["MELODY", "BRIDGE", "PIANO"]}
        for mid in sorted((DATA_ROOT / "training").glob("*/*.mid")):
            for inst in pretty_midi.PrettyMIDI(str(mid)).instruments:
                if inst.name.upper() in counts:
                    np.add.at(counts[inst.name.upper()], [n.velocity for n in inst.notes], 1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({k: v.tolist() for k, v in counts.items()}))
    raw = {k: np.array(v) for k, v in json.loads(cache.read_text()).items()}
    return {role: sum(raw[p] for p in parts) for role, parts in REAL_PARTS.items()}


def velocity_maps(midis: list[pretty_midi.PrettyMIDI], real: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per track role, a 128-entry table that quantile-maps the pooled velocities of these pieces onto the real ones."""
    pooled = {}
    for midi in midis:
        for inst in midi.instruments:
            np.add.at(pooled.setdefault(track_role(midi, inst), np.zeros(128)), [n.velocity for n in inst.notes], 1)
    maps = {}
    for role, counts in pooled.items():
        cdf = np.cumsum(counts) / counts.sum()
        mid_rank = cdf - counts / counts.sum() / 2
        real_cdf = np.cumsum(real[role]) / real[role].sum()
        maps[role] = np.clip(np.searchsorted(real_cdf, mid_rank), 1, 127)
    return maps


def apply_velocity_map(midi: pretty_midi.PrettyMIDI, maps: dict[str, np.ndarray]) -> pretty_midi.PrettyMIDI:
    for inst in midi.instruments:
        table = maps[track_role(midi, inst)]
        for n in inst.notes:
            n.velocity = int(table[n.velocity])
    return midi


def loudness_lufs(wav_path: str | Path) -> float:
    err = subprocess.run(["ffmpeg", "-nostats", "-i", str(wav_path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def one_keyboard(midi: pretty_midi.PrettyMIDI) -> dict[str, int]:
    """A key cannot sound twice: per pitch across all tracks, drop a note struck at the same time as another
    (keep the louder, melody on ties) and end a sounding note where the next one on that key starts."""
    # FluidSynth ends both notes at the first note-off of a key on one channel, and doubles the level across channels
    rank = {inst.name.upper(): k for k, inst in enumerate(midi.instruments)}
    by_pitch = {}
    for inst in midi.instruments:
        for n in inst.notes:
            by_pitch.setdefault(n.pitch, []).append((n, inst))
    drop, cut = set(), 0
    for notes in by_pitch.values():
        notes.sort(key=lambda x: (x[0].start, -x[0].velocity, x[1].name.upper() != "MELODY", rank[x[1].name.upper()]))
        kept = []
        for n, inst in notes:
            if kept and abs(n.start - kept[-1].start) < 1e-3:
                drop.add(id(n))
                continue
            if kept and kept[-1].end > n.start:
                kept[-1].end = n.start
                cut += 1
            kept.append(n)
    for inst in midi.instruments:
        inst.notes = [n for n in inst.notes if id(n) not in drop]
    return {"dropped": len(drop), "shortened": cut}
