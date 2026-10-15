"""MIDI -> WAV with the FluidSynth CLI (pacman: fluidsynth, soundfont-fluid)."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
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


def trim_leading_silence(wav_path: str | Path, lead_s: float = 0.2, floor_db: float = -50.0) -> float:
    rate, x = wavfile.read(wav_path)
    full = np.iinfo(x.dtype).max if np.issubdtype(x.dtype, np.integer) else 1.0
    loud = np.flatnonzero(np.abs(x.reshape(len(x), -1)).max(1) > full * 10 ** (floor_db / 20))
    cut = max(0, int(loud[0]) - int(lead_s * rate)) if len(loud) else 0
    if cut:
        wavfile.write(wav_path, rate, x[cut:])
    return cut / rate


def midi_to_wav(midi_path: str | Path, wav_path: str | Path, soundfont: str | None = None, sample_rate: int = 44100,
                trim: bool = True):
    soundfont = soundfont or default_soundfont()
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth not found; install it (Arch: pacman -S fluidsynth soundfont-fluid) or skip --wav")
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["fluidsynth", "-ni", "-q", "-F", str(wav_path), "-r", str(sample_rate), soundfont, str(midi_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # POP909-style pickup bars start with seconds of silence; the MIDI keeps its 32 bars, only the audio is trimmed
    if trim:
        trim_leading_silence(wav_path)
