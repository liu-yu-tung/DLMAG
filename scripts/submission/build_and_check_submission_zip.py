"""Build and verify the NTU COOL submission zip (spec: Homework_3.pdf, page "Submission"). Usage notes: docs/readme_draft.txt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CLIP_COUNT = 100
CLIP_SECONDS = 10.0
DURATION_TOLERANCE = 0.1
SAMPLE_RATE = 44100
CHANNELS = 2
MUSICCAPS_DIR = PROJECT / "data/musiccaps/audio_mp3"
CFG_SOURCE = PROJECT / "text_to_music/cfg.py"
RETRIEVAL_SOURCE = PROJECT / "results/retrieval.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def encode(wav: Path, mp3: Path, bitrate: str) -> None:
    # bitexact + no metadata: the same wav gives the same mp3 bytes
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-map_metadata", "-1", "-fflags", "+bitexact",
         "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)],
        check=True,
    )


def probe(path: Path) -> dict:
    out = run_text(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                    "stream=sample_rate,channels:format=duration", "-of", "json", str(path)])
    data = json.loads(out)
    stream = data["streams"][0]
    return {"sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"]),
            "duration": float(data["format"]["duration"])}


def decode_errors(path: Path) -> str:
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                            capture_output=True, text=True)
    return (result.stderr + ("" if result.returncode == 0 else f"exit {result.returncode}")).strip()


def build(run_dir: Path, student_dir: Path, out_root: Path, bitrate: str) -> None:
    wavs = [run_dir / f"{index}.wav" for index in range(CLIP_COUNT)]
    missing = [wav.name for wav in wavs if not wav.is_file()]
    if missing:
        raise SystemExit(f"{run_dir} is missing {len(missing)} wav files, first: {missing[:5]}")
    if student_dir.exists():
        shutil.rmtree(student_dir)
    samples = student_dir / "generated_samples"
    samples.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: encode(wavs[index], samples / f"{index}.mp3", bitrate), range(CLIP_COUNT)))
    shutil.copyfile(CFG_SOURCE, student_dir / "cfg.py")
    shutil.copyfile(RETRIEVAL_SOURCE, student_dir / "retrieval.json")
    zip_path = out_root / f"{student_dir.name}.zip"
    zip_path.unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-r", f"./{student_dir.name}.zip", f"./{student_dir.name}", "-x", "**/.*"],
                   cwd=out_root, check=True)


def check_clips(student_dir: Path, failures: list[str]) -> dict:
    samples = student_dir / "generated_samples"
    names = sorted(path.name for path in samples.iterdir())
    expected = sorted(f"{index}.mp3" for index in range(CLIP_COUNT))
    if names != expected:
        failures.append(f"generated_samples holds {len(names)} files, expected exactly 0.mp3..99.mp3")
    durations, rates, channel_counts = [], set(), set()
    for name in expected:
        path = samples / name
        if not path.is_file():
            continue
        try:
            info = probe(path)
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as error:
            failures.append(f"{name}: ffprobe failed ({error})")
            continue
        problems = decode_errors(path)
        if problems:
            failures.append(f"{name}: decode problems: {problems[:200]}")
        if abs(info["duration"] - CLIP_SECONDS) > DURATION_TOLERANCE:
            failures.append(f"{name}: duration {info['duration']:.3f} s")
        durations.append(info["duration"])
        rates.add(info["sample_rate"])
        channel_counts.add(info["channels"])
    if rates != {SAMPLE_RATE}:
        failures.append(f"sample rates {sorted(rates)}, expected only {SAMPLE_RATE}")
    if channel_counts != {CHANNELS}:
        failures.append(f"channel counts {sorted(channel_counts)}, expected only {CHANNELS}")
    return {"durations": durations, "sample_rates": sorted(rates), "channels": sorted(channel_counts)}


def check_retrieval(student_dir: Path, failures: list[str]) -> dict:
    data = json.loads((student_dir / "retrieval.json").read_text())
    wanted_keys = {str(index) for index in range(CLIP_COUNT)}
    if not isinstance(data, dict) or set(data) != wanted_keys:
        failures.append("retrieval.json keys are not exactly the strings '0'..'99'")
        return {"unique_values": None}
    stems = {path.stem for path in MUSICCAPS_DIR.glob("*.mp3")}
    for key, value in data.items():
        if not isinstance(value, str):
            failures.append(f"retrieval.json[{key}] is not a string")
        elif value not in stems:
            failures.append(f"retrieval.json[{key}] = {value!r} is not a stem in {MUSICCAPS_DIR}")
    return {"unique_values": len(set(data.values())), "musiccaps_files": len(stems)}


def check_cfg(student_dir: Path, failures: list[str]) -> None:
    sys.dont_write_bytecode = True  # a __pycache__ in the folder would end up in a later zip
    spec = importlib.util.spec_from_file_location("submitted_cfg", student_dir / "cfg.py")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = module.cfg(1.0, 3.0, 7.0)
    except Exception as error:
        failures.append(f"cfg.py: {type(error).__name__}: {error}")
        return
    if abs(value - (1.0 + 7.0 * 2.0)) > 1e-9:
        failures.append(f"cfg(1, 3, 7) = {value}, expected 15")
    if (student_dir / "cfg.py").is_symlink():
        failures.append("cfg.py is a symlink")


def check_zip(zip_path: Path, student_id: str, student_dir: Path, failures: list[str]) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            failures.append("zip failed its CRC test")
        files = [name for name in archive.namelist() if not name.endswith("/")]
        dotfiles = [name for name in archive.namelist() if any(part.startswith(".") for part in name.split("/"))]
        wrong_root = [name for name in files if name.split("/")[0] != student_id]
        expected = {f"{student_id}/cfg.py", f"{student_id}/retrieval.json"}
        expected |= {f"{student_id}/generated_samples/{index}.mp3" for index in range(CLIP_COUNT)}
        if dotfiles:
            failures.append(f"zip holds dotfiles: {dotfiles[:5]}")
        if wrong_root:
            failures.append(f"zip entries outside {student_id}/: {wrong_root[:5]}")
        if set(files) != expected or len(files) != len(expected):
            failures.append(f"zip files differ from the expected {len(expected)}: has {len(files)}")
        for name in sorted(expected & set(files)):
            if archive.read(name) != (student_dir.parent / name).read_bytes():
                failures.append(f"zip member differs from the folder file: {name}")
    return {"files": len(files), "bytes": zip_path.stat().st_size}


def write_manifest(out_root: Path, student_id: str, run_dir: Path, student_dir: Path, bitrate: str) -> Path:
    git_dirty = bool(run_text(["git", "status", "--porcelain"], cwd=PROJECT))
    manifest = {
        "student_id": student_id,
        "source_run_dir": str(run_dir),
        "bitrate": bitrate,
        "code_version": run_text(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT),
        "git_dirty": git_dirty,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cfg_py_sha256": sha256_of(student_dir / "cfg.py"),
        "retrieval_json_sha256": sha256_of(student_dir / "retrieval.json"),
        "clips": {
            str(index): {
                "wav_sha256": sha256_of(run_dir / f"{index}.wav"),
                "mp3_sha256": sha256_of(student_dir / "generated_samples" / f"{index}.mp3"),
            }
            for index in range(CLIP_COUNT)
        },
    }
    path = out_root / f"{student_id}_manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify <student-id>.zip for NTU COOL.")
    parser.add_argument("--run-dir", type=Path, required=True, help="folder with 0.wav..99.wav")
    parser.add_argument("--student-id", default="b10901114")
    parser.add_argument("--out-root", type=Path, default=PROJECT / "outputs/submission")
    parser.add_argument("--bitrate", default="192k")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_root = args.out_root.resolve()
    student_dir = out_root / args.student_id
    out_root.mkdir(parents=True, exist_ok=True)

    build(run_dir, student_dir, out_root, args.bitrate)
    failures: list[str] = []
    clips = check_clips(student_dir, failures)
    retrieval = check_retrieval(student_dir, failures)
    check_cfg(student_dir, failures)
    zip_path = out_root / f"{args.student_id}.zip"
    archive = check_zip(zip_path, args.student_id, student_dir, failures)
    manifest_path = write_manifest(out_root, args.student_id, run_dir, student_dir, args.bitrate)

    durations = clips["durations"]
    print(f"run dir        {run_dir}")
    print(f"zip            {zip_path} ({archive['bytes']} bytes, {archive['files']} files)")
    print(f"manifest       {manifest_path}")
    print(f"mp3 clips      {len(durations)} decoded, bitrate {args.bitrate}")
    if durations:
        print(f"mp3 duration   min {min(durations):.4f} s, max {max(durations):.4f} s")
    print(f"sample rates   {clips['sample_rates']}  channels {clips['channels']}")
    print(f"retrieval      unique values {retrieval['unique_values']}, musiccaps files {retrieval.get('musiccaps_files')}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
