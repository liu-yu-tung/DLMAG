import argparse
import fcntl
import json
import os
import subprocess
import threading
import time
import tomllib
from pathlib import Path

HW3 = Path(__file__).resolve().parents[2]
STATE = HW3 / "logs/queue"
GPU_FIELDS = "temperature.gpu,utilization.gpu,memory.used,clocks.sm,power.draw,clocks_throttle_reasons.active"
GPU_KEYS = ["temperature", "utilization", "memory_used", "clock_sm", "power", "throttle"]
LANES = ("gpu", "side")


def gpu_reading() -> dict | None:
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    values = [v.strip() for v in result.stdout.strip().split(",")]
    if result.returncode != 0 or len(values) != len(GPU_KEYS):
        return None
    reading = dict(zip(GPU_KEYS, values))
    # Driver 620 prints fractional temperatures ("66.95"), so parse floats, not digit strings.
    try:
        reading["temperature"], reading["memory_used"] = float(reading["temperature"]), float(reading["memory_used"])
    except ValueError:
        return None
    return reading


def gpu_temperature() -> int:
    while (reading := gpu_reading()) is None:
        time.sleep(30)
    return int(reading["temperature"])


class GpuLogger(threading.Thread):
    def __init__(self, job_name: str, period: float = 30.0):
        super().__init__(daemon=True)
        self.job_name, self.period, self.peak, self.stop_event = job_name, period, 0, threading.Event()

    def run(self):
        with open(HW3 / "logs/gpu.csv", "a") as handle:
            while not self.stop_event.is_set():
                reading = gpu_reading()
                if reading is not None:
                    self.peak = max(self.peak, int(reading["temperature"]))
                    handle.write(",".join([time.strftime("%Y-%m-%d %H:%M:%S"), self.job_name, *map(str, reading.values())]) + "\n")
                    handle.flush()
                self.stop_event.wait(self.period)


def write_atomic(path: Path, text: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    records = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[record.get("id", len(records))] = record
    return list(records.values())


class QueueFile:
    def __init__(self, path: Path, log):
        self.path, self.log, self.last, self.last_error = path, log, None, None

    def load(self) -> tuple[dict, list[dict]]:
        try:
            config = tomllib.loads(self.path.read_text())
            jobs = config.pop("job", [])
            names = [job["name"] for job in jobs]
            assert len(names) == len(set(names)), "duplicate job names in queue"
            for job in jobs:
                assert isinstance(job["command"], list), f"{job['name']}: command must be a list"
                assert job.get("lane", "gpu") in LANES, f"{job['name']}: lane must be one of {LANES}"
                for dependency in job.get("after", []):
                    assert dependency in names, f"{job['name']} waits for unknown job {dependency}"
            self.last, self.last_error = (config, jobs), None
        except (OSError, tomllib.TOMLDecodeError, AssertionError, KeyError) as error:
            if self.last is None:
                raise
            if repr(error) != self.last_error:
                self.log(f"queue.toml unreadable, keeping the last good version: {error!r}")
                self.last_error = repr(error)
        return self.last


def marker(name: str, state: str) -> dict | None:
    path = STATE / f"{name}.{state}"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def retries(job: dict) -> int:
    return job.get("retries", 1 if job.get("kind") == "generate" else 0)


def job_state(job: dict) -> str:
    name = job["name"]
    for state in ("done", "running"):
        if (STATE / f"{name}.{state}").exists():
            return state
    failed = marker(name, "failed")
    if failed is not None:
        return "retry" if failed.get("attempts", 1) <= retries(job) else "failed"
    return "waiting"


def is_runnable(job: dict, by_name: dict[str, dict]) -> bool:
    return job_state(job) in ("waiting", "retry") and all(
        job_state(by_name[dependency]) == "done" for dependency in job.get("after", []))


def progress(job: dict) -> tuple[str, float | None]:
    path = job.get("progress_file")
    if not path or not (HW3 / path).exists():
        return "", None
    records = read_jsonl(HW3 / path)
    if not records:
        return "", None
    total = job.get("total", 100)
    times = sorted(r["time"] for r in records if "time" in r)
    if len(times) >= 2:
        seconds = (times[-1] - times[0]) / (len(times) - 1)
    else:
        seconds = sum(r["seconds"] for r in records) / len(records)
    remaining = seconds * max(total - len(records), 0) / 60
    text = f"{len(records)}/{total}, {seconds:.1f} s/item, job ETA {time.strftime('%H:%M', time.localtime(time.time() + 60 * remaining))}"
    if "clap_quick" in records[0]:
        text += f", quick CLAP {sum(r['clap_quick'] for r in records) / len(records):.3f}"
    if "clip_fraction" in records[0]:
        text += f", clipped {100 * sum(r['clip_fraction'] for r in records) / len(records):.3f}%"
    return text, remaining


def write_status(jobs: list[dict], running: dict[str, dict]):
    by_name = {job["name"]: job for job in jobs}
    durations = {}
    for job in jobs:
        done = marker(job["name"], "done")
        if done and "minutes" in done:
            durations.setdefault(job.get("kind", "other"), []).append(done["minutes"])
    remaining = {lane: 0.0 for lane in LANES}
    lines = [f"# HW3 queue status (updated {time.strftime('%m-%d %H:%M')})", "",
             "| job | lane | kind | state | minutes | peak C | note |", "|---|---|---|---|---|---|---|"]
    for job in jobs:
        name, kind, lane = job["name"], job.get("kind", "other"), job.get("lane", "gpu")
        state = job_state(job)
        minutes, peak, note = "", "", ""
        known = durations.get(kind)
        expected = (sum(known) / len(known)) if known else job.get("estimate_minutes", 0)
        if state in ("done", "failed", "retry"):
            record = marker(name, "done" if state == "done" else "failed") or {}
            minutes, peak = f"{record.get('minutes', 0):.1f}", str(record.get("peak_temperature", ""))
        if state == "running":
            note, left = progress(job)
            started = (marker(name, "running") or {}).get("epoch", time.time())
            remaining[lane] += left if left is not None else max(expected - (time.time() - started) / 60, 0)
        elif state in ("waiting", "retry"):
            pending = [d for d in job.get("after", []) if job_state(by_name[d]) != "done"]
            failed = [d for d in pending if job_state(by_name[d]) == "failed"]
            note = f"blocked by failed {', '.join(failed)}" if failed else (f"after {', '.join(pending)}" if pending else "")
            if not failed:
                remaining[lane] += expected
        lines.append(f"| {name} | {lane} | {kind} | {state} | {minutes} | {peak} | {note} |")
    left = max(remaining.values())
    finish = time.strftime("%m-%d %H:%M", time.localtime(time.time() + 60 * left))
    lines += ["", f"Queue ETA: about {left / 60:.1f} h of work left, done about {finish} "
              "(lanes run in parallel; per-kind mean of finished jobs, else estimate_minutes)."]
    for lane, info in running.items():
        lines.append(f"Running on `{lane}`: `{info['name']}` since {info['started']}.")
    write_atomic(HW3 / "results/status.md", "\n".join(lines) + "\n")


def wait_for_cool(limit: int, log, max_memory_used: int = 3000):
    while True:
        reading = gpu_reading()
        if reading is None:
            time.sleep(30)
            continue
        temperature, memory_used = int(reading["temperature"]), int(float(reading["memory_used"]))
        # Another process (an agent's model) may hold the GPU; a second big model would run out of memory.
        if temperature <= limit and memory_used <= max_memory_used:
            return
        log(f"GPU {temperature} C (limit {limit}), {memory_used} MiB used (limit {max_memory_used}), waiting")
        time.sleep(60)


def run_job(job: dict, config: dict, log, post_lock: threading.Lock):
    name, lane = job["name"], job.get("lane", "gpu")
    previous = marker(name, "failed")
    attempt = (previous or {}).get("attempts", 0) + 1
    write_atomic(STATE / f"{name}.running", json.dumps({"started": time.strftime("%m-%d %H:%M"), "epoch": time.time(),
                                                         "attempt": attempt, "phase": "cooling"}))
    (STATE / f"{name}.failed").unlink(missing_ok=True)
    if lane == "gpu":
        wait_for_cool(config.get("cool_temperature", 75), log, config.get("max_memory_used_mib", 3000))
    started = time.strftime("%m-%d %H:%M")
    write_atomic(STATE / f"{name}.running", json.dumps({"started": started, "epoch": time.time(), "attempt": attempt}))
    gpu_logger = GpuLogger(name) if lane == "gpu" else None
    if gpu_logger:
        gpu_logger.start()
    env = {**os.environ, **{k: str(v) for k, v in config.get("env", {}).items()},
           **{k: str(v) for k, v in job.get("env", {}).items()}}
    start = time.time()
    log(f"start {name} (lane {lane}, attempt {attempt}): {' '.join(job['command'])}")
    try:
        with open(STATE / f"{name}.log", "a") as job_log:
            job_log.write(f"=== attempt {attempt}, {started}\n")
            job_log.flush()
            code = subprocess.run(job["command"], cwd=HW3, stdout=job_log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=env).returncode
    except OSError as error:
        log(f"could not start {name}: {error!r}")
        code = -1
    peak = 0
    if gpu_logger:
        gpu_logger.stop_event.set()
        gpu_logger.join()
        peak = gpu_logger.peak
    record = {"minutes": (time.time() - start) / 60, "exit_code": code, "started": started, "attempts": attempt,
              "finished": time.strftime("%m-%d %H:%M"), "peak_temperature": peak}
    write_atomic(STATE / f"{name}.{'done' if code == 0 else 'failed'}", json.dumps(record))
    (STATE / f"{name}.running").unlink(missing_ok=True)
    log(f"{'done' if code == 0 else 'FAILED'} {name} in {record['minutes']:.1f} min, peak {peak} C")
    if code == 0 and config.get("post_command"):
        with post_lock:
            try:
                subprocess.run(config["post_command"], cwd=HW3, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=300)
            except subprocess.TimeoutExpired:
                log(f"post_command timed out after {name}")


def main():
    parser = argparse.ArgumentParser(description="Run hw3 jobs from queue.toml: one job per lane (gpu, side) at a time; "
                                                 "re-reads the file every loop. Touch logs/queue/STOP to stop after the running jobs.")
    parser.add_argument("--queue", type=Path, default=HW3 / "queue.toml")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock_handle = open(STATE / "lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another queue runner holds logs/queue/lock; exiting", flush=True)
        return
    for stale in STATE.glob("*.running"):
        stale.unlink()

    def log(message: str):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(STATE / "queue.log", "a") as handle:
            handle.write(line + "\n")

    log("queue runner started")
    queue_file = QueueFile(args.queue, log)
    slots: dict[str, tuple[threading.Thread, dict] | None] = {lane: None for lane in LANES}
    post_lock = threading.Lock()
    stopping = False
    while True:
        config, jobs = queue_file.load()
        by_name = {job["name"]: job for job in jobs}
        stopping = stopping or (STATE / "STOP").exists()
        for lane in LANES:
            if slots[lane] and not slots[lane][0].is_alive():
                slots[lane] = None
            if slots[lane] is None and not stopping:
                candidates = [job for job in jobs if job.get("lane", "gpu") == lane and is_runnable(job, by_name)]
                if candidates:
                    thread = threading.Thread(target=run_job, args=(candidates[0], config, log, post_lock), daemon=True)
                    slots[lane] = (thread, {"name": candidates[0]["name"], "started": time.strftime("%H:%M")})
                    thread.start()
        try:
            write_status(jobs, {lane: slot[1] for lane, slot in slots.items() if slot})
        except (OSError, ValueError, KeyError) as error:
            log(f"status write failed: {error!r}")
        if stopping and all(slot is None for slot in slots.values()):
            break
        time.sleep(15)
    (STATE / "STOP").unlink(missing_ok=True)
    log("STOP file found, exiting")


if __name__ == "__main__":
    main()
