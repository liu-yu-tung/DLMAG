import argparse
import json
import os
import subprocess
from pathlib import Path

HW3 = Path(__file__).resolve().parents[2]
CACHE = HW3 / "outputs/pred_cache"
PYTHON = HW3 / "external/av-benchmark/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description="Add the runs that have a feature cache but no entry yet to "
                                                 "results/analysis/paired_fd_<set>.json (scripts/evaluation/paired_fd.py, CPU, about 1-2 min per new run).")
    parser.add_argument("--set", dest="set_name", default="dev")
    parser.add_argument("--baseline", default="cfg7")
    args = parser.parse_args()
    target = HW3 / f"results/analysis/paired_fd_{args.set_name}.json"
    current = json.loads(target.read_text())
    assert current["baseline"] == args.baseline, current["baseline"]
    cached = sorted(p.name.split("__", 1)[1] for p in CACHE.glob(f"{args.set_name}__*")
                    if (p / "pann_features.pth").exists() and (HW3 / "results" / p.name / "metrics.json").exists())
    missing = [run for run in cached if run not in current["runs"] and run != "real"]
    if not missing:
        return
    partial = target.with_name(target.name + ".partial")
    subprocess.run([str(PYTHON), "scripts/evaluation/paired_fd.py", "--set", args.set_name, "--baseline", args.baseline,
                    "--runs", ",".join(missing), "--out", str(partial)], cwd=HW3, check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "3"})
    current = json.loads(target.read_text())
    current["runs"].update(json.loads(partial.read_text())["runs"])
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(current, indent=1) + "\n")
    temporary.replace(target)
    partial.unlink()
    print("added", ", ".join(missing))


if __name__ == "__main__":
    main()
