#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
mkdir -p runs/night_0929
log() { echo "$(date +%T) $*" | tee -a runs/night_0929/status.txt; }
run() {
    local name=$1; shift
    log "start $name"
    if timeout 3h "$@" > "runs/night_0929/$name.log" 2>&1; then log "done $name"; else log "FAILED $name (exit $?)"; fi
}
for k in A B; do run "stems_$k" uv run python scripts/extract_stems.py --dataset "$k"; done
for k in A B; do
    for s in vocals drums bass other; do run "mert_${k}_$s" uv run python scripts/extract_mert.py --dataset "$k" --stem "$s"; done
done
for k in A B; do run "mixbalance_$k" uv run python scripts/extract_mixbalance.py --dataset "$k"; done
log "all finished"
