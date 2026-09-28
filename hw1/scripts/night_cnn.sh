#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
mkdir -p runs/night_cnn
log() { echo "$(date +%T) $*" | tee -a runs/night_cnn/status.txt; }
[ -f features/A_logmel.npy ] || uv run python scripts/extract_logmel.py --dataset A --jobs 4 > runs/night_cnn/logmel_A.log 2>&1
[ -f features/B_logmel.npy ] || uv run python scripts/extract_logmel.py --dataset B --jobs 4 > runs/night_cnn/logmel_B.log 2>&1
for k in A B; do
    for s in 0 1 2; do
        log "start cnn_${k}_s$s"
        if timeout 2h uv run python scripts/train_cnn.py --dataset "$k" --seed "$s" --resume > "runs/night_cnn/cnn_${k}_s$s.log" 2>&1; then log "done cnn_${k}_s$s"; else log "FAILED cnn_${k}_s$s"; fi
    done
done
log "all finished"
