#!/usr/bin/env bash
# MERT-v2 top-block fine-tuning, night of 2026-09-29: A first (top 4 and top 8 blocks, seeds 0 and 1), then B.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=../runs/night_ft
mkdir -p $LOG
run() {
    echo "$(date +%T) start $*" >> $LOG/status.txt
    uv run python finetune_mert.py "$@" > "$LOG/$(echo "$@" | tr ' -' '_').log" 2>&1
    echo "$(date +%T) exit $? $*" >> $LOG/status.txt
}
run --dataset A --from-layer 20 --seed 0
run --dataset A --from-layer 16 --seed 0
run --dataset A --from-layer 20 --seed 1
run --dataset A --from-layer 16 --seed 1
run --dataset B --cache 16 20
run --dataset B --from-layer 20 --seed 0
run --dataset B --from-layer 16 --seed 0
echo "$(date +%T) all done" >> $LOG/status.txt
