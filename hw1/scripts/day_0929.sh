#!/usr/bin/env bash
# Daytime GPU chain, 9/29: log-mel for A, Short-Chunk CNN 5-fold (seed 0) on A and B for the fusion search,
# then full-train seeds 0-2 for the report curves. Every job is resumable and has its own timeout.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
D=runs/day_0929
mkdir -p $D
log() { echo "$(date +%T) $*" | tee -a $D/status.txt; }
run() { local name=$1; shift; log "start $name"; if timeout 90m "$@" > "$D/$name.log" 2>&1; then log "done $name"; else log "FAILED $name"; fi; }
[ -f features/A_logmel.npy ] || run logmel_A uv run python scripts/extract_logmel.py --dataset A --jobs 4
for k in A B; do for f in 0 1 2 3 4; do
    [ -f features/${k}_cnn_s0_f${f}_logp.npz ] || run cnn_${k}_f$f uv run python scripts/train_cnn.py --dataset $k --seed 0 --fold $f --resume
done; done
for k in A B; do for s in 0 1 2; do
    [ -f features/${k}_cnn_s${s}_logp.npz ] || run cnn_${k}_s$s uv run python scripts/train_cnn.py --dataset $k --seed $s --resume
done; done
log "all finished"
