#!/usr/bin/env bash
# Follow-up chain, 9/29: waits for day_0929.sh, then A 5-fold CNN variants on the same folds:
# ordinal soft targets (eps 0.1, fixed in advance) and no random gain (keeps absolute level as a decade cue).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
D=runs/day_0929
log() { echo "$(date +%T) $*" | tee -a $D/status.txt; }
run() { local name=$1; shift; log "start $name"; if timeout 90m "$@" > "$D/$name.log" 2>&1; then log "done $name"; else log "FAILED $name"; fi; }
while pgrep -f "scripts/day_0929.sh" > /dev/null; do sleep 60; done
for f in 0 1 2 3 4; do
    [ -f features/A_cnn_s0_f${f}_ord10_logp.npz ] || run cnn_A_f${f}_ord10 uv run python scripts/train_cnn.py --dataset A --seed 0 --fold $f --ordinal-eps 0.1 --tag _ord10 --resume
done
for f in 0 1 2 3 4; do
    [ -f features/A_cnn_s0_f${f}_nogain_logp.npz ] || run cnn_A_f${f}_nogain uv run python scripts/train_cnn.py --dataset A --seed 0 --fold $f --no-gain --tag _nogain --resume
done
log "chain b finished"
