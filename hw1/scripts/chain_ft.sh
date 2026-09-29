#!/usr/bin/env bash
# Fine-tuning chain with a GPU logger and cool-downs (2026-09-30). Jobs run in order; after each training job the GPU
# rests until it is at COOL_TEMP or below (at least COOL_MIN s, at most COOL_MAX s). If the logger saw HOT_TEMP or a
# thermal slowdown during a job, the rest is HOT_REST s instead. Jobs are never killed; finetune_mert.py resumes per fold.
# Overrides for a dry run: JOBS (one job per line), COOL_MIN, COOL_MAX, COOL_TEMP, HOT_TEMP, HOT_REST, LOG_EVERY, LOG.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=${LOG:-../runs/night_ft}
COOL_MIN=${COOL_MIN:-300} COOL_MAX=${COOL_MAX:-600} COOL_TEMP=${COOL_TEMP:-55}
HOT_TEMP=${HOT_TEMP:-87} HOT_REST=${HOT_REST:-1200} LOG_EVERY=${LOG_EVERY:-60}
JOBS=${JOBS:-"--dataset B --from-layer 16 --seed 1
--dataset B --from-layer 16 --seed 2
--dataset A --from-layer 16 --seed 2
--dataset A --cache 12
--dataset A --from-layer 12 --seed 0
--dataset B --cache 12
--dataset B --from-layer 12 --seed 0"}
mkdir -p "$LOG"
S=$LOG/status.txt
GPU=$LOG/gpu.csv
note() { echo "$(date +%T) $*" >> "$S"; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }

nvidia-smi --query-gpu=timestamp,temperature.gpu,fan.speed,power.draw,clocks.sm,memory.used,utilization.gpu,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
    --format=csv -l "$LOG_EVERY" >> "$GPU" &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT
note "chain start (logger pid $LOGGER, cool-down to ${COOL_TEMP}C, ${COOL_MIN}-${COOL_MAX}s)"

while IFS= read -r job; do
    [ -z "$job" ] && continue
    start_line=$(wc -l < "$GPU")
    note "start $job"
    uv run python finetune_mert.py $job > "$LOG/run_$(echo "$job" | tr ' -' '__').log" 2>&1
    note "exit $? $job"
    case "$job" in *--cache*) continue ;; esac
    peak=$(tail -n +"$((start_line + 1))" "$GPU" | awk -F', ' '$2 ~ /^[0-9]+$/ && $2 > m {m = $2} END {print m + 0}')
    slow=$(tail -n +"$((start_line + 1))" "$GPU" | awk -F', ' '$8 == "Active" || $9 == "Active"' | wc -l)
    if [ "$peak" -ge "$HOT_TEMP" ] || [ "$slow" -gt 0 ]; then
        note "HOT peak ${peak}C, slowdown rows $slow: resting ${HOT_REST}s"
        sleep "$HOT_REST"
        continue
    fi
    t0=$(temp) waited=0
    while [ "$waited" -lt "$COOL_MAX" ] && { [ "$waited" -lt "$COOL_MIN" ] || [ "$(temp)" -gt "$COOL_TEMP" ]; }; do
        sleep 10
        waited=$((waited + 10))
    done
    note "cool-down ${waited}s: ${t0}C -> $(temp)C (job peak ${peak}C)"
done <<< "$JOBS"
note "chain done"
